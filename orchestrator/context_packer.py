"""上下文打包器（spec v4 §8 灵魂模块）—— 阶段 4c 完整版。

四个来源（spec §8.1）全部就位：

1. **任务主题**：``tasks.title``
2. **接力点原文**：``handoff_conversation_id`` + ``handoff_turn_range`` → ``transcript``
3. **上游产出（精确）**：按 ``node.input_memory_ids`` 顺序 ``get_by_ids``，与
   ``depends_on`` 一一对应；上游 ``skipped`` 显式注明
4. **语义补充检索**：query 按 spec §8.2 拼装（``title + sub_task + 上游摘要``，
   每条 ≤ 50 字），总长 ≤ 200 token，超出按 30/20/0 字阶梯截断；
   强过滤 ``where={task_id, status=active}``，去重已在 input_memory_ids 的；
   排序按 ``memory_level``（``task_conclusion`` 优先，spec §8.3）+ 距离

最终输出受 **token budget**（默认 2K token，spec §8.2）约束：

- 必保：``task.title`` / 接力原文 / 上游产出 / 子任务说明
- 可裁：语义补充记忆（按相关度从低到高裁，相关度低的先被丢弃）
- 必保段已超 budget 时硬截接力原文，再丢语义补充

底线：``task.title`` + ``子任务`` 永远保留。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import tiktoken

from storage.memory_store import MemoryStore
from storage.state_store import DagNodeRow, StateStore, TaskRow
from storage.transcript_store import TranscriptStore

_log = logging.getLogger(__name__)

DEFAULT_MAX_CONTEXT_TOKENS = 2000
DEFAULT_MAX_QUERY_TOKENS = 200
DEFAULT_SEMANTIC_K = 3
UPSTREAM_SUMMARY_CHARS = 50  # spec §8.2

_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_get_encoder().encode(text))


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    enc = _get_encoder()
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])


@dataclass
class PackedContext:
    text: str
    handoff_present: bool
    upstream_present: int
    upstream_missing: int
    semantic_added: int
    semantic_dropped_for_budget: int
    query_used: str
    token_count: int


class ContextPacker:
    def __init__(
        self,
        *,
        state_store: StateStore,
        transcript_store: TranscriptStore,
        memory_store: MemoryStore,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        max_query_tokens: int = DEFAULT_MAX_QUERY_TOKENS,
        semantic_k: int = DEFAULT_SEMANTIC_K,
    ) -> None:
        self._state = state_store
        self._transcript = transcript_store
        self._memory = memory_store
        self._max_context_tokens = max_context_tokens
        self._max_query_tokens = max_query_tokens
        self._semantic_k = semantic_k

    async def pack(
        self,
        *,
        task_id: str,
        node_id: str,
        sub_task_description: str,
    ) -> PackedContext:
        task = await self._state.get_task(task_id)
        if task is None:
            raise ValueError(f"task not found: {task_id}")
        node = await self._state.get_dag_node(node_id)
        if node is None:
            raise ValueError(f"node not found: {node_id}")

        handoff_text = await self._format_handoff(task)
        upstream_text, present, missing, upstream_mems = await self._format_upstream(
            task, node
        )

        query = self._build_query(task, sub_task_description, upstream_mems)
        semantic_results = await self._semantic_supplement(
            task, node, query, upstream_mems
        )

        text, kept, dropped = self._apply_budget(
            title=task.title,
            handoff=handoff_text,
            upstream=upstream_text,
            sub_task=sub_task_description,
            semantic=semantic_results,
        )

        return PackedContext(
            text=text,
            handoff_present=handoff_text is not None,
            upstream_present=present,
            upstream_missing=missing,
            semantic_added=kept,
            semantic_dropped_for_budget=dropped,
            query_used=query,
            token_count=count_tokens(text),
        )

    # ---- handoff ----

    async def _format_handoff(self, task: TaskRow) -> str | None:
        if not task.handoff_conversation_id or not task.handoff_turn_range:
            return None
        rng = task.handoff_turn_range
        if not isinstance(rng, list) or len(rng) != 2:
            _log.warning("invalid handoff_turn_range: %r", rng)
            return None
        start, end = int(rng[0]), int(rng[1])
        if start > end:
            return None
        turns = await self._transcript.get_turns_by_range(
            task.handoff_conversation_id, start, end
        )
        if not turns:
            return (
                f"（接力点 conv={task.handoff_conversation_id} "
                f"turns=[{start},{end}] 未找到对话）"
            )
        lines: list[str] = []
        for t in turns:
            lines.append(f"## 第 {t.turn_index} 轮")
            lines.append(f"用户：{t.user_input}")
            lines.append(f"Agent({t.agent_id or '?'})：{t.agent_output}")
        return "\n".join(lines)

    # ---- upstream ----

    async def _format_upstream(
        self, task: TaskRow, node: DagNodeRow
    ) -> tuple[str, int, int, list[dict]]:
        """除了文本，还返回精确取到的上游记忆列表（供 query 拼装 + 语义去重用）。"""
        if not node.depends_on:
            return ("（无上游）", 0, 0, [])

        all_nodes = await self._state.list_dag_nodes(task.id)
        dep_map = {n.id: n for n in all_nodes}

        mids_to_fetch = [m for m in node.input_memory_ids if m]
        mems_by_id: dict[str, dict] = {}
        if mids_to_fetch:
            fetched = await self._memory.get_by_ids(task.user_id, mids_to_fetch)
            mems_by_id = {m["id"]: m for m in fetched}

        lines: list[str] = []
        upstream_mems_ordered: list[dict] = []
        present = 0
        missing = 0
        for i, dep_id in enumerate(node.depends_on):
            dep = dep_map.get(dep_id)
            dep_label = (
                f"{dep_id} ({dep.node_name})" if dep is not None else dep_id
            )
            mid = (
                node.input_memory_ids[i]
                if i < len(node.input_memory_ids)
                else None
            )

            if mid is None:
                missing += 1
                if dep is not None and dep.status == "skipped":
                    lines.append(f"- {dep_label}: 已跳过，无产出")
                elif dep is not None:
                    lines.append(
                        f"- {dep_label}: 无产出（status={dep.status}）"
                    )
                else:
                    lines.append(f"- {dep_label}: 依赖节点不存在")
                continue

            mem = mems_by_id.get(mid)
            if mem is None:
                missing += 1
                lines.append(f"- {dep_label}: 产出 {mid} 已丢失")
                continue

            mem_status = mem["metadata"].get("status", "?")
            if mem_status != "active":
                missing += 1
                lines.append(
                    f"- {dep_label}: （记忆 status={mem_status}）{mem['document']}"
                )
            else:
                present += 1
                lines.append(f"- {dep_label}: {mem['document']}")
                upstream_mems_ordered.append(mem)

        return ("\n".join(lines), present, missing, upstream_mems_ordered)

    # ---- query 构造 (spec §8.2) ----

    def _build_query(
        self,
        task: TaskRow,
        sub_task_desc: str,
        upstream_mems: list[dict],
    ) -> str:
        title = task.title
        upstream_docs = [m["document"] for m in upstream_mems if m.get("document")]

        # 阶梯截断：50 → 30 → 20 → 0
        for max_chars in (UPSTREAM_SUMMARY_CHARS, 30, 20, 0):
            parts = [title, sub_task_desc]
            if max_chars > 0:
                parts.extend(d[:max_chars] for d in upstream_docs)
            query = " ".join(p for p in parts if p)
            if count_tokens(query) <= self._max_query_tokens:
                return query
        # 底线：只保 title + sub_task
        return " ".join(p for p in (title, sub_task_desc) if p)

    # ---- 语义补充 ----

    async def _semantic_supplement(
        self,
        task: TaskRow,
        node: DagNodeRow,
        query: str,
        upstream_mems: list[dict],
    ) -> list[dict]:
        upstream_ids = {m["id"] for m in upstream_mems if m.get("id")}
        # 同时排除本节点自己之前的产出（重跑场景）
        if node.output_memory_id:
            upstream_ids.add(node.output_memory_id)

        # 多取一些，去重后再取 top
        raw = await self._memory.search(
            query,
            task.user_id,
            task.id,
            k=max(self._semantic_k * 3, 5),
            cross_task=False,
            status="active",
        )
        filtered = [r for r in raw if r["id"] not in upstream_ids]

        # spec §8.3：task_conclusion 优先；其次按距离（小 = 相关度高）
        def _level_rank(r: dict) -> int:
            return 0 if r["metadata"].get("memory_level") == "task_conclusion" else 1

        filtered.sort(key=lambda r: (_level_rank(r), r["distance"]))
        return filtered[: self._semantic_k]

    # ---- token budget ----

    def _apply_budget(
        self,
        *,
        title: str,
        handoff: str | None,
        upstream: str,
        sub_task: str,
        semantic: Iterable[dict],
    ) -> tuple[str, int, int]:
        semantic_list = list(semantic)

        def _assemble(handoff_part: str | None, kept_semantic: list[str]) -> str:
            parts = [f"# 任务主题\n\n{title}"]
            if handoff_part:
                parts.append(f"# 接力点（原文）\n\n{handoff_part}")
            parts.append(f"# 上游产出\n\n{upstream}")
            sem_body = "\n".join(kept_semantic) if kept_semantic else "（暂无）"
            parts.append(f"# 语义补充记忆\n\n{sem_body}")
            parts.append(f"# 子任务\n\n{sub_task}")
            return "\n\n".join(parts)

        # 第一轮：把全部语义补充放进去，看总 token
        sem_lines = [f"- {r['document']}" for r in semantic_list]
        full = _assemble(handoff, sem_lines)
        full_tokens = count_tokens(full)

        if full_tokens <= self._max_context_tokens:
            return full, len(sem_lines), 0

        # 超 budget：先逐条丢语义补充（按相关度从低到高 = 列表末尾）
        kept = list(sem_lines)
        dropped = 0
        while kept:
            kept.pop()
            dropped += 1
            text = _assemble(handoff, kept)
            if count_tokens(text) <= self._max_context_tokens:
                return text, len(kept), dropped

        # 全丢光了还超 → 截 handoff
        text_no_sem = _assemble(handoff, [])
        if count_tokens(text_no_sem) <= self._max_context_tokens:
            return text_no_sem, 0, dropped

        if handoff:
            # 计算可分配给 handoff 的 token：先得到不含 handoff 的核心段尺寸
            core = _assemble(None, [])
            core_tokens = count_tokens(core)
            # 留给 handoff 的预算（含 header 框架开销，预留 30 token）
            available = self._max_context_tokens - core_tokens - 30
            if available > 0:
                truncated = _truncate_to_tokens(handoff, available)
                truncated_marked = truncated + "\n（…接力点过长，已截断…）"
                text_truncated = _assemble(truncated_marked, [])
                if count_tokens(text_truncated) <= self._max_context_tokens:
                    return text_truncated, 0, dropped
            # 实在塞不下，直接丢 handoff
            return _assemble(None, []), 0, dropped

        # 没 handoff 也丢不动了 —— 已经只剩 title+upstream+sub_task
        return text_no_sem, 0, dropped
