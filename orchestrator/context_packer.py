"""上下文打包器（spec v4 §8 灵魂模块）—— 阶段 3 早期版。

四个来源（spec §8.1）当前只实现前三个：

1. **任务主题**：``tasks.title``
2. **接力点原文**：按 ``handoff_conversation_id`` + ``handoff_turn_range``
   去 ``transcript_store`` 取原文（不是摘要）
3. **上游产出（精确）**：按 ``node.input_memory_ids`` 顺序取，与 ``node.depends_on``
   一一对应；上游 ``skipped`` 时显式注明（spec §8.1 v4 新增）

第四个「相关记忆（语义检索）」+ §8.2 query 拼装 + token budget 由阶段 4c 落地。
本模块只保证：**上游产出按 id 精确取，不靠召回**——这是 spec §3.3 「P0 级」的判断。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from storage.memory_store import MemoryStore
from storage.state_store import DagNodeRow, StateStore, TaskRow
from storage.transcript_store import TranscriptStore

_log = logging.getLogger(__name__)


@dataclass
class PackedContext:
    text: str
    """对外暴露的最终 prompt 文本。"""

    handoff_present: bool
    upstream_present: int
    """有产出的上游条数（不含 skipped / None）。"""

    upstream_missing: int
    """缺失的上游条数（skipped / 产出丢失 / 还没填）。"""


class ContextPacker:
    def __init__(
        self,
        *,
        state_store: StateStore,
        transcript_store: TranscriptStore,
        memory_store: MemoryStore,
    ) -> None:
        self._state = state_store
        self._transcript = transcript_store
        self._memory = memory_store

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

        sections: list[str] = []
        sections.append("# 任务主题")
        sections.append(task.title)

        handoff_text = await self._format_handoff(task)
        handoff_present = handoff_text is not None
        if handoff_present:
            sections.append("# 接力点（原文）")
            sections.append(handoff_text)

        upstream_text, present, missing = await self._format_upstream(task, node)
        sections.append("# 上游产出")
        sections.append(upstream_text)

        sections.append("# 子任务")
        sections.append(sub_task_description)

        return PackedContext(
            text="\n\n".join(sections),
            handoff_present=handoff_present,
            upstream_present=present,
            upstream_missing=missing,
        )

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

    async def _format_upstream(
        self, task: TaskRow, node: DagNodeRow
    ) -> tuple[str, int, int]:
        if not node.depends_on:
            return ("（无上游）", 0, 0)

        # 取 dependency 节点
        all_nodes = await self._state.list_dag_nodes(task.id)
        dep_map = {n.id: n for n in all_nodes}

        # 一次性按 id 取 active 记忆
        mids_to_fetch = [m for m in node.input_memory_ids if m]
        mems_by_id: dict[str, dict] = {}
        if mids_to_fetch:
            fetched = await self._memory.get_by_ids(task.user_id, mids_to_fetch)
            mems_by_id = {m["id"]: m for m in fetched}

        lines: list[str] = []
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

        return ("\n".join(lines), present, missing)
