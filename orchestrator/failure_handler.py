"""失败处理（spec v4 §5.2 / §5.3 / 阶段 4a 任务 4a.2）。

把「节点失败」翻译成「下一步动作」：

- 还有 retry 额度 → ``retry``：清理 §6.3 的脏数据 + 节点退回 pending（``retry_count+1``）
- 重试耗尽 → 按 ``failure_policy`` 落终态（spec §5.2 表，v4 修正）：
  - ``fail_retry`` → ``failed``，任务级 ``failed``
  - ``fail_skip`` → ``skipped``，任务继续
  - ``fail_fast`` → ``failed``，任务级 ``failed`` + 取消并发兄弟

实际「取消兄弟」由 scheduler 负责（持有 sandbox handles）；本模块只回报需要这么做。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from storage.memory_store import MemoryStore
from storage.state_store import DagNodeRow, StateStore

_log = logging.getLogger(__name__)

NodeActionKind = Literal["retry", "skip", "fail"]


@dataclass(frozen=True)
class NodeAction:
    kind: NodeActionKind
    task_should_fail: bool
    cancel_siblings: bool = False


class FailureHandler:
    def __init__(
        self, state_store: StateStore, memory_store: MemoryStore
    ) -> None:
        self._state = state_store
        self._memory = memory_store

    async def on_node_failed(self, node: DagNodeRow) -> NodeAction:
        """节点失败时调用，决定下一步动作并落 DB。"""
        if node.retry_count < node.max_retries:
            # 还有 retry 额度 —— 不管 failure_policy。
            # spec §5.2「重试前先按 §6.3 清理」：手动删本节点关联的 pending 记忆，
            # 不依赖通用 recovery 扫描（节点状态还没切换，类 1/类 3 都不命中）。
            await self._clean_pending_for_node(node)
            # 退回 pending；retry_count+1 由 reset 完成
            await self._state.reset_node_to_pending(
                node.id, increment_retry=True
            )
            _log.info(
                "node %s failed (retry %d/%d), reset to pending",
                node.id,
                node.retry_count + 1,
                node.max_retries,
            )
            return NodeAction(kind="retry", task_should_fail=False)

        # 重试耗尽：按 failure_policy 终态
        policy = node.failure_policy
        if policy == "fail_skip":
            await self._state.mark_node_terminal(node.id, "skipped")
            _log.info(
                "node %s failed with fail_skip policy after %d retries; marking skipped",
                node.id, node.retry_count,
            )
            return NodeAction(kind="skip", task_should_fail=False)

        # fail_retry / fail_fast 都是 failed 终态；区别在 cancel_siblings
        await self._state.mark_node_terminal(node.id, "failed")
        cancel_siblings = policy == "fail_fast"
        _log.info(
            "node %s failed with %s policy after %d retries; marking failed (cancel_siblings=%s)",
            node.id, policy, node.retry_count, cancel_siblings,
        )
        return NodeAction(
            kind="fail", task_should_fail=True, cancel_siblings=cancel_siblings
        )

    async def on_dependency_terminal_failed(self, node: DagNodeRow) -> None:
        """spec §5.2：fail_retry / fail_fast 失败 → 下游全部 skipped。

        scheduler 在标记终态节点为 failed 之后，对所有「依赖该节点的 pending 节点」
        调用此方法，递归把下游链置为 skipped。
        """
        if node.status != "pending":
            return
        await self._state.mark_node_terminal(node.id, "skipped")

    async def _clean_pending_for_node(self, node: DagNodeRow) -> None:
        task = await self._state.get_task(node.task_id)
        if task is None:
            return
        pendings = await self._memory.list_pending_for_node(
            task.user_id, node.id
        )
        if not pendings:
            return
        await self._memory.delete(task.user_id, [p["id"] for p in pendings])
        _log.info(
            "cleaned %d pending memory for retry of node %s",
            len(pendings), node.id,
        )
