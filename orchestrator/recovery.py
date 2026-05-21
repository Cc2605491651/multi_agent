"""崩溃恢复扫描（spec v4 §6.3，三类缺一不可）。

启动时跑一次，运行期每 30s 跑一次（建议）。所有扫描幂等：

- **类 1**：``status=running`` + 心跳老化 → 删 pending 记忆 + 节点退回 ``pending``
  且 ``retry_count+1``（spec §5.2「重试前先按 §6.3 清理」）。
- **类 2**：``status=done`` 且 ``output_memory_id`` 非空，但 Chroma 里该记忆仍是
  ``pending`` → 重新 ``update_status → active``。修 §6.2 第 3 步 Chroma 那一拍失败。
- **类 3**：``status∈{failed, skipped}`` 节点关联的 pending 记忆 → 删。修 fail_fast
  取消后 / 重试耗尽的悬挂半成品。

幂等保证：三类扫描都是「读后修复」，重复跑结果一致。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

from storage.memory_store import MemoryStore
from storage.state_store import DagNodeRow, StateStore

_log = logging.getLogger(__name__)

DEFAULT_STALE_SECONDS = 300  # D-2.1: 5 分钟超时


@dataclass
class RecoveryReport:
    class_1_reset_running: int
    class_1_deleted_pending: int
    class_2_activated: int
    class_3_deleted_pending: int

    def total(self) -> int:
        return (
            self.class_1_reset_running
            + self.class_1_deleted_pending
            + self.class_2_activated
            + self.class_3_deleted_pending
        )


class Recovery:
    def __init__(
        self,
        state_store: StateStore,
        memory_store: MemoryStore,
        *,
        stale_seconds: int = DEFAULT_STALE_SECONDS,
    ) -> None:
        self._state_store = state_store
        self._memory_store = memory_store
        self._stale_seconds = stale_seconds

    async def sweep_all(self) -> RecoveryReport:
        c1_nodes, c1_mems = await self._sweep_running_stale()
        c2 = await self._sweep_done_pending_memory()
        c3 = await self._sweep_terminal_pending_memory()
        report = RecoveryReport(
            class_1_reset_running=c1_nodes,
            class_1_deleted_pending=c1_mems,
            class_2_activated=c2,
            class_3_deleted_pending=c3,
        )
        if report.total() > 0:
            _log.info("recovery sweep: %s", report)
        return report

    # ---------- 类 1 ----------

    async def _sweep_running_stale(self) -> tuple[int, int]:
        stale = await self._state_store.find_stale_running(self._stale_seconds)
        if not stale:
            return 0, 0
        user_id_cache = await self._cache_user_ids(stale)

        reset = 0
        deleted = 0
        for node in stale:
            uid = user_id_cache.get(node.task_id)
            if uid is not None:
                # 删该节点产出的 pending 记忆
                pendings = await self._memory_store.list_pending_for_node(
                    uid, node.id
                )
                if pendings:
                    n = await self._memory_store.delete(
                        uid, [p["id"] for p in pendings]
                    )
                    deleted += n
            await self._state_store.reset_node_to_pending(
                node.id, increment_retry=True
            )
            reset += 1
        return reset, deleted

    # ---------- 类 2 ----------

    async def _sweep_done_pending_memory(self) -> int:
        candidates = await self._state_store.find_done_with_memory()
        if not candidates:
            return 0
        user_id_cache = await self._cache_user_ids(candidates)
        activated = 0
        for node in candidates:
            uid = user_id_cache.get(node.task_id)
            if uid is None or node.output_memory_id is None:
                continue
            status = await self._memory_store.get_status(
                uid, node.output_memory_id
            )
            if status == "pending":
                await self._memory_store.update_status(
                    uid, node.output_memory_id, "active"
                )
                activated += 1
                _log.info(
                    "recovery class-2: activated mem=%s (node=%s)",
                    node.output_memory_id,
                    node.id,
                )
        return activated

    # ---------- 类 3 ----------

    async def _sweep_terminal_pending_memory(self) -> int:
        terminal = await self._state_store.find_terminal_nodes()
        if not terminal:
            return 0
        user_id_cache = await self._cache_user_ids(terminal)
        deleted = 0
        for node in terminal:
            uid = user_id_cache.get(node.task_id)
            if uid is None:
                continue
            pendings = await self._memory_store.list_pending_for_node(
                uid, node.id
            )
            if pendings:
                n = await self._memory_store.delete(
                    uid, [p["id"] for p in pendings]
                )
                deleted += n
                _log.info(
                    "recovery class-3: deleted %d pending mem(s) for terminal node=%s",
                    n,
                    node.id,
                )
        return deleted

    async def _cache_user_ids(
        self, nodes: Iterable[DagNodeRow]
    ) -> dict[str, str]:
        out: dict[str, str] = {}
        for n in nodes:
            if n.task_id in out:
                continue
            task = await self._state_store.get_task(n.task_id)
            if task is not None:
                out[n.task_id] = task.user_id
        return out
