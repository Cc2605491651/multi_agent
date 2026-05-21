"""节点心跳上报（spec v4 §6.1 / D-2.1）。

每 30 秒往 ``state_store.update_heartbeat`` 写一次；
作为 asyncio 后台任务跑，``async with`` 退出时自动取消。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

from storage.state_store import StateStore

_log = logging.getLogger(__name__)

DEFAULT_INTERVAL = 30.0  # D-2.1: 30s


class HeartbeatTask:
    """async context manager 包装一个后台心跳 task。

    用法::

        async with HeartbeatTask(state_store, node_id, interval=30):
            # 做活儿，期间 heartbeat_at 每 interval 秒被刷一次
            ...
    """

    def __init__(
        self,
        state_store: StateStore,
        node_id: str,
        interval: float = DEFAULT_INTERVAL,
    ) -> None:
        self._state_store = state_store
        self._node_id = node_id
        self._interval = interval
        self._task: asyncio.Task | None = None

    async def __aenter__(self) -> "HeartbeatTask":
        # 启动时无需立即打拍：调用者通常刚 claim_node_running，state_store 已写入
        # heartbeat_at；首拍由后台 task 在 ``interval`` 秒后完成。
        self._task = asyncio.create_task(
            self._run(), name=f"heartbeat:{self._node_id}"
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def _run(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._interval)
                try:
                    await self._state_store.update_heartbeat(self._node_id)
                except Exception as e:  # noqa: BLE001
                    # 心跳失败不应中断主流程，只记日志
                    _log.warning(
                        "heartbeat failed for node=%s: %s", self._node_id, e
                    )
        except asyncio.CancelledError:
            raise
