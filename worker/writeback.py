"""阶段 2 回写（spec v4 §6.2）。

严格三步顺序：

1. 写 ``transcript``（叶子数据）
2. 写 ``memory`` 但 ``status="pending"``（对其他 Worker 不可见）
3. **唯一提交点**：状态库一个事务里写 ``dag_nodes.status=done`` + ``output_memory_id``
   + ``finished_at``；事务成功后立即调 ``memory_store.update_status(pending→active)``。
   后者失败留给 :mod:`orchestrator.recovery` 的类 2 扫描修复（spec §6.2 注脚
   「以状态库为准，Chroma 最终一致」）。

阶段 1 的简化版（直接写 active）由 ``orchestrator/main.py`` demo 自行处理；
此模块从阶段 2 起只服务 DAG 节点回写。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from storage.memory_store import MemoryStore
from storage.state_store import StateStore
from storage.transcript_store import TranscriptStore
from worker.agent import Agent

_log = logging.getLogger(__name__)


@dataclass
class WritebackResult:
    transcript_id: str
    memory_id: Optional[str]
    memory_doc: Optional[str]
    committed: bool
    chroma_activated: bool


async def writeback_turn(
    *,
    transcript_store: TranscriptStore,
    memory_store: MemoryStore,
    state_store: StateStore,
    agent: Agent,
    user_id: str,
    task_id: str,
    node_id: str,
    conversation_id: str,
    turn_index: int,
    user_input: str,
    agent_output: str,
    chroma_update_hook: Optional[Callable[[], Awaitable[None]]] = None,
) -> WritebackResult:
    """执行 §6.2 三步回写。

    ``chroma_update_hook`` 用于故障注入测试：在第 3 步 (b) 真正调 Chroma update 之前
    先 await 它，hook 可抛异常模拟「事务成功但 Chroma 更新失败」。
    """
    # 第 1 步
    transcript_id = await transcript_store.add_turn(
        conversation_id=conversation_id,
        turn_index=turn_index,
        user_input=user_input,
        agent_output=agent_output,
        agent_id=agent.agent_id,
    )

    # 第 2 步：写 pending 记忆。memory_level 从 dag_nodes 取（spec §8.3）。
    doc = await agent.distill(user_input, agent_output)
    pending_mem_id: Optional[str] = None
    if doc:
        node_row = await state_store.get_dag_node(node_id)
        memory_level = node_row.memory_level if node_row else "node_output"
        metadata = {
            "task_id": task_id,
            "source_conversation_id": conversation_id,
            "source_turn_index": turn_index,
            "produced_by_agent": agent.agent_id,
            "produced_by_node": node_id,
            "memory_level": memory_level,
            "status": "pending",
        }
        pending_mem_id = await memory_store.add(user_id, doc, metadata)

    # 第 3 步 (a)：状态库唯一提交点
    committed = await state_store.commit_node_done(node_id, pending_mem_id)
    if not committed:
        # 节点状态已变（被 recovery 清理 / 已是终态）。pending 记忆留给类 3 清。
        _log.warning(
            "writeback aborted: node %s no longer running; pending mem=%s",
            node_id,
            pending_mem_id,
        )
        return WritebackResult(
            transcript_id=transcript_id,
            memory_id=None,
            memory_doc=None,
            committed=False,
            chroma_activated=False,
        )

    # 第 3 步 (b)：Chroma update pending → active；失败留给类 2 修
    chroma_activated = False
    if pending_mem_id:
        try:
            if chroma_update_hook is not None:
                await chroma_update_hook()
            await memory_store.update_status(user_id, pending_mem_id, "active")
            chroma_activated = True
        except Exception as e:  # noqa: BLE001
            _log.warning(
                "chroma update_status failed for mem=%s; recovery class-2 will fix: %s",
                pending_mem_id,
                e,
            )

    return WritebackResult(
        transcript_id=transcript_id,
        memory_id=pending_mem_id,
        memory_doc=doc or None,
        committed=True,
        chroma_activated=chroma_activated,
    )
