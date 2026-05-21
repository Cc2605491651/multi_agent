"""阶段 1 简化版回写（spec v4 §6.2 的剥皮版）。

阶段 1 不做 pending → active 的两步原子性，**直接落 active**；
阶段 2 升级为完整三步顺序 + recovery。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from storage.memory_store import MemoryStore
from storage.transcript_store import TranscriptStore
from worker.agent import Agent


@dataclass
class WritebackResult:
    transcript_id: str
    memory_id: Optional[str]
    memory_doc: Optional[str]


async def writeback_turn(
    *,
    transcript_store: TranscriptStore,
    memory_store: MemoryStore,
    agent: Agent,
    user_id: str,
    task_id: str,
    conversation_id: str,
    turn_index: int,
    user_input: str,
    agent_output: str,
    node_id: Optional[str] = None,
) -> WritebackResult:
    """单轮对话回写：写 transcript → 提炼 → 写记忆（active）。"""
    transcript_id = await transcript_store.add_turn(
        conversation_id=conversation_id,
        turn_index=turn_index,
        user_input=user_input,
        agent_output=agent_output,
        agent_id=agent.agent_id,
    )

    doc = await agent.distill(user_input, agent_output)
    if not doc:
        return WritebackResult(transcript_id=transcript_id, memory_id=None, memory_doc=None)

    metadata = {
        "task_id": task_id,
        "source_conversation_id": conversation_id,
        "source_turn_index": turn_index,
        "produced_by_agent": agent.agent_id,
        "produced_by_node": node_id or "",
        "memory_level": "node_output",
        # 阶段 1 简化：直接 active；阶段 2 改成 pending → 顺序回写后 update active
        "status": "active",
    }
    mem_id = await memory_store.add(user_id, doc, metadata)
    return WritebackResult(transcript_id=transcript_id, memory_id=mem_id, memory_doc=doc)
