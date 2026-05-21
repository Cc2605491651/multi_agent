"""writeback v1 集成测（任务 1.8）。

用真实 transcript_store + memory_store + stub LLM。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from storage.memory_store import MemoryStore
from storage.transcript_store import TranscriptStore
from worker.agent import Agent
from worker.writeback import writeback_turn


@dataclass
class _StubClient:
    distilled: str
    calls: list[dict] = field(default_factory=list)

    async def complete(self, *, model: str, system: str, messages, max_tokens=1024):
        self.calls.append({"model": model, "messages": list(messages)})
        if "提炼员" in system:
            return self.distilled
        return "（聊天回复，未用到）"


@pytest.fixture
def stores(tmp_path: Path):
    return (
        TranscriptStore(tmp_path / "transcript.db"),
        MemoryStore(tmp_path / "chroma"),
    )


async def test_writeback_writes_transcript_and_memory(stores) -> None:
    transcript_store, memory_store = stores
    agent = Agent(agent_id="research_agent", client=_StubClient(distilled="用户养橘猫米饭"))

    result = await writeback_turn(
        transcript_store=transcript_store,
        memory_store=memory_store,
        agent=agent,
        user_id="default_user",
        task_id="task_pets",
        conversation_id="conv_a",
        turn_index=1,
        user_input="我养了一只橘猫，叫米饭",
        agent_output="好的",
        node_id="node_001",
    )

    assert result.transcript_id
    assert result.memory_id and result.memory_id.startswith("mem_")
    assert result.memory_doc == "用户养橘猫米饭"

    # transcript 落库
    turns = await transcript_store.get_turns_by_range("conv_a", 1, 1)
    assert len(turns) == 1
    assert turns[0].agent_id == "research_agent"

    # memory 可被搜
    hits = await memory_store.search("用户的猫", "default_user", "task_pets", k=3)
    assert any(h["id"] == result.memory_id for h in hits)
    meta = next(h["metadata"] for h in hits if h["id"] == result.memory_id)
    assert meta["status"] == "active"
    assert meta["produced_by_node"] == "node_001"
    assert meta["memory_level"] == "node_output"


async def test_empty_distill_skips_memory(stores) -> None:
    transcript_store, memory_store = stores
    agent = Agent(agent_id="a", client=_StubClient(distilled=""))

    result = await writeback_turn(
        transcript_store=transcript_store,
        memory_store=memory_store,
        agent=agent,
        user_id="default_user",
        task_id="task_x",
        conversation_id="conv_x",
        turn_index=1,
        user_input="你好",
        agent_output="你好",
    )

    assert result.memory_id is None
    assert result.memory_doc is None
    # transcript 仍然写了
    turns = await transcript_store.get_turns_by_range("conv_x", 1, 1)
    assert len(turns) == 1
