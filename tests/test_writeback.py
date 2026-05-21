"""writeback v2 集成测（阶段 2 任务 2.4）。

覆盖 spec §6.2 三步顺序 + 唯一提交点 + Chroma update 失败的可恢复性。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from storage.memory_store import MemoryStore
from storage.state_store import StateStore
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
        return "（聊天回复）"


@pytest.fixture
def stores(tmp_path: Path):
    return (
        TranscriptStore(tmp_path / "transcript.db"),
        MemoryStore(tmp_path / "chroma"),
        StateStore(tmp_path / "state.db"),
    )


async def _setup_running_node(state_store: StateStore, user_id="default_user"):
    tid = await state_store.create_task(
        user_id=user_id, title="t", dag_id="d"
    )
    nid = await state_store.create_dag_node(task_id=tid, node_name="n")
    assert await state_store.claim_node_running(nid, "w1") is True
    return tid, nid


async def test_writeback_v2_three_step_success(stores) -> None:
    transcript_store, memory_store, state_store = stores
    agent = Agent(agent_id="a1", client=_StubClient(distilled="用户养橘猫米饭"))
    task_id, node_id = await _setup_running_node(state_store)

    result = await writeback_turn(
        transcript_store=transcript_store,
        memory_store=memory_store,
        state_store=state_store,
        agent=agent,
        user_id="default_user",
        task_id=task_id,
        node_id=node_id,
        conversation_id="conv_a",
        turn_index=1,
        user_input="我养了一只橘猫，叫米饭",
        agent_output="好的",
    )

    assert result.committed is True
    assert result.chroma_activated is True
    assert result.memory_id and result.memory_id.startswith("mem_")
    assert result.memory_doc == "用户养橘猫米饭"

    # 节点已 done + output_memory_id 已填
    n = await state_store.get_dag_node(node_id)
    assert n.status == "done"
    assert n.output_memory_id == result.memory_id
    assert n.finished_at is not None

    # 记忆已 active
    assert await memory_store.get_status("default_user", result.memory_id) == "active"

    # 默认搜应能搜到
    hits = await memory_store.search("用户的猫", "default_user", task_id, k=3)
    assert any(h["id"] == result.memory_id for h in hits)


async def test_writeback_v2_empty_distill_still_commits(stores) -> None:
    transcript_store, memory_store, state_store = stores
    agent = Agent(agent_id="a1", client=_StubClient(distilled=""))
    task_id, node_id = await _setup_running_node(state_store)

    result = await writeback_turn(
        transcript_store=transcript_store,
        memory_store=memory_store,
        state_store=state_store,
        agent=agent,
        user_id="default_user",
        task_id=task_id,
        node_id=node_id,
        conversation_id="conv_b",
        turn_index=1,
        user_input="你好",
        agent_output="你好",
    )
    assert result.committed is True
    assert result.memory_id is None
    assert result.chroma_activated is False  # 没记忆就没 activate

    n = await state_store.get_dag_node(node_id)
    assert n.status == "done"
    assert n.output_memory_id is None


async def test_writeback_v2_chroma_update_failure_leaves_pending(stores) -> None:
    """spec §6.2 注脚：事务成功后 Chroma update 失败 → 记忆仍 pending → 等类 2 修。"""
    transcript_store, memory_store, state_store = stores
    agent = Agent(agent_id="a1", client=_StubClient(distilled="重要结论"))
    task_id, node_id = await _setup_running_node(state_store)

    async def explode():
        raise RuntimeError("模拟 Chroma 网络抖动")

    result = await writeback_turn(
        transcript_store=transcript_store,
        memory_store=memory_store,
        state_store=state_store,
        agent=agent,
        user_id="default_user",
        task_id=task_id,
        node_id=node_id,
        conversation_id="conv_c",
        turn_index=1,
        user_input="u",
        agent_output="a",
        chroma_update_hook=explode,
    )

    # 状态库事务依然成功
    assert result.committed is True
    assert result.chroma_activated is False
    assert result.memory_id

    n = await state_store.get_dag_node(node_id)
    assert n.status == "done"
    assert n.output_memory_id == result.memory_id

    # Chroma 里记忆仍是 pending —— 默认 active 搜搜不到
    assert await memory_store.get_status("default_user", result.memory_id) == "pending"
    hits = await memory_store.search("结论", "default_user", task_id, k=3)
    assert all(h["id"] != result.memory_id for h in hits)


async def test_writeback_v2_node_not_running_aborts(stores) -> None:
    """节点不是 running（如已被 recovery 清理）→ commit 失败，结果返回 committed=False。"""
    transcript_store, memory_store, state_store = stores
    agent = Agent(agent_id="a1", client=_StubClient(distilled="结论"))

    tid = await state_store.create_task(user_id="default_user", title="t", dag_id="d")
    nid = await state_store.create_dag_node(task_id=tid, node_name="n")
    # 故意不 claim_running，节点保持 pending

    result = await writeback_turn(
        transcript_store=transcript_store,
        memory_store=memory_store,
        state_store=state_store,
        agent=agent,
        user_id="default_user",
        task_id=tid,
        node_id=nid,
        conversation_id="conv_d",
        turn_index=1,
        user_input="u",
        agent_output="a",
    )

    assert result.committed is False
    assert result.memory_id is None
    n = await state_store.get_dag_node(nid)
    assert n.status == "pending"
    # transcript 仍写入了（叶子数据无害）
    turns = await transcript_store.get_turns_by_range("conv_d", 1, 1)
    assert len(turns) == 1


async def test_writeback_v2_step_order_pending_visible_before_commit(stores) -> None:
    """spec §6.2：第 2 步之后第 3 步之前，记忆已落库但 status=pending（其他 Worker 不可见）。"""
    transcript_store, memory_store, state_store = stores
    agent = Agent(agent_id="a1", client=_StubClient(distilled="即将生效的结论"))
    task_id, node_id = await _setup_running_node(state_store)

    captured: dict = {}

    async def inspect_hook():
        # 此时第 2 步已完成（pending 记忆写入），但第 3 步 Chroma update 还没跑
        # 注意：第 3 步 (a) state 事务已 commit 完毕，这是 (b) 之前
        # 这里检查 pending 记忆按状态搜不到（默认 active）
        hits = await memory_store.search(
            "结论", "default_user", task_id, k=5
        )
        captured["active_hits"] = [h["id"] for h in hits]
        pending_hits = await memory_store.search(
            "结论", "default_user", task_id, k=5, status="pending"
        )
        captured["pending_hits"] = [h["id"] for h in pending_hits]

    result = await writeback_turn(
        transcript_store=transcript_store,
        memory_store=memory_store,
        state_store=state_store,
        agent=agent,
        user_id="default_user",
        task_id=task_id,
        node_id=node_id,
        conversation_id="conv_e",
        turn_index=1,
        user_input="u",
        agent_output="a",
        chroma_update_hook=inspect_hook,
    )

    assert result.committed is True
    assert result.memory_id in captured["pending_hits"]
    assert result.memory_id not in captured["active_hits"]
