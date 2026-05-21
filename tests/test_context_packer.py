"""context_packer 单测（阶段 3 任务 3.5）。

边界覆盖：input_memory_ids 为空 / 单条 / 多条 / 含 None（skipped 上游）/
对应记忆 status=pending / 接力点缺失 / handoff_turn_range 越界。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.context_packer import ContextPacker
from storage.memory_store import MemoryStore
from storage.state_store import StateStore
from storage.transcript_store import TranscriptStore


@pytest.fixture
def env(tmp_path: Path):
    state = StateStore(tmp_path / "state.db")
    transcript = TranscriptStore(tmp_path / "t.db")
    memory = MemoryStore(tmp_path / "chroma")
    packer = ContextPacker(
        state_store=state, transcript_store=transcript, memory_store=memory
    )
    return state, transcript, memory, packer


async def _add_done_node_with_memory(
    state: StateStore,
    memory: MemoryStore,
    *,
    task_id: str,
    user_id: str,
    node_name: str,
    doc: str,
    status: str = "active",
    depends_on: list[str] | None = None,
) -> tuple[str, str]:
    nid = await state.create_dag_node(
        task_id=task_id, node_name=node_name, depends_on=depends_on or []
    )
    await state.claim_node_running(nid, "w")
    mid = await memory.add(
        user_id,
        doc,
        {
            "task_id": task_id,
            "produced_by_node": nid,
            "produced_by_agent": node_name,
            "memory_level": "node_output",
            "status": status,
        },
    )
    await state.commit_node_done(nid, mid)
    return nid, mid


# ---- 任务主题恒在 ----


async def test_task_title_always_present(env) -> None:
    state, transcript, memory, packer = env
    tid = await state.create_task(user_id="u", title="调研：橘猫护理", dag_id="d")
    nid = await state.create_dag_node(task_id=tid, node_name="x")

    packed = await packer.pack(
        task_id=tid, node_id=nid, sub_task_description="干活"
    )
    assert "# 任务主题" in packed.text
    assert "调研：橘猫护理" in packed.text


# ---- 接力点：有/无/越界 ----


async def test_handoff_section_present_when_set(env) -> None:
    state, transcript, memory, packer = env
    # 先写几轮 transcript
    for i in range(1, 4):
        await transcript.add_turn(
            conversation_id="conv_hist",
            turn_index=i,
            user_input=f"用户第 {i} 轮发言",
            agent_output=f"Agent 第 {i} 轮回复",
            agent_id="planner",
        )
    tid = await state.create_task(
        user_id="u",
        title="t",
        dag_id="d",
        handoff_conversation_id="conv_hist",
        handoff_turn_range=[2, 3],
    )
    nid = await state.create_dag_node(task_id=tid, node_name="x")

    packed = await packer.pack(
        task_id=tid, node_id=nid, sub_task_description="干活"
    )
    assert packed.handoff_present is True
    assert "用户第 2 轮发言" in packed.text
    assert "Agent 第 3 轮回复" in packed.text
    # 范围外不应出现
    assert "用户第 1 轮发言" not in packed.text


async def test_no_handoff_when_unset(env) -> None:
    state, transcript, memory, packer = env
    tid = await state.create_task(user_id="u", title="t", dag_id="d")
    nid = await state.create_dag_node(task_id=tid, node_name="x")
    packed = await packer.pack(
        task_id=tid, node_id=nid, sub_task_description="干活"
    )
    assert packed.handoff_present is False
    assert "# 接力点" not in packed.text


async def test_handoff_range_with_no_turns(env) -> None:
    state, transcript, memory, packer = env
    tid = await state.create_task(
        user_id="u",
        title="t",
        dag_id="d",
        handoff_conversation_id="ghost_conv",
        handoff_turn_range=[1, 5],
    )
    nid = await state.create_dag_node(task_id=tid, node_name="x")
    packed = await packer.pack(
        task_id=tid, node_id=nid, sub_task_description="干活"
    )
    assert packed.handoff_present is True
    assert "未找到对话" in packed.text


# ---- 上游产出 ----


async def test_no_upstream_when_no_depends_on(env) -> None:
    state, transcript, memory, packer = env
    tid = await state.create_task(user_id="u", title="t", dag_id="d")
    nid = await state.create_dag_node(task_id=tid, node_name="root")
    packed = await packer.pack(
        task_id=tid, node_id=nid, sub_task_description="干活"
    )
    assert "（无上游）" in packed.text
    assert packed.upstream_present == 0
    assert packed.upstream_missing == 0


async def test_single_upstream_active(env) -> None:
    state, transcript, memory, packer = env
    tid = await state.create_task(user_id="u", title="t", dag_id="d")
    upstream_id, upstream_mid = await _add_done_node_with_memory(
        state, memory, task_id=tid, user_id="u", node_name="research",
        doc="结论：方案 X 因为 Y"
    )
    downstream_id = await state.create_dag_node(
        task_id=tid, node_name="writing", depends_on=[upstream_id]
    )
    await state.set_node_input_memory_ids(downstream_id, [upstream_mid])

    packed = await packer.pack(
        task_id=tid, node_id=downstream_id, sub_task_description="干活"
    )
    # spec §3.3 P0：上游原文必须**直接出现**在打包结果里
    assert "结论：方案 X 因为 Y" in packed.text
    assert "(research)" in packed.text
    assert packed.upstream_present == 1
    assert packed.upstream_missing == 0


async def test_multiple_upstreams_in_depends_order(env) -> None:
    state, transcript, memory, packer = env
    tid = await state.create_task(user_id="u", title="t", dag_id="d")
    a_id, a_mid = await _add_done_node_with_memory(
        state, memory, task_id=tid, user_id="u", node_name="a", doc="结论 A"
    )
    b_id, b_mid = await _add_done_node_with_memory(
        state, memory, task_id=tid, user_id="u", node_name="b", doc="结论 B"
    )
    c_id = await state.create_dag_node(
        task_id=tid, node_name="c", depends_on=[a_id, b_id]
    )
    await state.set_node_input_memory_ids(c_id, [a_mid, b_mid])

    packed = await packer.pack(
        task_id=tid, node_id=c_id, sub_task_description="干活"
    )
    assert "结论 A" in packed.text
    assert "结论 B" in packed.text
    # depends_on 顺序：A 在 B 前
    assert packed.text.index("结论 A") < packed.text.index("结论 B")
    assert packed.upstream_present == 2


async def test_skipped_upstream_marked_explicitly(env) -> None:
    """spec §8.1 v4：上游 skipped → input_memory_ids 这一项是 null，打包时显式注明。"""
    state, transcript, memory, packer = env
    tid = await state.create_task(user_id="u", title="t", dag_id="d")
    a_id, a_mid = await _add_done_node_with_memory(
        state, memory, task_id=tid, user_id="u", node_name="research_a",
        doc="A 的产出"
    )
    # b 节点 skipped
    b_id = await state.create_dag_node(
        task_id=tid, node_name="research_b", failure_policy="fail_skip"
    )
    await state.claim_node_running(b_id, "w")
    await state.mark_node_terminal(b_id, "skipped")

    c_id = await state.create_dag_node(
        task_id=tid, node_name="summarize", depends_on=[a_id, b_id]
    )
    await state.set_node_input_memory_ids(c_id, [a_mid, None])

    packed = await packer.pack(
        task_id=tid, node_id=c_id, sub_task_description="汇总"
    )
    assert "A 的产出" in packed.text
    assert "已跳过，无产出" in packed.text
    assert "research_b" in packed.text
    assert packed.upstream_present == 1
    assert packed.upstream_missing == 1


async def test_upstream_with_pending_memory_marked(env) -> None:
    """记忆还在 pending（recovery 类 2 没修）→ 显式标 status=pending 而非装作 active。"""
    state, transcript, memory, packer = env
    tid = await state.create_task(user_id="u", title="t", dag_id="d")
    a_id, a_mid = await _add_done_node_with_memory(
        state, memory, task_id=tid, user_id="u", node_name="a",
        doc="未生效的结论", status="pending",
    )
    b_id = await state.create_dag_node(
        task_id=tid, node_name="b", depends_on=[a_id]
    )
    await state.set_node_input_memory_ids(b_id, [a_mid])

    packed = await packer.pack(
        task_id=tid, node_id=b_id, sub_task_description="x"
    )
    # 文本仍然取出来了（让下游 Worker 知道），但标了状态
    assert "未生效的结论" in packed.text
    assert "status=pending" in packed.text
    assert packed.upstream_missing == 1


async def test_input_memory_ids_lookup_does_not_call_search(env, monkeypatch) -> None:
    """spec §3.3 / §8.1 关键不变量：input_memory_ids 路径只走 get_by_ids，不走 search。"""
    state, transcript, memory, packer = env
    tid = await state.create_task(user_id="u", title="t", dag_id="d")
    a_id, a_mid = await _add_done_node_with_memory(
        state, memory, task_id=tid, user_id="u", node_name="a", doc="原文"
    )
    b_id = await state.create_dag_node(
        task_id=tid, node_name="b", depends_on=[a_id]
    )
    await state.set_node_input_memory_ids(b_id, [a_mid])

    search_called: list[tuple] = []

    real_search = memory.search

    async def spy_search(*args, **kwargs):
        search_called.append((args, kwargs))
        return await real_search(*args, **kwargs)

    monkeypatch.setattr(memory, "search", spy_search)

    await packer.pack(task_id=tid, node_id=b_id, sub_task_description="x")
    assert search_called == [], "context_packer must not call search() in phase 3"


# ---- 错误 ----


async def test_unknown_task_raises(env) -> None:
    _, _, _, packer = env
    with pytest.raises(ValueError):
        await packer.pack(task_id="ghost", node_id="ghost", sub_task_description="x")


async def test_unknown_node_raises(env) -> None:
    state, _, _, packer = env
    tid = await state.create_task(user_id="u", title="t", dag_id="d")
    with pytest.raises(ValueError):
        await packer.pack(task_id=tid, node_id="ghost", sub_task_description="x")
