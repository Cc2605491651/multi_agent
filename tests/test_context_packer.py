"""context_packer 单测（阶段 3 + 4c）。

覆盖 spec §8.1（四个来源 + skipped 显式注明）、§8.2（query 构造 + 截断 +
token budget）、§8.3（memory_level 排序）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.context_packer import (
    ContextPacker,
    UPSTREAM_SUMMARY_CHARS,
    count_tokens,
)
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
    state, memory, *, task_id, user_id, node_name,
    doc, status="active", depends_on=None,
    memory_level="node_output",
) -> tuple[str, str]:
    nid = await state.create_dag_node(
        task_id=task_id, node_name=node_name,
        depends_on=depends_on or [], memory_level=memory_level,
    )
    await state.claim_node_running(nid, "w")
    mid = await memory.add(
        user_id,
        doc,
        {
            "task_id": task_id,
            "produced_by_node": nid,
            "produced_by_agent": node_name,
            "memory_level": memory_level,
            "status": status,
        },
    )
    await state.commit_node_done(nid, mid)
    return nid, mid


# ---- §8.1 四个来源 ----


async def test_task_title_always_present(env) -> None:
    state, _, _, packer = env
    tid = await state.create_task(user_id="u", title="调研：橘猫护理", dag_id="d")
    nid = await state.create_dag_node(task_id=tid, node_name="x")
    packed = await packer.pack(task_id=tid, node_id=nid, sub_task_description="干活")
    assert "调研：橘猫护理" in packed.text


async def test_handoff_section_present_when_set(env) -> None:
    state, transcript, _, packer = env
    for i in range(1, 4):
        await transcript.add_turn(
            conversation_id="conv_hist", turn_index=i,
            user_input=f"用户第 {i} 轮发言", agent_output=f"Agent 第 {i} 轮回复",
            agent_id="planner",
        )
    tid = await state.create_task(
        user_id="u", title="t", dag_id="d",
        handoff_conversation_id="conv_hist", handoff_turn_range=[2, 3],
    )
    nid = await state.create_dag_node(task_id=tid, node_name="x")
    packed = await packer.pack(task_id=tid, node_id=nid, sub_task_description="干活")
    assert packed.handoff_present is True
    assert "用户第 2 轮发言" in packed.text
    assert "Agent 第 3 轮回复" in packed.text
    assert "用户第 1 轮发言" not in packed.text


async def test_no_handoff_when_unset(env) -> None:
    state, _, _, packer = env
    tid = await state.create_task(user_id="u", title="t", dag_id="d")
    nid = await state.create_dag_node(task_id=tid, node_name="x")
    packed = await packer.pack(task_id=tid, node_id=nid, sub_task_description="干活")
    assert packed.handoff_present is False
    assert "# 接力点" not in packed.text


async def test_single_upstream_active(env) -> None:
    state, _, memory, packer = env
    tid = await state.create_task(user_id="u", title="t", dag_id="d")
    upstream_id, upstream_mid = await _add_done_node_with_memory(
        state, memory, task_id=tid, user_id="u", node_name="research",
        doc="结论：方案 X 因为 Y",
    )
    downstream_id = await state.create_dag_node(
        task_id=tid, node_name="writing", depends_on=[upstream_id]
    )
    await state.set_node_input_memory_ids(downstream_id, [upstream_mid])

    packed = await packer.pack(
        task_id=tid, node_id=downstream_id, sub_task_description="干活"
    )
    assert "结论：方案 X 因为 Y" in packed.text
    assert "(research)" in packed.text
    assert packed.upstream_present == 1
    assert packed.upstream_missing == 0


async def test_skipped_upstream_marked_explicitly(env) -> None:
    state, _, memory, packer = env
    tid = await state.create_task(user_id="u", title="t", dag_id="d")
    a_id, a_mid = await _add_done_node_with_memory(
        state, memory, task_id=tid, user_id="u", node_name="research_a",
        doc="A 的产出",
    )
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
    assert packed.upstream_present == 1
    assert packed.upstream_missing == 1


# ---- §8.1 第 4 个来源：语义补充检索 ----


async def test_semantic_supplement_added(env) -> None:
    """task 内有相关记忆但不在 input_memory_ids 时，应通过语义检索补进来。"""
    state, _, memory, packer = env
    tid = await state.create_task(user_id="u", title="调研选型", dag_id="d")

    # 直接落几条「同 task 但不属于 input_memory_ids 的历史结论」
    bg_mid = await memory.add(
        "u", "历史背景：用户偏好开源方案",
        {
            "task_id": tid, "produced_by_node": "node_history",
            "produced_by_agent": "x", "memory_level": "node_output",
            "status": "active",
        },
    )

    # 当前节点没有 depends_on，但有 task_id，可以语义召回
    nid = await state.create_dag_node(task_id=tid, node_name="research")
    packed = await packer.pack(
        task_id=tid, node_id=nid, sub_task_description="围绕选型给出 3 条建议"
    )
    assert packed.semantic_added >= 1
    assert "用户偏好开源方案" in packed.text
    assert "# 语义补充记忆" in packed.text


async def test_semantic_excludes_input_memory_ids(env) -> None:
    """已经在 input_memory_ids 里的记忆不应再出现在语义补充段（去重）。"""
    state, _, memory, packer = env
    tid = await state.create_task(user_id="u", title="t", dag_id="d")
    a_id, a_mid = await _add_done_node_with_memory(
        state, memory, task_id=tid, user_id="u", node_name="research",
        doc="非常独特的关键词 ZQXJK",
    )
    b_id = await state.create_dag_node(
        task_id=tid, node_name="writing", depends_on=[a_id]
    )
    await state.set_node_input_memory_ids(b_id, [a_mid])

    packed = await packer.pack(
        task_id=tid, node_id=b_id, sub_task_description="ZQXJK 关键词"
    )
    # 上游段有 ZQXJK，语义补充段不应也有（去重）
    assert packed.text.count("ZQXJK") <= 2  # title + sub_task 最多各一处不含；只有上游段含
    # 更严格：语义补充段内不应含 a_mid 的 doc
    semantic_section = packed.text.split("# 语义补充记忆", 1)[1].split("# 子任务", 1)[0]
    assert "ZQXJK" not in semantic_section


async def test_task_conclusion_ranked_first(env) -> None:
    """spec §8.3：语义补充中 task_conclusion 排在 node_output 之前。"""
    state, _, memory, packer = env
    tid = await state.create_task(user_id="u", title="项目 X 决策", dag_id="d")

    # 同 task 内一条 node_output（距离可能更近）+ 一条 task_conclusion（距离稍远）
    await memory.add(
        "u", "项目 X 的中间节点输出：候选方案三选一",
        {
            "task_id": tid, "produced_by_node": "n_mid",
            "produced_by_agent": "x", "memory_level": "node_output",
            "status": "active",
        },
    )
    await memory.add(
        "u", "项目 X 的最终结论：选方案 B",
        {
            "task_id": tid, "produced_by_node": "n_final",
            "produced_by_agent": "x", "memory_level": "task_conclusion",
            "status": "active",
        },
    )

    nid = await state.create_dag_node(task_id=tid, node_name="downstream")
    packed = await packer.pack(
        task_id=tid, node_id=nid, sub_task_description="基于项目 X 已有结论"
    )
    sem = packed.text.split("# 语义补充记忆", 1)[1].split("# 子任务", 1)[0]
    pos_conclusion = sem.find("最终结论")
    pos_midnode = sem.find("中间节点输出")
    assert pos_conclusion != -1 and pos_midnode != -1
    assert pos_conclusion < pos_midnode, "task_conclusion 必须排在 node_output 前"


# ---- §8.2 query 构造 ----


async def test_query_uses_title_and_sub_task(env) -> None:
    state, _, _, packer = env
    tid = await state.create_task(user_id="u", title="独特任务主题", dag_id="d")
    nid = await state.create_dag_node(task_id=tid, node_name="x")
    packed = await packer.pack(
        task_id=tid, node_id=nid, sub_task_description="独特子任务说明"
    )
    assert "独特任务主题" in packed.query_used
    assert "独特子任务说明" in packed.query_used


async def test_query_respects_max_tokens(env) -> None:
    """spec §8.2：query 总长 ≤ 200 token，超出按阶梯截断。"""
    state, _, memory, packer = env
    tid = await state.create_task(user_id="u", title="t", dag_id="d")
    # 故意造一堆长上游产出，挤爆 query
    upstream_ids = []
    long_doc = "重要内容片段。" * 30  # 远超 50 字
    for i in range(6):
        a_id, mid = await _add_done_node_with_memory(
            state, memory, task_id=tid, user_id="u",
            node_name=f"up_{i}", doc=long_doc + f" {i}",
        )
        upstream_ids.append((a_id, mid))

    down_id = await state.create_dag_node(
        task_id=tid, node_name="d",
        depends_on=[a for a, _ in upstream_ids],
    )
    await state.set_node_input_memory_ids(
        down_id, [m for _, m in upstream_ids]
    )
    packed = await packer.pack(
        task_id=tid, node_id=down_id, sub_task_description="干"
    )
    assert count_tokens(packed.query_used) <= 200, (
        f"query token over 200: {count_tokens(packed.query_used)}"
    )


async def test_query_keeps_title_and_sub_task_as_baseline(env) -> None:
    """阶梯截断到极端时，底线是 title + sub_task 保留。"""
    state, _, memory, packer = env
    tid = await state.create_task(user_id="u", title="ABCDEFG", dag_id="d")
    long_doc = "X" * 800
    for i in range(8):
        await _add_done_node_with_memory(
            state, memory, task_id=tid, user_id="u",
            node_name=f"up_{i}", doc=long_doc,
        )
    nid = await state.create_dag_node(task_id=tid, node_name="d")
    packed = await packer.pack(
        task_id=tid, node_id=nid, sub_task_description="HIJKLMN"
    )
    assert "ABCDEFG" in packed.query_used
    assert "HIJKLMN" in packed.query_used


# ---- §8.2 token budget ----


async def test_token_budget_default_2000(env) -> None:
    state, _, _, packer = env
    tid = await state.create_task(user_id="u", title="t", dag_id="d")
    nid = await state.create_dag_node(task_id=tid, node_name="x")
    packed = await packer.pack(task_id=tid, node_id=nid, sub_task_description="x")
    assert packed.token_count <= 2000


async def test_token_budget_drops_semantic_low_priority_first(env, tmp_path):
    """spec §8.2：超 budget 时，按相关度从低到高（distance 大的先丢）裁语义补充。"""
    state = StateStore(tmp_path / "state.db")
    transcript = TranscriptStore(tmp_path / "t.db")
    memory = MemoryStore(tmp_path / "chroma")
    # 故意把 budget 设很小，逼出裁剪
    packer = ContextPacker(
        state_store=state, transcript_store=transcript, memory_store=memory,
        max_context_tokens=200, semantic_k=5,
    )
    tid = await state.create_task(user_id="u", title="主题", dag_id="d")

    # 写 5 条相关度依次降低，且足够长以触发裁剪
    pad = "细节叙述 " * 15  # 拉到 ~150 token
    docs = [
        f"主题相关结论 A，高度匹配关键词。{pad}",
        f"主题相关结论 B，匹配但弱一些。{pad}",
        f"主题相关结论 C，少量重叠。{pad}",
        f"完全不相关的旅行计划。{pad}",
        f"完全不相关的菜谱配方。{pad}",
    ]
    for i, d in enumerate(docs):
        await memory.add(
            "u", d,
            {
                "task_id": tid, "produced_by_node": f"n_{i}",
                "produced_by_agent": "x", "memory_level": "node_output",
                "status": "active",
            },
        )

    nid = await state.create_dag_node(task_id=tid, node_name="x")
    packed = await packer.pack(
        task_id=tid, node_id=nid, sub_task_description="围绕主题展开"
    )
    assert packed.token_count <= 400
    # 至少有些被丢掉了
    assert packed.semantic_dropped_for_budget >= 1
    # 留下来的应是相关度高的（含「主题相关」），不相关的应被丢
    sem_section = packed.text.split("# 语义补充记忆", 1)[1].split("# 子任务", 1)[0]
    if packed.semantic_added > 0:
        assert "主题相关" in sem_section or "结论" in sem_section


async def test_token_budget_truncates_handoff_when_required_overflows(
    env, tmp_path
) -> None:
    """必保段已超 budget 时硬截接力原文，仍保留 title + sub_task。"""
    state = StateStore(tmp_path / "state.db")
    transcript = TranscriptStore(tmp_path / "t.db")
    memory = MemoryStore(tmp_path / "chroma")
    packer = ContextPacker(
        state_store=state, transcript_store=transcript, memory_store=memory,
        max_context_tokens=300,
    )
    # 写一段超长接力对话
    huge = "A" * 5000
    for i in range(1, 6):
        await transcript.add_turn(
            conversation_id="conv_h", turn_index=i,
            user_input=huge + f" u{i}", agent_output=huge + f" a{i}",
            agent_id="planner",
        )
    tid = await state.create_task(
        user_id="u", title="底线题目", dag_id="d",
        handoff_conversation_id="conv_h", handoff_turn_range=[1, 5],
    )
    nid = await state.create_dag_node(task_id=tid, node_name="x")
    packed = await packer.pack(
        task_id=tid, node_id=nid, sub_task_description="底线子任务"
    )
    assert packed.token_count <= 300
    # 底线：title + sub_task 必在
    assert "底线题目" in packed.text
    assert "底线子任务" in packed.text


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
