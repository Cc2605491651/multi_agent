"""spec v6 协作模型测试：多轮 transcript + 节点级接力。

覆盖 P1（多轮 transcript）+ P2（节点级 handoff），是用户最初要求的「后 agent
切入前 agent 某轮 session」功能的端到端验证。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from orchestrator.context_packer import ContextPacker
from orchestrator.dag_loader import instantiate_dag, parse_dag
from orchestrator.failure_handler import FailureHandler
from orchestrator.recovery import Recovery
from orchestrator.scheduler import Scheduler
from storage.memory_store import MemoryStore
from storage.state_store import StateStore
from storage.transcript_store import TranscriptStore
from worker.harness import AgentHarness, HandoffSpec, ToolSpec
from worker.sandbox import LocalBackend
from worker.tool_loop import ToolCallRecord, ToolLoopResult


# ---------- P1.2 add_tool_loop_turns ----------


@pytest.fixture
def transcript(tmp_path: Path) -> TranscriptStore:
    return TranscriptStore(tmp_path / "t.db")


async def test_add_tool_loop_turns_produces_multi_turn(transcript) -> None:
    loop = ToolLoopResult(
        final_text="最终结论：选 A",
        turns=3,
        tool_calls=[
            ToolCallRecord(
                tool_name="web_search",
                args={"query": "国产 RAG 框架"},
                result="找到 3 个候选",
                is_error=False,
            ),
            ToolCallRecord(
                tool_name="read_file",
                args={"path": "memo.md"},
                result="memo content",
                is_error=False,
            ),
        ],
        stop_reason="end_turn",
    )
    ids = await transcript.add_tool_loop_turns(
        conversation_id="conv_test_n1",
        agent_id="research_agent",
        initial_user_input="调研国内 RAG 框架",
        loop_result=loop,
    )
    # 预期：1 user + 2 (tool_call+tool_result) + 1 final = 6 turn
    assert len(ids) == 6

    rows = await transcript.get_turns_by_range("conv_test_n1", 1, 99)
    kinds = [r.turn_kind for r in rows]
    assert kinds == [
        "user", "tool_call", "tool_result",
        "tool_call", "tool_result", "final",
    ]
    # tool_call 有 turn_meta
    tc1 = rows[1]
    assert tc1.turn_meta["tool_name"] == "web_search"
    assert tc1.turn_meta["args"]["query"] == "国产 RAG 框架"
    # tool_result 是上一步那次调用的结果
    tr1 = rows[2]
    assert tr1.turn_meta["tool_use_id"] == tc1.turn_meta["tool_use_id"]
    assert tr1.agent_output == "找到 3 个候选"
    # final
    assert rows[-1].turn_kind == "final"
    assert rows[-1].agent_output == "最终结论：选 A"


async def test_count_turns(transcript) -> None:
    assert await transcript.count_turns("ghost") == 0
    await transcript.add_turn("c1", 1, "u", "a")
    await transcript.add_turn("c1", 2, "u", "a")
    assert await transcript.count_turns("c1") == 2


# ---------- P2.1 HandoffSpec ----------


def test_handoff_spec_round_trip() -> None:
    h = HandoffSpec.from_obj({"from_node": "n1", "turn_range": [1, 5]})
    assert h.from_node == "n1"
    assert h.turn_range == [1, 5]
    d = h.to_dict()
    h2 = HandoffSpec.from_obj(d)
    assert h2 == h


def test_handoff_spec_no_turn_range_means_all() -> None:
    h = HandoffSpec.from_obj({"from_node": "n2"})
    assert h.from_node == "n2"
    assert h.turn_range is None


@pytest.mark.parametrize(
    "bad",
    [
        {"from_node": "", "turn_range": [1, 2]},
        {"turn_range": [1, 2]},
        {"from_node": "n1", "turn_range": [5, 1]},
        {"from_node": "n1", "turn_range": [1]},
        {"from_node": "n1", "turn_range": "not-a-list"},
    ],
)
def test_handoff_spec_invalid_raises(bad: dict) -> None:
    with pytest.raises((ValueError, TypeError)):
        HandoffSpec.from_obj(bad)


def test_harness_with_handoff_round_trip() -> None:
    raw = {
        "model": "deepseek-chat",
        "provider": "deepseek",
        "handoff": {"from_node": "n_research", "turn_range": [1, 10]},
    }
    h = AgentHarness.from_dict(raw)
    assert h.handoff.from_node == "n_research"
    assert AgentHarness.from_dict(h.to_dict()) == h


# ---------- DAG instantiate 翻译 from_node ----------


async def test_instantiate_translates_handoff_from_node(tmp_path: Path) -> None:
    state = StateStore(tmp_path / "state.db")
    d = parse_dag(
        {
            "dag_id": "coop",
            "nodes": [
                {
                    "id": "n1", "name": "research", "deps": [],
                    "harness": {
                        "model": "deepseek-chat", "provider": "deepseek",
                        "tools": ["web_search"],
                    },
                },
                {
                    "id": "n2", "name": "reviewer", "deps": ["n1"],
                    "harness": {
                        "model": "deepseek-chat", "provider": "deepseek",
                        "handoff": {"from_node": "n1"},
                    },
                },
            ],
        }
    )
    _, mapping = await instantiate_dag(state, d, user_id="u", title="t")
    n2 = await state.get_dag_node(mapping["n2"])
    h2 = AgentHarness.from_dict(n2.harness_json)
    # 翻译后应是真实 node_id，不是 "n1"
    assert h2.handoff.from_node == mapping["n1"]
    assert h2.handoff.from_node != "n1"


async def test_instantiate_unknown_handoff_from_node_raises(tmp_path: Path) -> None:
    state = StateStore(tmp_path / "state.db")
    d = parse_dag(
        {
            "dag_id": "bad_handoff",
            "nodes": [
                {
                    "id": "n1", "name": "x", "deps": [],
                    "harness": {
                        "handoff": {"from_node": "ghost_node"},
                    },
                },
            ],
        }
    )
    with pytest.raises(ValueError, match="not a defined node"):
        await instantiate_dag(state, d, user_id="u", title="t")


# ---------- P2.2 context_packer 节点级接力 ----------


@pytest.fixture
def packer_env(tmp_path: Path):
    state = StateStore(tmp_path / "state.db")
    transcript = TranscriptStore(tmp_path / "t.db")
    memory = MemoryStore(tmp_path / "chroma")
    packer = ContextPacker(
        state_store=state, transcript_store=transcript, memory_store=memory
    )
    return state, transcript, memory, packer


async def test_context_packer_pulls_node_handoff_multi_turn(packer_env) -> None:
    state, transcript, memory, packer = packer_env

    tid = await state.create_task(user_id="u", title="测试协作", dag_id="d")

    # n1：scheduler 用的 conv_id 约定 = conv_{task_id}_{node_id}
    n1_id = await state.create_dag_node(task_id=tid, node_name="research")
    conv_n1 = f"conv_{tid}_{n1_id}"
    # 模拟 n1 跑完，多轮 transcript 已写
    loop = ToolLoopResult(
        final_text="结论：建议方案 X",
        turns=2,
        tool_calls=[
            ToolCallRecord(
                tool_name="web_search",
                args={"query": "方案对比"},
                result="搜到 3 篇文章",
                is_error=False,
            ),
        ],
        stop_reason="end_turn",
    )
    await transcript.add_tool_loop_turns(
        conversation_id=conv_n1,
        agent_id="research",
        initial_user_input="对比方案",
        loop_result=loop,
    )
    # 同时给 n1 一个 output_memory（不然 input_memory_ids 没东西）
    mid = await memory.add(
        "u", "n1 提炼后的一句话",
        {
            "task_id": tid, "produced_by_node": n1_id,
            "produced_by_agent": "research",
            "memory_level": "node_output", "status": "active",
        },
    )
    # 把 n1 standalone 标 done + output_memory_id
    await state.claim_node_running(n1_id, "w")
    await state.commit_node_done(n1_id, mid)

    # n2：声明 handoff from n1
    harness_n2 = AgentHarness(
        model="deepseek-chat", provider="deepseek",
        system_prompt="你是 reviewer，看上游 agent 的完整思考链路",
        handoff=HandoffSpec(from_node=n1_id, turn_range=None),  # 全 4 turn
    )
    n2_id = await state.create_dag_node(
        task_id=tid, node_name="reviewer",
        depends_on=[n1_id], harness=harness_n2,
    )
    await state.set_node_input_memory_ids(n2_id, [mid])

    packed = await packer.pack(
        task_id=tid, node_id=n2_id,
        sub_task_description="评审上游调研",
    )

    # 关键断言：packed.text 含 n1 的 tool_call/tool_result 多轮原文
    assert packed.node_handoff_present is True
    assert packed.node_handoff_turn_count == 4  # 1 user + 1 tc + 1 tr + 1 final
    assert "# 上游节点接力（多轮原文）" in packed.text
    assert "工具调用 [web_search]" in packed.text
    assert "方案对比" in packed.text  # tool args 进去了
    assert "搜到 3 篇文章" in packed.text  # tool result 进去了
    assert "结论：建议方案 X" in packed.text  # final 也进去了


async def test_context_packer_handoff_turn_range_subset(packer_env) -> None:
    """turn_range 只取一部分。"""
    state, transcript, memory, packer = packer_env
    tid = await state.create_task(user_id="u", title="t", dag_id="d")
    n1_id = await state.create_dag_node(task_id=tid, node_name="up")
    conv = f"conv_{tid}_{n1_id}"
    for i in range(1, 6):
        await transcript.add_turn(
            conversation_id=conv, turn_index=i,
            user_input=f"u{i}", agent_output=f"a{i}",
            agent_id="up", turn_kind="single",
        )

    harness = AgentHarness(handoff=HandoffSpec(from_node=n1_id, turn_range=[2, 3]))
    n2_id = await state.create_dag_node(
        task_id=tid, node_name="down", depends_on=[n1_id], harness=harness
    )

    packed = await packer.pack(
        task_id=tid, node_id=n2_id, sub_task_description="x"
    )
    assert packed.node_handoff_turn_count == 2
    assert "u2" in packed.text
    assert "u3" in packed.text
    assert "u4" not in packed.text


async def test_context_packer_no_handoff_section_when_absent(packer_env) -> None:
    state, transcript, memory, packer = packer_env
    tid = await state.create_task(user_id="u", title="t", dag_id="d")
    n_id = await state.create_dag_node(task_id=tid, node_name="x")
    packed = await packer.pack(
        task_id=tid, node_id=n_id, sub_task_description="x"
    )
    assert packed.node_handoff_present is False
    assert "# 上游节点接力" not in packed.text


# ---------- scheduler 端到端：n1 多轮 → n2 接力看见全过程 ----------


@dataclass
class _ScriptedClient:
    """返回 plain final_text（不真走 tool loop，但通过 mock 模拟 loop_result）。"""

    chat_map: dict[str, str] = field(default_factory=dict)
    calls: list[dict] = field(default_factory=list)

    async def complete(self, *, model, system, messages, max_tokens=1024):
        text = messages[-1]["content"]
        self.calls.append({"system": system, "text": text})
        if "提炼员" in system:
            tag = "【Agent 输出】"
            if tag in text:
                tail = text.split(tag, 1)[1].split("【", 1)[0].strip()
                return tail[:120]
            return text[:120]
        for key, out in self.chat_map.items():
            if key in text:
                return out
        return "default"


async def test_scheduler_node_handoff_pipes_multi_turn_through(tmp_path: Path) -> None:
    """spec v6：n1 单轮（无 tool loop）→ n2 仍能通过 handoff 看到 n1 final turn 原文。"""
    state = StateStore(tmp_path / "state.db")
    transcript = TranscriptStore(tmp_path / "t.db")
    memory = MemoryStore(tmp_path / "chroma")
    sandbox = LocalBackend(root_dir=tmp_path / "sb")
    recovery = Recovery(state, memory, stale_seconds=60)
    packer = ContextPacker(
        state_store=state, transcript_store=transcript, memory_store=memory
    )
    failure = FailureHandler(state, memory)

    d = parse_dag({
        "dag_id": "coop",
        "nodes": [
            {
                "id": "n1", "name": "research", "deps": [],
                "harness": {
                    "system_prompt": "你是 research agent",
                },
            },
            {
                "id": "n2", "name": "reviewer", "deps": ["n1"],
                "harness": {
                    "system_prompt": "你是 reviewer",
                    "handoff": {"from_node": "n1"},
                },
            },
        ],
    })
    tid, mapping = await instantiate_dag(
        state, d, user_id="default_user", title="协作测试",
    )

    client = _ScriptedClient(chat_map={
        "research": "方案 X 是首选，因为性价比 ratio=2.3",
        "reviewer": "已收到上游分析",
    })
    scheduler = Scheduler(
        state_store=state, transcript_store=transcript, memory_store=memory,
        sandbox=sandbox, llm_client=client, recovery=recovery,
        context_packer=packer, failure_handler=failure,
        sub_task_builder=lambda node, ctx: f"[{node.node_name}] do task",
        heartbeat_interval=10.0,
        force_default_client=True,
    )
    final = await scheduler.run_task(tid)
    assert final == "done"

    # 关键断言：reviewer 节点收到的 prompt 里含 research 节点的原文
    reviewer_chat = next(
        c for c in client.calls
        if "reviewer" in c["text"] and "提炼员" not in c["system"]
    )
    assert "# 上游节点接力（多轮原文）" in reviewer_chat["text"]
    # n1 是单轮节点，转 transcript 是 turn_kind="single"
    assert "方案 X 是首选，因为性价比 ratio=2.3" in reviewer_chat["text"]
