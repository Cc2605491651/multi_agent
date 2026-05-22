"""Scheduler 集成测（阶段 2/3/4a 验收）。

阶段 4a 新增覆盖：
- 并发节点同时跑
- fail_retry 重试成功
- fail_skip 重试耗尽 → skipped + 下游正常
- fail_fast 重试耗尽 → 取消兄弟 + 任务 failed
- 上游 fail_retry 重试耗尽 → 下游 cascade skipped
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from orchestrator.context_packer import ContextPacker
from orchestrator.failure_handler import FailureHandler
from orchestrator.recovery import Recovery
from orchestrator.scheduler import Scheduler
from storage.memory_store import MemoryStore
from storage.state_store import StateStore
from storage.transcript_store import TranscriptStore
from worker.sandbox import LocalBackend


@dataclass
class _ScriptedClient:
    """根据 user prompt 关键词返回 mock 输出。"""

    chat_map: dict[str, str]
    fail_for_nodes: set[str] = field(default_factory=set)
    """这些 node tag 出现在 prompt 时抛错（用于模拟节点失败）。"""

    fail_counter: dict[str, int] = field(default_factory=dict)
    """每个 node tag 还需要失败几次（用于模拟 fail_retry 成功）。"""

    chat_inputs: list[str] = field(default_factory=list)
    sleep_seconds: float = 0.0

    async def complete(self, *, model: str, system: str, messages, max_tokens=1024):
        text = messages[-1]["content"]
        if "提炼员" in system:
            tag = "【Agent 输出】"
            if tag in text:
                tail = text.split(tag, 1)[1]
                for stop in ("\n\n请按", "\n请按", "【"):
                    if stop in tail:
                        tail = tail.split(stop, 1)[0]
                        break
                return tail.strip()[:160]
            return text.strip()[:160]

        self.chat_inputs.append(text)
        if self.sleep_seconds > 0:
            await asyncio.sleep(self.sleep_seconds)

        for tag in self.fail_for_nodes:
            if tag in text:
                raise RuntimeError(f"scripted failure for {tag}")

        for tag, remaining in list(self.fail_counter.items()):
            if tag in text and remaining > 0:
                self.fail_counter[tag] = remaining - 1
                raise RuntimeError(f"scripted retry failure for {tag} (left={remaining-1})")

        for key, out in self.chat_map.items():
            if key in text:
                return out
        return "default"


@pytest.fixture
def env(tmp_path: Path):
    state = StateStore(tmp_path / "state.db")
    transcript = TranscriptStore(tmp_path / "transcript.db")
    memory = MemoryStore(tmp_path / "chroma")
    sandbox = LocalBackend(root_dir=tmp_path / "sb")
    recovery = Recovery(state, memory, stale_seconds=60)
    packer = ContextPacker(
        state_store=state, transcript_store=transcript, memory_store=memory
    )
    failure = FailureHandler(state, memory)
    return state, transcript, memory, sandbox, recovery, packer, failure


def _sub_task(node, ctx) -> str:
    return f"[node:{node.node_name}] {ctx.title}"


def _build(env, client, *, max_concurrent=5) -> Scheduler:
    state, transcript, memory, sandbox, recovery, packer, failure = env
    return Scheduler(
        state_store=state,
        transcript_store=transcript,
        memory_store=memory,
        sandbox=sandbox,
        llm_client=client,
        recovery=recovery,
        context_packer=packer,
        failure_handler=failure,
        sub_task_builder=_sub_task,
        max_concurrent_workers=max_concurrent,
        heartbeat_interval=10.0,
        cancel_timeout=1.0,
    )


# ============ 阶段 2/3 基线 ============


async def test_serial_two_node_dag_done(env) -> None:
    state, *_ = env
    client = _ScriptedClient(
        chat_map={
            "node:research": "事实1；事实2；事实3",
            "node:writing": "综合：123",
        }
    )
    tid = await state.create_task(user_id="default_user", title="t", dag_id="d")
    n1 = await state.create_dag_node(task_id=tid, node_name="research")
    n2 = await state.create_dag_node(
        task_id=tid, node_name="writing", depends_on=[n1]
    )

    assert await _build(env, client).run_task(tid) == "done"


async def test_downstream_receives_input_memory_ids(env) -> None:
    state, *_ = env
    client = _ScriptedClient(
        chat_map={"node:research": "A、B、C", "node:writing": "综合 A B C"}
    )
    tid = await state.create_task(user_id="default_user", title="t", dag_id="d")
    n1 = await state.create_dag_node(task_id=tid, node_name="research")
    n2 = await state.create_dag_node(
        task_id=tid, node_name="writing", depends_on=[n1]
    )
    await _build(env, client).run_task(tid)
    n1_row = await state.get_dag_node(n1)
    n2_row = await state.get_dag_node(n2)
    assert n2_row.input_memory_ids == [n1_row.output_memory_id]


async def test_downstream_context_contains_upstream_raw_output(env) -> None:
    state, *_ = env
    upstream_doc = "方案 X：因为 Y，所以选 X"
    client = _ScriptedClient(
        chat_map={"node:research": upstream_doc, "node:writing": "已收到方案 X"}
    )
    tid = await state.create_task(user_id="default_user", title="选型", dag_id="d")
    n1 = await state.create_dag_node(task_id=tid, node_name="research")
    n2 = await state.create_dag_node(
        task_id=tid, node_name="writing", depends_on=[n1]
    )
    await _build(env, client).run_task(tid)
    writing_chat = client.chat_inputs[1]
    assert "方案 X" in writing_chat
    assert "因为 Y" in writing_chat


# ============ 阶段 4a 新增 ============


async def test_three_research_nodes_run_concurrently(env) -> None:
    """3 个独立 research 节点应能并发跑（总耗时 ≈ 单节点 sleep，不是 3 倍）。"""
    state, *_ = env
    client = _ScriptedClient(
        chat_map={
            "node:research_a": "A",
            "node:research_b": "B",
            "node:research_c": "C",
        },
        sleep_seconds=0.3,
    )
    tid = await state.create_task(user_id="default_user", title="t", dag_id="d")
    await state.create_dag_node(task_id=tid, node_name="research_a")
    await state.create_dag_node(task_id=tid, node_name="research_b")
    await state.create_dag_node(task_id=tid, node_name="research_c")

    import time

    start = time.perf_counter()
    assert await _build(env, client, max_concurrent=3).run_task(tid) == "done"
    elapsed = time.perf_counter() - start
    # 顺序跑 ≥ 0.9s；并发应 ≤ 0.7s（留余量）
    assert elapsed < 0.7, f"expected concurrent execution, took {elapsed:.2f}s"


async def test_max_concurrent_workers_serializes_extras(env) -> None:
    state, *_ = env
    client = _ScriptedClient(
        chat_map={f"node:n{i}": str(i) for i in range(4)},
        sleep_seconds=0.3,
    )
    tid = await state.create_task(user_id="default_user", title="t", dag_id="d")
    for i in range(4):
        await state.create_dag_node(task_id=tid, node_name=f"n{i}")

    import time

    start = time.perf_counter()
    await _build(env, client, max_concurrent=2).run_task(tid)
    elapsed = time.perf_counter() - start
    # 4 个节点 / 并发 2 = 2 批 → ~0.6s
    assert 0.5 < elapsed < 1.1, f"unexpected timing {elapsed:.2f}s"


async def test_fail_retry_succeeds_after_one_retry(env) -> None:
    state, *_ = env
    client = _ScriptedClient(
        chat_map={"node:research": "OK"},
        fail_counter={"node:research": 1},  # 第 1 次失败，第 2 次成功
    )
    tid = await state.create_task(user_id="default_user", title="t", dag_id="d")
    nid = await state.create_dag_node(
        task_id=tid, node_name="research",
        failure_policy="fail_retry", max_retries=2,
    )

    assert await _build(env, client).run_task(tid) == "done"
    n = await state.get_dag_node(nid)
    assert n.status == "done"
    assert n.retry_count == 1


async def test_fail_skip_after_exhaust_lets_downstream_run(env) -> None:
    """spec §5.2：fail_skip 节点重试耗尽 → skipped；下游正常执行（拿不到本节点产出）。"""
    state, *_ = env
    client = _ScriptedClient(
        chat_map={"node:downstream": "下游 OK"},
        fail_for_nodes={"node:skip_me"},
    )
    tid = await state.create_task(user_id="default_user", title="t", dag_id="d")
    skip_id = await state.create_dag_node(
        task_id=tid, node_name="skip_me",
        failure_policy="fail_skip", max_retries=1,
    )
    down_id = await state.create_dag_node(
        task_id=tid, node_name="downstream",
        depends_on=[skip_id], failure_policy="fail_retry",
    )

    final = await _build(env, client).run_task(tid)
    assert final == "done"  # spec §5.2 表：fail_skip 不影响任务终态
    assert (await state.get_dag_node(skip_id)).status == "skipped"
    assert (await state.get_dag_node(down_id)).status == "done"


async def test_fail_fast_cancels_siblings_and_fails_task(env) -> None:
    """spec §5.3：fail_fast 重试耗尽 → 任务 failed + 取消并发兄弟。"""
    state, *_ = env
    client = _ScriptedClient(
        chat_map={"node:slow_a": "A", "node:slow_b": "B"},
        fail_for_nodes={"node:boom"},
        sleep_seconds=0.3,  # 兄弟节点 sleep，给 cancel 时间
    )
    tid = await state.create_task(user_id="default_user", title="t", dag_id="d")
    a = await state.create_dag_node(
        task_id=tid, node_name="slow_a", failure_policy="fail_retry"
    )
    b = await state.create_dag_node(
        task_id=tid, node_name="slow_b", failure_policy="fail_retry"
    )
    boom = await state.create_dag_node(
        task_id=tid, node_name="boom",
        failure_policy="fail_fast", max_retries=0,
    )

    final = await _build(env, client).run_task(tid)
    assert final == "failed"
    assert (await state.get_dag_node(boom)).status == "failed"
    # 兄弟节点应被取消（status 是 done / cancelled / failed 都可能；关键是任务 failed）
    sibs = [await state.get_dag_node(a), await state.get_dag_node(b)]
    # 至少有节点没顺利 done（cancel 把它打断）
    statuses = {n.status for n in sibs}
    assert "done" not in statuses or len(statuses) > 1, (
        f"expected siblings cancelled, got {statuses}"
    )


async def test_upstream_fail_retry_cascades_skip_to_downstream(env) -> None:
    state, *_ = env
    client = _ScriptedClient(
        chat_map={"node:final": "OK"},
        fail_for_nodes={"node:up"},
    )
    tid = await state.create_task(user_id="default_user", title="t", dag_id="d")
    up = await state.create_dag_node(
        task_id=tid, node_name="up",
        failure_policy="fail_retry", max_retries=0,
    )
    down = await state.create_dag_node(
        task_id=tid, node_name="final", depends_on=[up]
    )

    assert await _build(env, client).run_task(tid) == "failed"
    assert (await state.get_dag_node(up)).status == "failed"
    assert (await state.get_dag_node(down)).status == "skipped"


async def test_scheduler_uses_tool_loop_when_harness_declares_tools(env, monkeypatch) -> None:
    """阶段 ABC.B.4：harness 声明 tools + OpenAI 兼容 client → 真走 tool-use loop。"""
    import json as _json
    from pathlib import Path

    import httpx

    from orchestrator.context_packer import ContextPacker
    from orchestrator.failure_handler import FailureHandler
    from orchestrator.recovery import Recovery
    from orchestrator.scheduler import Scheduler
    from storage.memory_store import MemoryStore
    from storage.state_store import StateStore
    from storage.transcript_store import TranscriptStore
    from worker.harness import AgentHarness, ToolSpec
    from worker.llm_clients import OpenAICompatibleClient
    from worker.sandbox import LocalBackend

    # 脚本化两轮：1) 模型说调 run_code 2) 拿到结果给最终文本
    scripted = [
        {"choices": [{
            "message": {
                "role": "assistant", "content": None,
                "tool_calls": [{
                    "id": "c1", "type": "function",
                    "function": {
                        "name": "run_code",
                        "arguments": _json.dumps({"code": "print('ABC.B done')"}),
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }]},
        {"choices": [{
            "message": {"role": "assistant", "content": "已运行代码，结论：ABC.B done"},
            "finish_reason": "stop",
        }]},
    ]
    state_holder = {"idx": 0, "requests": []}

    def handler(request: httpx.Request) -> httpx.Response:
        body = _json.loads(request.content.decode())
        state_holder["requests"].append(body)
        # writeback 的 distill 也走同一 client；用 system prompt 关键词区分
        for m in body.get("messages", []):
            if m.get("role") == "system" and "提炼员" in (m.get("content") or ""):
                return httpx.Response(
                    200,
                    json={"choices": [{
                        "message": {"role": "assistant", "content": "mock 提炼结论"},
                        "finish_reason": "stop",
                    }]},
                )
        if state_holder["idx"] >= len(scripted):
            return httpx.Response(500, json={"error": "no more scripted"})
        resp = scripted[state_holder["idx"]]
        state_holder["idx"] += 1
        return httpx.Response(200, json=resp)

    real_async = httpx.AsyncClient

    class _Patched(real_async):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)
    monkeypatch.setattr(httpx, "AsyncClient", _Patched)

    state, *_ = env
    tid = await state.create_task(user_id="default_user", title="t", dag_id="d")
    harness = AgentHarness(
        model="gpt-4o-mini", provider="openai",
        tools=[ToolSpec(name="run_code"), ToolSpec(name="read_file")],
    )
    nid = await state.create_dag_node(
        task_id=tid, node_name="research_w_tools", harness=harness
    )

    # 构造一个 OpenAI 兼容 mock client；force_default_client=True 让 scheduler 不去
    # 重新 make_llm_client（避免 ANTHROPIC_API_KEY 必填）
    mock_client = OpenAICompatibleClient(
        base_url="https://mock/v1", api_key="k",
        default_model="gpt-4o-mini",
    )

    scheduler = _build(env, mock_client)
    # 替换 force flag：因为 _build 默认设 default_provider=anthropic，
    # 而我们的 mock 是 OpenAI provider；用 force_default_client 让它仍走 mock_client
    scheduler._force_default = True

    final = await scheduler.run_task(tid)
    assert final == "done"
    n = await state.get_dag_node(nid)
    assert n.status == "done"

    # 第 2 轮请求里应该有 role=tool 消息（OpenAI 协议）
    second_req = state_holder["requests"][1]
    roles = [m["role"] for m in second_req["messages"]]
    assert "tool" in roles, f"expected tool role in second request, got {roles}"


async def test_recovery_runs_before_task(env) -> None:
    state, *_ = env
    client = _ScriptedClient(
        chat_map={"node:research": "OK", "node:writing": "OK"}
    )
    tid = await state.create_task(user_id="default_user", title="t", dag_id="d")
    n1 = await state.create_dag_node(task_id=tid, node_name="research")
    await state.create_dag_node(
        task_id=tid, node_name="writing", depends_on=[n1]
    )

    await state.claim_node_running(n1, "ghost")
    import sqlite3

    with sqlite3.connect(state._db_path) as conn:
        conn.execute(
            "UPDATE dag_nodes SET heartbeat_at = '2020-01-01T00:00:00.000+00:00' WHERE id = ?",
            (n1,),
        )

    assert await _build(env, client).run_task(tid) == "done"
    n = await state.get_dag_node(n1)
    assert n.status == "done"
    assert n.retry_count == 1
