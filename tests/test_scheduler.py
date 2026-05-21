"""Scheduler 集成测（阶段 2 任务 2.8 验收 demo 的自动化版本）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

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
    calls: list[dict] = field(default_factory=list)

    async def complete(self, *, model: str, system: str, messages, max_tokens=1024):
        self.calls.append({"model": model, "system": system})
        text = messages[-1]["content"]
        if "提炼员" in system:
            tag = "【用户输入】"
            if tag in text:
                text = text.split(tag, 1)[1].split("【", 1)[0]
            return text.strip()[:60]
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
    return state, transcript, memory, sandbox, recovery


def _prompt(node, input_mems, ctx) -> str:
    if node.node_name == "research":
        return f"[research] 围绕 {ctx.title}：列出 3 条事实"
    if node.node_name == "writing":
        bg = "\n".join(m["document"] for m in input_mems) or "(无)"
        return f"[writing] 根据上游：\n{bg}\n请写总结"
    return f"[{node.node_name}]"


async def test_serial_two_node_dag_done(env) -> None:
    state, transcript, memory, sandbox, recovery = env
    client = _ScriptedClient(
        chat_map={
            "[research]": "事实1；事实2；事实3",
            "[writing]": "综合总结：基于上游 3 条事实",
        }
    )
    tid = await state.create_task(
        user_id="default_user", title="护理任务", dag_id="d"
    )
    n1 = await state.create_dag_node(task_id=tid, node_name="research")
    n2 = await state.create_dag_node(
        task_id=tid, node_name="writing", depends_on=[n1]
    )

    scheduler = Scheduler(
        state_store=state,
        transcript_store=transcript,
        memory_store=memory,
        sandbox=sandbox,
        llm_client=client,
        recovery=recovery,
        prompt_builder=_prompt,
        heartbeat_interval=10.0,
    )
    final = await scheduler.run_task(tid)
    assert final == "done"

    task = await state.get_task(tid)
    assert task.status == "done"

    nodes = await state.list_dag_nodes(tid)
    by_name = {n.node_name: n for n in nodes}
    assert by_name["research"].status == "done"
    assert by_name["writing"].status == "done"
    assert by_name["research"].output_memory_id is not None
    assert by_name["writing"].output_memory_id is not None


async def test_downstream_receives_input_memory_ids(env) -> None:
    state, transcript, memory, sandbox, recovery = env
    client = _ScriptedClient(
        chat_map={
            "[research]": "重要事实 A、B、C",
            "[writing]": "综合：A、B、C 已收到",
        }
    )
    tid = await state.create_task(
        user_id="default_user", title="t", dag_id="d"
    )
    n1 = await state.create_dag_node(task_id=tid, node_name="research")
    n2 = await state.create_dag_node(
        task_id=tid, node_name="writing", depends_on=[n1]
    )

    scheduler = Scheduler(
        state_store=state,
        transcript_store=transcript,
        memory_store=memory,
        sandbox=sandbox,
        llm_client=client,
        recovery=recovery,
        prompt_builder=_prompt,
        heartbeat_interval=10.0,
    )
    await scheduler.run_task(tid)

    n1_row = await state.get_dag_node(n1)
    n2_row = await state.get_dag_node(n2)
    # spec §3.3：调度某节点前把上游 done 的 output_memory_id 填进 input_memory_ids
    assert n2_row.input_memory_ids == [n1_row.output_memory_id]
    # 上游产出 + 下游写作 都已 active
    upstream_status = await memory.get_status(
        "default_user", n1_row.output_memory_id
    )
    downstream_status = await memory.get_status(
        "default_user", n2_row.output_memory_id
    )
    assert upstream_status == "active"
    assert downstream_status == "active"


async def test_recovery_runs_before_task(env) -> None:
    """启动前 recovery sweep：故意留一个 stale running 节点，看是否被回收。"""
    state, transcript, memory, sandbox, recovery = env
    client = _ScriptedClient(
        chat_map={"[research]": "OK", "[writing]": "OK"}
    )
    tid = await state.create_task(
        user_id="default_user", title="t", dag_id="d"
    )
    n1 = await state.create_dag_node(task_id=tid, node_name="research")
    n2 = await state.create_dag_node(
        task_id=tid, node_name="writing", depends_on=[n1]
    )

    # 模拟 research 之前被 kill：状态 running + 心跳老化
    await state.claim_node_running(n1, "ghost_worker")
    import sqlite3

    with sqlite3.connect(state._db_path) as conn:
        conn.execute(
            "UPDATE dag_nodes SET heartbeat_at = '2020-01-01T00:00:00.000+00:00' WHERE id = ?",
            (n1,),
        )

    scheduler = Scheduler(
        state_store=state,
        transcript_store=transcript,
        memory_store=memory,
        sandbox=sandbox,
        llm_client=client,
        recovery=recovery,
        prompt_builder=_prompt,
        heartbeat_interval=10.0,
    )
    final = await scheduler.run_task(tid)
    assert final == "done"

    n1_row = await state.get_dag_node(n1)
    assert n1_row.status == "done"
    assert n1_row.retry_count == 1  # 因为 recovery 重置时计数 +1


async def test_task_marked_failed_when_node_fails(env) -> None:
    state, transcript, memory, sandbox, recovery = env

    @dataclass
    class _Boom:
        async def complete(self, *, model, system, messages, max_tokens=1024):
            raise RuntimeError("simulated LLM crash")

    tid = await state.create_task(
        user_id="default_user", title="t", dag_id="d"
    )
    await state.create_dag_node(task_id=tid, node_name="x")
    scheduler = Scheduler(
        state_store=state,
        transcript_store=transcript,
        memory_store=memory,
        sandbox=sandbox,
        llm_client=_Boom(),
        recovery=recovery,
        prompt_builder=_prompt,
        heartbeat_interval=10.0,
    )
    final = await scheduler.run_task(tid)
    assert final == "failed"
