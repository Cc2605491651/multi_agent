"""failure_handler 单测（阶段 4a 任务 4a.2）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.failure_handler import FailureHandler
from storage.memory_store import MemoryStore
from storage.state_store import StateStore


@pytest.fixture
def env(tmp_path: Path):
    state = StateStore(tmp_path / "state.db")
    memory = MemoryStore(tmp_path / "chroma")
    handler = FailureHandler(state, memory)
    return state, memory, handler


async def _make_running_node(
    state: StateStore,
    *,
    failure_policy: str = "fail_retry",
    max_retries: int = 2,
):
    tid = await state.create_task(user_id="u", title="t", dag_id="d")
    nid = await state.create_dag_node(
        task_id=tid,
        node_name="n",
        failure_policy=failure_policy,
        max_retries=max_retries,
    )
    await state.claim_node_running(nid, "w")
    return tid, nid


# ---- retry 路径 ----


async def test_retry_resets_to_pending_increments_count(env) -> None:
    state, _, handler = env
    _, nid = await _make_running_node(state, max_retries=2)
    node = await state.get_dag_node(nid)

    action = await handler.on_node_failed(node)
    assert action.kind == "retry"
    assert action.task_should_fail is False
    assert action.cancel_siblings is False
    n = await state.get_dag_node(nid)
    assert n.status == "pending"
    assert n.retry_count == 1


async def test_retry_path_cleans_pending_memory(env) -> None:
    state, memory, handler = env
    tid, nid = await _make_running_node(state, max_retries=2)
    # 模拟该节点写过 pending 记忆
    mid = await memory.add(
        "u",
        "脏数据",
        {
            "task_id": tid,
            "produced_by_node": nid,
            "produced_by_agent": "x",
            "memory_level": "node_output",
            "status": "pending",
        },
    )
    node = await state.get_dag_node(nid)
    await handler.on_node_failed(node)
    # retry 路径必须清掉本节点的 pending 半成品（spec §5.2 「重试前先按 §6.3 清理」）
    assert await memory.get_status("u", mid) is None


# ---- 重试耗尽 ----


async def test_exhausted_fail_retry_terminates_failed(env) -> None:
    state, _, handler = env
    _, nid = await _make_running_node(
        state, failure_policy="fail_retry", max_retries=0
    )
    node = await state.get_dag_node(nid)
    action = await handler.on_node_failed(node)
    assert action.kind == "fail"
    assert action.task_should_fail is True
    assert action.cancel_siblings is False
    assert (await state.get_dag_node(nid)).status == "failed"


async def test_exhausted_fail_skip_terminates_skipped(env) -> None:
    state, _, handler = env
    _, nid = await _make_running_node(
        state, failure_policy="fail_skip", max_retries=0
    )
    node = await state.get_dag_node(nid)
    action = await handler.on_node_failed(node)
    assert action.kind == "skip"
    assert action.task_should_fail is False  # 任务继续！
    assert action.cancel_siblings is False
    assert (await state.get_dag_node(nid)).status == "skipped"


async def test_exhausted_fail_fast_cancels_siblings(env) -> None:
    state, _, handler = env
    _, nid = await _make_running_node(
        state, failure_policy="fail_fast", max_retries=0
    )
    node = await state.get_dag_node(nid)
    action = await handler.on_node_failed(node)
    assert action.kind == "fail"
    assert action.task_should_fail is True
    assert action.cancel_siblings is True  # spec §5.3 关键！
    assert (await state.get_dag_node(nid)).status == "failed"


# ---- 多次重试一直到耗尽 ----


async def test_multi_retry_until_exhausted(env) -> None:
    state, _, handler = env
    _, nid = await _make_running_node(
        state, failure_policy="fail_retry", max_retries=2
    )

    for expected_retry in range(1, 3):
        node = await state.get_dag_node(nid)
        action = await handler.on_node_failed(node)
        assert action.kind == "retry"
        assert (await state.get_dag_node(nid)).retry_count == expected_retry
        # 模拟重新拉起
        await state.claim_node_running(nid, f"w{expected_retry}")

    # 第 3 次失败 → 重试耗尽
    node = await state.get_dag_node(nid)
    action = await handler.on_node_failed(node)
    assert action.kind == "fail"
    assert action.task_should_fail is True


# ---- 下游 skip 联动 ----


async def test_downstream_marked_skipped_after_upstream_failed(env) -> None:
    state, _, handler = env
    tid = await state.create_task(user_id="u", title="t", dag_id="d")
    upstream = await state.create_dag_node(task_id=tid, node_name="up")
    downstream = await state.create_dag_node(
        task_id=tid, node_name="down", depends_on=[upstream]
    )
    # downstream pending，未运行
    n_down = await state.get_dag_node(downstream)
    await handler.on_dependency_terminal_failed(n_down)
    assert (await state.get_dag_node(downstream)).status == "skipped"
