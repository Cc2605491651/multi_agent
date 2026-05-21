"""state_store 单测（阶段 2 任务 2.1+2.3）。"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from storage.state_store import StateStore


@pytest.fixture
def store(tmp_path: Path) -> StateStore:
    return StateStore(tmp_path / "state.db")


async def test_create_and_get_task(store: StateStore) -> None:
    tid = await store.create_task(
        user_id="alice",
        title="写一份调研报告",
        dag_id="research_report",
        handoff_conversation_id="conv_x",
        handoff_turn_range=[6, 8],
    )
    assert tid.startswith("task_")
    t = await store.get_task(tid)
    assert t is not None
    assert t.user_id == "alice"
    assert t.title == "写一份调研报告"
    assert t.dag_id == "research_report"
    assert t.handoff_conversation_id == "conv_x"
    assert t.handoff_turn_range == [6, 8]
    assert t.status == "pending"


async def test_update_task_status(store: StateStore) -> None:
    tid = await store.create_task(user_id="alice", title="t", dag_id="d")
    await store.update_task_status(tid, "running")
    assert (await store.get_task(tid)).status == "running"

    with pytest.raises(ValueError):
        await store.update_task_status(tid, "garbage")


async def test_create_and_list_nodes(store: StateStore) -> None:
    tid = await store.create_task(user_id="u", title="t", dag_id="d")
    n1 = await store.create_dag_node(
        task_id=tid, node_name="research", depends_on=[]
    )
    n2 = await store.create_dag_node(
        task_id=tid,
        node_name="writing",
        depends_on=[n1],
        failure_policy="fail_fast",
        max_retries=1,
    )
    nodes = await store.list_dag_nodes(tid)
    assert {n.id for n in nodes} == {n1, n2}

    n2_row = await store.get_dag_node(n2)
    assert n2_row.depends_on == [n1]
    assert n2_row.failure_policy == "fail_fast"
    assert n2_row.max_retries == 1
    assert n2_row.status == "pending"
    assert n2_row.input_memory_ids == []
    assert n2_row.retry_count == 0


async def test_invalid_failure_policy_rejected(store: StateStore) -> None:
    tid = await store.create_task(user_id="u", title="t", dag_id="d")
    with pytest.raises(ValueError):
        await store.create_dag_node(
            task_id=tid, node_name="x", failure_policy="invalid"
        )


async def test_claim_running_only_once(store: StateStore) -> None:
    tid = await store.create_task(user_id="u", title="t", dag_id="d")
    nid = await store.create_dag_node(task_id=tid, node_name="n")
    assert await store.claim_node_running(nid, "worker-1") is True
    # 第二次必须返回 False（状态已是 running）
    assert await store.claim_node_running(nid, "worker-2") is False
    n = await store.get_dag_node(nid)
    assert n.status == "running"
    assert n.worker_id == "worker-1"
    assert n.started_at is not None
    assert n.heartbeat_at is not None


async def test_heartbeat_only_updates_when_running(store: StateStore) -> None:
    tid = await store.create_task(user_id="u", title="t", dag_id="d")
    nid = await store.create_dag_node(task_id=tid, node_name="n")
    # pending 时心跳不写
    await store.update_heartbeat(nid)
    assert (await store.get_dag_node(nid)).heartbeat_at is None

    await store.claim_node_running(nid, "w1")
    n_before = await store.get_dag_node(nid)
    time.sleep(1.0)
    await store.update_heartbeat(nid)
    n_after = await store.get_dag_node(nid)
    assert n_after.heartbeat_at != n_before.heartbeat_at


async def test_commit_done_only_if_running(store: StateStore) -> None:
    tid = await store.create_task(user_id="u", title="t", dag_id="d")
    nid = await store.create_dag_node(task_id=tid, node_name="n")

    # pending 状态下 commit 失败
    assert await store.commit_node_done(nid, "mem_abc") is False

    await store.claim_node_running(nid, "w1")
    assert await store.commit_node_done(nid, "mem_abc") is True

    # done 后再 commit 也失败
    assert await store.commit_node_done(nid, "mem_xyz") is False
    n = await store.get_dag_node(nid)
    assert n.status == "done"
    assert n.output_memory_id == "mem_abc"
    assert n.finished_at is not None


async def test_commit_done_allows_null_memory(store: StateStore) -> None:
    tid = await store.create_task(user_id="u", title="t", dag_id="d")
    nid = await store.create_dag_node(task_id=tid, node_name="n")
    await store.claim_node_running(nid, "w1")
    assert await store.commit_node_done(nid, None) is True
    assert (await store.get_dag_node(nid)).output_memory_id is None


async def test_set_input_memory_ids(store: StateStore) -> None:
    tid = await store.create_task(user_id="u", title="t", dag_id="d")
    nid = await store.create_dag_node(task_id=tid, node_name="n")
    await store.set_node_input_memory_ids(nid, ["mem_1", "mem_2"])
    assert (await store.get_dag_node(nid)).input_memory_ids == ["mem_1", "mem_2"]


async def test_reset_to_pending_with_retry(store: StateStore) -> None:
    tid = await store.create_task(user_id="u", title="t", dag_id="d")
    nid = await store.create_dag_node(task_id=tid, node_name="n")
    await store.claim_node_running(nid, "w1")

    await store.reset_node_to_pending(nid, increment_retry=True)
    n = await store.get_dag_node(nid)
    assert n.status == "pending"
    assert n.worker_id is None
    assert n.heartbeat_at is None
    assert n.retry_count == 1


async def test_mark_terminal(store: StateStore) -> None:
    tid = await store.create_task(user_id="u", title="t", dag_id="d")
    nid = await store.create_dag_node(task_id=tid, node_name="n")
    await store.claim_node_running(nid, "w1")
    await store.mark_node_terminal(nid, "skipped")
    n = await store.get_dag_node(nid)
    assert n.status == "skipped"
    assert n.finished_at is not None
    with pytest.raises(ValueError):
        await store.mark_node_terminal(nid, "running")


async def test_find_stale_running(store: StateStore) -> None:
    tid = await store.create_task(user_id="u", title="t", dag_id="d")
    fresh = await store.create_dag_node(task_id=tid, node_name="fresh")
    stale = await store.create_dag_node(task_id=tid, node_name="stale")
    pending_only = await store.create_dag_node(task_id=tid, node_name="p")

    await store.claim_node_running(fresh, "w1")
    await store.claim_node_running(stale, "w2")
    # 把 stale 的心跳手工拨老
    import sqlite3

    with sqlite3.connect(store._db_path) as conn:
        conn.execute(
            "UPDATE dag_nodes SET heartbeat_at = '2020-01-01T00:00:00+00:00' WHERE id = ?",
            (stale,),
        )

    stales = await store.find_stale_running(threshold_seconds=60)
    ids = {n.id for n in stales}
    assert ids == {stale}
    assert pending_only not in ids


async def test_find_done_with_memory(store: StateStore) -> None:
    tid = await store.create_task(user_id="u", title="t", dag_id="d")
    a = await store.create_dag_node(task_id=tid, node_name="a")
    b = await store.create_dag_node(task_id=tid, node_name="b")
    await store.claim_node_running(a, "w1")
    await store.commit_node_done(a, "mem_a")
    await store.claim_node_running(b, "w2")
    await store.commit_node_done(b, None)  # 无 memory

    found = await store.find_done_with_memory()
    ids = {n.id for n in found}
    assert ids == {a}


async def test_find_terminal_nodes(store: StateStore) -> None:
    tid = await store.create_task(user_id="u", title="t", dag_id="d")
    a = await store.create_dag_node(task_id=tid, node_name="a")
    b = await store.create_dag_node(task_id=tid, node_name="b")
    c = await store.create_dag_node(task_id=tid, node_name="c")
    await store.claim_node_running(a, "w1")
    await store.mark_node_terminal(a, "failed")
    await store.claim_node_running(b, "w2")
    await store.mark_node_terminal(b, "skipped")
    # c 还在 pending，不应被找到
    found = await store.find_terminal_nodes()
    ids = {n.id for n in found}
    assert ids == {a, b}
