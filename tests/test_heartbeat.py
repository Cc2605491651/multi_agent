"""HeartbeatTask 单测（阶段 2 任务 2.5）。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from storage.state_store import StateStore
from worker.heartbeat import HeartbeatTask


@pytest.fixture
def state_store(tmp_path: Path) -> StateStore:
    return StateStore(tmp_path / "state.db")


async def _make_running_node(state_store: StateStore) -> str:
    tid = await state_store.create_task(user_id="u", title="t", dag_id="d")
    nid = await state_store.create_dag_node(task_id=tid, node_name="n")
    await state_store.claim_node_running(nid, "w1")
    return nid


async def test_heartbeat_writes_within_one_interval(state_store: StateStore) -> None:
    nid = await _make_running_node(state_store)
    import sqlite3

    with sqlite3.connect(state_store._db_path) as conn:
        conn.execute(
            "UPDATE dag_nodes SET heartbeat_at = NULL WHERE id = ?", (nid,)
        )
    assert (await state_store.get_dag_node(nid)).heartbeat_at is None

    async with HeartbeatTask(state_store, nid, interval=0.1):
        await asyncio.sleep(0.25)
        n = await state_store.get_dag_node(nid)
        assert n.heartbeat_at is not None


async def test_heartbeat_ticks_periodically(state_store: StateStore) -> None:
    nid = await _make_running_node(state_store)
    async with HeartbeatTask(state_store, nid, interval=0.2):
        n1 = await state_store.get_dag_node(nid)
        await asyncio.sleep(0.5)
        n2 = await state_store.get_dag_node(nid)
        assert n2.heartbeat_at != n1.heartbeat_at


async def test_heartbeat_cancels_cleanly_on_exit(state_store: StateStore) -> None:
    nid = await _make_running_node(state_store)
    task_ref = []

    async with HeartbeatTask(state_store, nid, interval=0.1) as hb:
        task_ref.append(hb._task)
        await asyncio.sleep(0.2)
    # 退出后 task 应被取消
    assert task_ref[0].cancelled() or task_ref[0].done()


async def test_heartbeat_survives_transient_store_errors(
    state_store: StateStore, monkeypatch
) -> None:
    """模拟 update_heartbeat 偶发异常，主流程不应被中断。"""
    nid = await _make_running_node(state_store)
    call_count = {"n": 0}
    real_update = state_store.update_heartbeat

    async def flaky(node_id):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("boom")
        await real_update(node_id)

    monkeypatch.setattr(state_store, "update_heartbeat", flaky)

    async with HeartbeatTask(state_store, nid, interval=0.1):
        await asyncio.sleep(0.35)
    # 期间至少有过一次成功 + 一次失败，未抛出
    assert call_count["n"] >= 2
