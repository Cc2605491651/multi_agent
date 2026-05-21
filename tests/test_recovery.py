"""recovery 三类扫描单测 + 幂等性 + 故障注入（阶段 2 任务 2.6 / 2.7）。"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from orchestrator.recovery import Recovery
from storage.memory_store import MemoryStore
from storage.state_store import StateStore


@pytest.fixture
def env(tmp_path: Path):
    state = StateStore(tmp_path / "state.db")
    mem = MemoryStore(tmp_path / "chroma")
    rec = Recovery(state, mem, stale_seconds=60)
    return state, mem, rec


async def _add_pending(mem: MemoryStore, *, user_id, task_id, node_id, doc) -> str:
    return await mem.add(
        user_id,
        doc,
        {
            "task_id": task_id,
            "produced_by_node": node_id,
            "produced_by_agent": "a1",
            "memory_level": "node_output",
            "status": "pending",
        },
    )


def _force_old_heartbeat(state_store: StateStore, node_id: str) -> None:
    with sqlite3.connect(state_store._db_path) as conn:
        conn.execute(
            "UPDATE dag_nodes SET heartbeat_at = '2020-01-01T00:00:00.000+00:00' WHERE id = ?",
            (node_id,),
        )


# ---------- 类 1 ----------


async def test_class1_resets_stale_running_and_deletes_pending(env) -> None:
    state, mem, rec = env
    tid = await state.create_task(user_id="alice", title="t", dag_id="d")
    nid = await state.create_dag_node(task_id=tid, node_name="n")
    await state.claim_node_running(nid, "w1")
    pending_mid = await _add_pending(
        mem, user_id="alice", task_id=tid, node_id=nid, doc="半成品"
    )
    _force_old_heartbeat(state, nid)

    report = await rec.sweep_all()
    assert report.class_1_reset_running == 1
    assert report.class_1_deleted_pending == 1

    n = await state.get_dag_node(nid)
    assert n.status == "pending"
    assert n.worker_id is None
    assert n.retry_count == 1
    # pending 记忆已删
    assert await mem.get_status("alice", pending_mid) is None


async def test_class1_only_targets_stale(env) -> None:
    state, mem, rec = env
    tid = await state.create_task(user_id="u", title="t", dag_id="d")
    fresh = await state.create_dag_node(task_id=tid, node_name="fresh")
    stale = await state.create_dag_node(task_id=tid, node_name="stale")
    await state.claim_node_running(fresh, "w1")
    await state.claim_node_running(stale, "w2")
    _force_old_heartbeat(state, stale)

    await rec.sweep_all()

    assert (await state.get_dag_node(fresh)).status == "running"
    assert (await state.get_dag_node(stale)).status == "pending"


# ---------- 类 2 ----------


async def test_class2_activates_done_pending_memory(env) -> None:
    """模拟 §6.2 第 3 步 (b) Chroma update 失败：state 是 done，但 mem 还是 pending。"""
    state, mem, rec = env
    tid = await state.create_task(user_id="bob", title="t", dag_id="d")
    nid = await state.create_dag_node(task_id=tid, node_name="n")
    await state.claim_node_running(nid, "w1")
    mid = await _add_pending(mem, user_id="bob", task_id=tid, node_id=nid, doc="结论")
    # 状态库事务成功，但 chroma update 没跑
    assert await state.commit_node_done(nid, mid) is True
    assert await mem.get_status("bob", mid) == "pending"

    report = await rec.sweep_all()
    assert report.class_2_activated == 1
    assert await mem.get_status("bob", mid) == "active"

    # 默认搜得到了
    hits = await mem.search("结论", "bob", tid, k=3)
    assert any(h["id"] == mid for h in hits)


async def test_class2_skips_already_active(env) -> None:
    state, mem, rec = env
    tid = await state.create_task(user_id="bob", title="t", dag_id="d")
    nid = await state.create_dag_node(task_id=tid, node_name="n")
    await state.claim_node_running(nid, "w1")
    mid = await _add_pending(mem, user_id="bob", task_id=tid, node_id=nid, doc="x")
    await mem.update_status("bob", mid, "active")
    await state.commit_node_done(nid, mid)

    report = await rec.sweep_all()
    assert report.class_2_activated == 0


# ---------- 类 3 ----------


async def test_class3_deletes_pending_for_terminal_nodes(env) -> None:
    """fail_fast 取消信号下：节点已被标 skipped/failed，但写过 pending 记忆。"""
    state, mem, rec = env
    tid = await state.create_task(user_id="carol", title="t", dag_id="d")
    sk = await state.create_dag_node(task_id=tid, node_name="sk")
    fa = await state.create_dag_node(task_id=tid, node_name="fa")
    await state.claim_node_running(sk, "w1")
    await state.claim_node_running(fa, "w2")
    p_sk = await _add_pending(
        mem, user_id="carol", task_id=tid, node_id=sk, doc="sk 半成品"
    )
    p_fa = await _add_pending(
        mem, user_id="carol", task_id=tid, node_id=fa, doc="fa 半成品"
    )
    await state.mark_node_terminal(sk, "skipped")
    await state.mark_node_terminal(fa, "failed")

    report = await rec.sweep_all()
    assert report.class_3_deleted_pending == 2
    assert await mem.get_status("carol", p_sk) is None
    assert await mem.get_status("carol", p_fa) is None


# ---------- 幂等性 ----------


async def test_sweep_idempotent(env) -> None:
    """spec §6.3 末尾强调：重复跑不能出错。"""
    state, mem, rec = env
    tid = await state.create_task(user_id="u", title="t", dag_id="d")
    nid = await state.create_dag_node(task_id=tid, node_name="n")
    await state.claim_node_running(nid, "w1")
    await _add_pending(mem, user_id="u", task_id=tid, node_id=nid, doc="half")
    _force_old_heartbeat(state, nid)

    r1 = await rec.sweep_all()
    r2 = await rec.sweep_all()
    r3 = await rec.sweep_all()

    assert r1.total() > 0
    # 第一次扫完，节点已退回 pending → 后续扫描不再触发任何修复
    assert r2.total() == 0
    assert r3.total() == 0


# ---------- 故障注入：writeback 中崩 → recovery 修复 → 节点重跑 ----------


async def test_kill_after_pending_then_recovery_resets(env) -> None:
    """模拟 spec 阶段 2 验收 demo 的关键路径：
    writeback 写完 pending 记忆 + 状态库事务尚未提交时被 kill。"""
    state, mem, rec = env
    tid = await state.create_task(user_id="dave", title="t", dag_id="d")
    nid = await state.create_dag_node(task_id=tid, node_name="n")
    await state.claim_node_running(nid, "w1")
    # 模拟 worker 写了 pending 记忆，紧接着崩了
    p_mid = await _add_pending(mem, user_id="dave", task_id=tid, node_id=nid, doc="half")
    # 状态库 commit 没发生 —— 节点仍是 running
    assert (await state.get_dag_node(nid)).status == "running"
    _force_old_heartbeat(state, nid)

    report = await rec.sweep_all()
    assert report.class_1_reset_running == 1
    assert report.class_1_deleted_pending == 1
    # 节点退回 pending 可重跑
    n = await state.get_dag_node(nid)
    assert n.status == "pending"
    assert n.retry_count == 1
    # 脏数据已清
    assert await mem.get_status("dave", p_mid) is None
