"""仪表盘 API 单测（阶段 5 任务 5.1）。"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from orchestrator.api import create_app
from storage.state_store import StateStore


@pytest.fixture
def app_env(tmp_path: Path):
    db = tmp_path / "state.db"
    app = create_app(db)
    client = TestClient(app)
    state = StateStore(db)
    return client, state


async def _seed_task(state: StateStore, *, title="t", with_extra=True) -> tuple[str, dict]:
    tid = await state.create_task(user_id="alice", title=title, dag_id="d")
    nids = {}
    nids["r"] = await state.create_dag_node(
        task_id=tid, node_name="research", depends_on=[],
        failure_policy="fail_retry", max_retries=3,
        model_name="claude-sonnet-4-6" if with_extra else None,
        tools=["web_search", "read_file"] if with_extra else [],
    )
    nids["w"] = await state.create_dag_node(
        task_id=tid, node_name="writing", depends_on=[nids["r"]],
        memory_level="task_conclusion" if with_extra else "node_output",
        model_name="claude-opus-4-7" if with_extra else None,
    )
    return tid, nids


def test_healthz(app_env) -> None:
    client, _ = app_env
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["ok"] is True


async def test_list_tasks(app_env) -> None:
    client, state = app_env
    tid1, _ = await _seed_task(state, title="任务 A")
    tid2, _ = await _seed_task(state, title="任务 B")

    r = client.get("/api/tasks")
    assert r.status_code == 200
    rows = r.json()
    ids = [t["id"] for t in rows]
    assert tid1 in ids and tid2 in ids
    # 元信息齐
    for t in rows:
        assert "title" in t and "dag_id" in t and "status" in t


async def test_dag_status_returns_nodes_with_full_fields(app_env) -> None:
    client, state = app_env
    tid, nids = await _seed_task(state)

    r = client.get(f"/api/dag-status?task_id={tid}")
    assert r.status_code == 200
    body = r.json()
    assert body["task"]["id"] == tid
    assert body["task"]["title"] == "t"

    by_name = {n["name"]: n for n in body["nodes"]}
    assert by_name["research"]["status"] == "pending"
    assert by_name["research"]["model_name"] == "claude-sonnet-4-6"
    assert "web_search" in by_name["research"]["tools"]
    assert by_name["writing"]["deps"] == [nids["r"]]
    assert by_name["writing"]["memory_level"] == "task_conclusion"
    assert by_name["writing"]["model_name"] == "claude-opus-4-7"


def test_dag_status_unknown_task_returns_404(app_env) -> None:
    client, _ = app_env
    r = client.get("/api/dag-status?task_id=ghost")
    assert r.status_code == 404


async def test_dag_status_reflects_state_changes(app_env) -> None:
    """spec §10.1：仪表盘是旁观者，状态变了应该看得见。"""
    client, state = app_env
    tid, nids = await _seed_task(state)

    await state.claim_node_running(nids["r"], "w1")

    r = client.get(f"/api/dag-status?task_id={tid}")
    by_name = {n["name"]: n for n in r.json()["nodes"]}
    assert by_name["research"]["status"] == "running"
    assert by_name["research"]["worker_id"] == "w1"
    assert by_name["research"]["heartbeat_at"] is not None
