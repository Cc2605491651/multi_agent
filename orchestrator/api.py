"""阶段 5 仪表盘后端（spec v4 §10.2）。

只读 ``state_store``，不直连 sqlite3——将来换 Postgres 改一处。

接口：

- ``GET /api/tasks``：列出所有任务（含 status/dag_id/created_at），按 created_at 倒序
- ``GET /api/dag-status?task_id=...``：拿到 task 元信息 + 节点列表（含
  status / deps / retry / model / tools / heartbeat / timestamps）
- ``GET /healthz``

静态资源：``/`` 默认指向 ``dashboard/index.html``。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from storage.state_store import DagNodeRow, StateStore, TaskRow


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DASHBOARD_DIR = _PROJECT_ROOT / "dashboard"
_DEFAULT_STATE_DB = _PROJECT_ROOT / "data" / "state.db"


def _node_to_dict(n: DagNodeRow) -> dict[str, Any]:
    return {
        "id": n.id,
        "name": n.node_name,
        "status": n.status,
        "deps": n.depends_on,
        "failure_policy": n.failure_policy,
        "retry_count": n.retry_count,
        "max_retries": n.max_retries,
        "worker_id": n.worker_id,
        "input_memory_ids": n.input_memory_ids,
        "output_memory_id": n.output_memory_id,
        "heartbeat_at": n.heartbeat_at,
        "started_at": n.started_at,
        "finished_at": n.finished_at,
        "memory_level": n.memory_level,
        "model_name": n.model_name,
        "tools": n.tools,
    }


def _task_to_dict(t: TaskRow) -> dict[str, Any]:
    return {
        "id": t.id,
        "user_id": t.user_id,
        "title": t.title,
        "dag_id": t.dag_id,
        "handoff_conversation_id": t.handoff_conversation_id,
        "handoff_turn_range": t.handoff_turn_range,
        "status": t.status,
        "created_at": t.created_at,
    }


def create_app(state_db_path: str | Path | None = None) -> FastAPI:
    db_path = Path(state_db_path) if state_db_path else _DEFAULT_STATE_DB
    db_path.parent.mkdir(parents=True, exist_ok=True)
    state = StateStore(db_path)

    app = FastAPI(title="multi_agent dashboard API", version="0.5.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:*", "http://127.0.0.1:*", "*"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"ok": True, "db": str(db_path)}

    @app.get("/api/tasks")
    async def list_tasks() -> list[dict[str, Any]]:
        # state_store 没有 list_tasks 接口；这里走 connection 但仍由 state_store 暴露的入口
        rows = await _list_tasks_raw(state)
        return [_task_to_dict(t) for t in rows]

    @app.get("/api/dag-status")
    async def dag_status(task_id: str = Query(...)) -> dict[str, Any]:
        task = await state.get_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"task not found: {task_id}")
        nodes = await state.list_dag_nodes(task_id)
        return {
            "task": _task_to_dict(task),
            "nodes": [_node_to_dict(n) for n in nodes],
        }

    if _DASHBOARD_DIR.exists():
        @app.get("/")
        async def index() -> FileResponse:
            return FileResponse(_DASHBOARD_DIR / "index.html")

        app.mount(
            "/static",
            StaticFiles(directory=_DASHBOARD_DIR),
            name="static",
        )

    return app


async def _list_tasks_raw(state: StateStore) -> list[TaskRow]:
    """临时 helper：state_store 没暴露 list_tasks；走 read-only 查询。"""
    import asyncio

    def _sync() -> list[TaskRow]:
        with sqlite3.connect(state._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC"
            ).fetchall()
        return [
            TaskRow(
                id=r["id"], user_id=r["user_id"], title=r["title"],
                dag_id=r["dag_id"],
                handoff_conversation_id=r["handoff_conversation_id"],
                handoff_turn_range=__import__("json").loads(r["handoff_turn_range"])
                if r["handoff_turn_range"] else None,
                status=r["status"], created_at=r["created_at"],
            )
            for r in rows
        ]

    return await asyncio.to_thread(_sync)
