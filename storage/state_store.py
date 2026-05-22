"""状态库（spec v4 §3.3 / §6.2 / §6.3）。

- 字段「一次到位」：含 failure_policy / retry_count / max_retries / heartbeat_at /
  input_memory_ids（spec §13 强调）。
- ``commit_node_done`` 是 §6.2 第 3 步的唯一提交点——一个事务里同时
  写 ``status=done`` + ``output_memory_id`` + ``finished_at``。
- 三类 recovery 查询都在这里暴露（``find_stale_running`` / ``find_done_with_memory`` /
  ``find_terminal_with_memory``）。
- 接口全 async；内部 ``asyncio.to_thread`` 包同步 ``sqlite3``。
- WAL 模式开启（spec §13），为阶段 4 并发铺路。
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id                       TEXT PRIMARY KEY,
    user_id                  TEXT NOT NULL,
    title                    TEXT NOT NULL,
    dag_id                   TEXT NOT NULL,
    handoff_conversation_id  TEXT,
    handoff_turn_range       TEXT,
    status                   TEXT NOT NULL,
    created_at               TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dag_nodes (
    id                TEXT PRIMARY KEY,
    task_id           TEXT NOT NULL,
    node_name         TEXT NOT NULL,
    depends_on        TEXT,
    status            TEXT NOT NULL,
    failure_policy    TEXT NOT NULL DEFAULT 'fail_retry',
    retry_count       INTEGER NOT NULL DEFAULT 0,
    max_retries       INTEGER NOT NULL DEFAULT 2,
    worker_id         TEXT,
    input_memory_ids  TEXT,
    output_memory_id  TEXT,
    heartbeat_at      TEXT,
    started_at        TEXT,
    finished_at       TEXT,
    memory_level      TEXT NOT NULL DEFAULT 'node_output',
    model_name        TEXT,
    tools             TEXT,
    FOREIGN KEY(task_id) REFERENCES tasks(id)
);
CREATE INDEX IF NOT EXISTS idx_dag_nodes_task   ON dag_nodes(task_id);
CREATE INDEX IF NOT EXISTS idx_dag_nodes_status ON dag_nodes(status);
"""

VALID_TASK_STATUS = {"pending", "running", "done", "failed"}
VALID_NODE_STATUS = {"pending", "running", "done", "failed", "skipped"}
VALID_FAILURE_POLICY = {"fail_retry", "fail_skip", "fail_fast"}
VALID_MEMORY_LEVEL = {"node_output", "task_conclusion"}


@dataclass(frozen=True)
class TaskRow:
    id: str
    user_id: str
    title: str
    dag_id: str
    handoff_conversation_id: str | None
    handoff_turn_range: list[int] | None
    status: str
    created_at: str


@dataclass(frozen=True)
class DagNodeRow:
    id: str
    task_id: str
    node_name: str
    depends_on: list[str]
    status: str
    failure_policy: str
    retry_count: int
    max_retries: int
    worker_id: str | None
    input_memory_ids: list[str]
    output_memory_id: str | None
    heartbeat_at: str | None
    started_at: str | None
    finished_at: str | None
    memory_level: str = "node_output"
    model_name: str | None = None
    tools: list[str] = field(default_factory=list)


def _utcnow() -> str:
    # 用 milliseconds 精度，避免心跳/状态变更在同一秒内被合并
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _parse_json_list(s: str | None) -> list[Any]:
    if not s:
        return []
    try:
        v = json.loads(s)
    except json.JSONDecodeError:
        return []
    return v if isinstance(v, list) else []


def _row_to_task(row: sqlite3.Row) -> TaskRow:
    turn_range = _parse_json_list(row["handoff_turn_range"]) or None
    return TaskRow(
        id=row["id"],
        user_id=row["user_id"],
        title=row["title"],
        dag_id=row["dag_id"],
        handoff_conversation_id=row["handoff_conversation_id"],
        handoff_turn_range=turn_range,
        status=row["status"],
        created_at=row["created_at"],
    )


def _row_to_node(row: sqlite3.Row) -> DagNodeRow:
    cols = row.keys()
    return DagNodeRow(
        id=row["id"],
        task_id=row["task_id"],
        node_name=row["node_name"],
        depends_on=_parse_json_list(row["depends_on"]),
        status=row["status"],
        failure_policy=row["failure_policy"],
        retry_count=row["retry_count"],
        max_retries=row["max_retries"],
        worker_id=row["worker_id"],
        input_memory_ids=_parse_json_list(row["input_memory_ids"]),
        output_memory_id=row["output_memory_id"],
        heartbeat_at=row["heartbeat_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        memory_level=row["memory_level"] if "memory_level" in cols else "node_output",
        model_name=row["model_name"] if "model_name" in cols else None,
        tools=_parse_json_list(row["tools"]) if "tools" in cols else [],
    )


class StateStore:
    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._init_schema()

    # ---------- async API ----------

    async def create_task(
        self,
        *,
        user_id: str,
        title: str,
        dag_id: str,
        handoff_conversation_id: str | None = None,
        handoff_turn_range: list[int] | None = None,
        task_id: str | None = None,
    ) -> str:
        return await asyncio.to_thread(
            self._create_task_sync,
            user_id,
            title,
            dag_id,
            handoff_conversation_id,
            handoff_turn_range,
            task_id,
        )

    async def get_task(self, task_id: str) -> TaskRow | None:
        return await asyncio.to_thread(self._get_task_sync, task_id)

    async def update_task_status(self, task_id: str, status: str) -> None:
        if status not in VALID_TASK_STATUS:
            raise ValueError(f"invalid task status: {status}")
        await asyncio.to_thread(self._update_task_status_sync, task_id, status)

    async def create_dag_node(
        self,
        *,
        task_id: str,
        node_name: str,
        depends_on: list[str] | None = None,
        failure_policy: str = "fail_retry",
        max_retries: int = 2,
        memory_level: str = "node_output",
        model_name: str | None = None,
        tools: list[str] | None = None,
        node_id: str | None = None,
    ) -> str:
        if failure_policy not in VALID_FAILURE_POLICY:
            raise ValueError(f"invalid failure_policy: {failure_policy}")
        if memory_level not in VALID_MEMORY_LEVEL:
            raise ValueError(f"invalid memory_level: {memory_level}")
        return await asyncio.to_thread(
            self._create_dag_node_sync,
            task_id,
            node_name,
            depends_on or [],
            failure_policy,
            max_retries,
            memory_level,
            model_name,
            list(tools) if tools else [],
            node_id,
        )

    async def get_dag_node(self, node_id: str) -> DagNodeRow | None:
        return await asyncio.to_thread(self._get_node_sync, node_id)

    async def list_dag_nodes(self, task_id: str) -> list[DagNodeRow]:
        return await asyncio.to_thread(self._list_nodes_sync, task_id)

    async def set_node_input_memory_ids(
        self, node_id: str, mem_ids: list[str]
    ) -> None:
        await asyncio.to_thread(self._set_input_memory_ids_sync, node_id, mem_ids)

    async def claim_node_running(self, node_id: str, worker_id: str) -> bool:
        """状态机：pending → running。只有 status=pending 才更新成功。"""
        return await asyncio.to_thread(self._claim_running_sync, node_id, worker_id)

    async def update_heartbeat(self, node_id: str) -> None:
        await asyncio.to_thread(self._update_heartbeat_sync, node_id)

    async def commit_node_done(
        self, node_id: str, output_memory_id: str | None
    ) -> bool:
        """spec §6.2 第 3 步：唯一提交点。一个事务里写 done + finished_at + output_memory_id。

        只有 status=running 时才提交成功；返回 ``False`` 表示节点状态已变
        （如已被 recovery 清理），调用方应放弃本次 writeback。
        """
        return await asyncio.to_thread(
            self._commit_done_sync, node_id, output_memory_id
        )

    async def mark_node_terminal(
        self,
        node_id: str,
        status: str,
        *,
        increment_retry: bool = False,
    ) -> None:
        """终态写入（failed / skipped）。"""
        if status not in {"failed", "skipped"}:
            raise ValueError(f"not a terminal status: {status}")
        await asyncio.to_thread(
            self._mark_terminal_sync, node_id, status, increment_retry
        )

    async def reset_node_to_pending(
        self, node_id: str, *, increment_retry: bool
    ) -> None:
        """recovery 类 1：把 running 节点退回 pending，可选地 retry_count+1。"""
        await asyncio.to_thread(
            self._reset_to_pending_sync, node_id, increment_retry
        )

    # ---- recovery 查询 ----

    async def find_stale_running(self, threshold_seconds: int) -> list[DagNodeRow]:
        """类 1：status=running 且心跳超过阈值的节点。"""
        return await asyncio.to_thread(self._find_stale_running_sync, threshold_seconds)

    async def find_done_with_memory(self) -> list[DagNodeRow]:
        """类 2：status=done 且 output_memory_id 非空的节点。"""
        return await asyncio.to_thread(self._find_done_with_memory_sync)

    async def find_terminal_nodes(self) -> list[DagNodeRow]:
        """类 3：status∈{failed, skipped}（用于清 pending 记忆残留）。"""
        return await asyncio.to_thread(self._find_terminal_sync)

    # ---------- sync impl ----------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            cols = {row[1] for row in conn.execute("PRAGMA table_info(dag_nodes)")}
            # 兼容性迁移：旧 DB 缺列
            if "memory_level" not in cols:
                conn.execute(
                    "ALTER TABLE dag_nodes ADD COLUMN memory_level "
                    "TEXT NOT NULL DEFAULT 'node_output'"
                )
            if "model_name" not in cols:
                conn.execute("ALTER TABLE dag_nodes ADD COLUMN model_name TEXT")
            if "tools" not in cols:
                conn.execute("ALTER TABLE dag_nodes ADD COLUMN tools TEXT")

    def _create_task_sync(
        self,
        user_id: str,
        title: str,
        dag_id: str,
        handoff_conversation_id: str | None,
        handoff_turn_range: list[int] | None,
        task_id: str | None,
    ) -> str:
        tid = task_id or f"task_{uuid.uuid4().hex[:12]}"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tasks
                    (id, user_id, title, dag_id,
                     handoff_conversation_id, handoff_turn_range,
                     status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (
                    tid,
                    user_id,
                    title,
                    dag_id,
                    handoff_conversation_id,
                    json.dumps(handoff_turn_range) if handoff_turn_range else None,
                    _utcnow(),
                ),
            )
        return tid

    def _get_task_sync(self, task_id: str) -> TaskRow | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
        return _row_to_task(row) if row else None

    def _update_task_status_sync(self, task_id: str, status: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE tasks SET status = ? WHERE id = ?", (status, task_id)
            )

    def _create_dag_node_sync(
        self,
        task_id: str,
        node_name: str,
        depends_on: list[str],
        failure_policy: str,
        max_retries: int,
        memory_level: str,
        model_name: str | None,
        tools: list[str],
        node_id: str | None,
    ) -> str:
        nid = node_id or f"node_{uuid.uuid4().hex[:12]}"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO dag_nodes
                    (id, task_id, node_name, depends_on, status,
                     failure_policy, retry_count, max_retries,
                     input_memory_ids, memory_level,
                     model_name, tools)
                VALUES (?, ?, ?, ?, 'pending', ?, 0, ?, '[]', ?, ?, ?)
                """,
                (
                    nid,
                    task_id,
                    node_name,
                    json.dumps(depends_on),
                    failure_policy,
                    max_retries,
                    memory_level,
                    model_name,
                    json.dumps(tools) if tools else None,
                ),
            )
        return nid

    def _get_node_sync(self, node_id: str) -> DagNodeRow | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM dag_nodes WHERE id = ?", (node_id,)
            ).fetchone()
        return _row_to_node(row) if row else None

    def _list_nodes_sync(self, task_id: str) -> list[DagNodeRow]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM dag_nodes WHERE task_id = ? ORDER BY started_at, id",
                (task_id,),
            ).fetchall()
        return [_row_to_node(r) for r in rows]

    def _set_input_memory_ids_sync(self, node_id: str, mem_ids: list[str]) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE dag_nodes SET input_memory_ids = ? WHERE id = ?",
                (json.dumps(mem_ids), node_id),
            )

    def _claim_running_sync(self, node_id: str, worker_id: str) -> bool:
        now = _utcnow()
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE dag_nodes
                SET status = 'running',
                    worker_id = ?,
                    started_at = COALESCE(started_at, ?),
                    heartbeat_at = ?
                WHERE id = ? AND status = 'pending'
                """,
                (worker_id, now, now, node_id),
            )
            return cur.rowcount == 1

    def _update_heartbeat_sync(self, node_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE dag_nodes SET heartbeat_at = ? WHERE id = ? AND status = 'running'",
                (_utcnow(), node_id),
            )

    def _commit_done_sync(self, node_id: str, output_memory_id: str | None) -> bool:
        now = _utcnow()
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE dag_nodes
                SET status = 'done',
                    output_memory_id = ?,
                    finished_at = ?,
                    heartbeat_at = ?
                WHERE id = ? AND status = 'running'
                """,
                (output_memory_id, now, now, node_id),
            )
            return cur.rowcount == 1

    def _mark_terminal_sync(
        self, node_id: str, status: str, increment_retry: bool
    ) -> None:
        now = _utcnow()
        with self._connect() as conn:
            if increment_retry:
                conn.execute(
                    """
                    UPDATE dag_nodes
                    SET status = ?, finished_at = ?, retry_count = retry_count + 1
                    WHERE id = ?
                    """,
                    (status, now, node_id),
                )
            else:
                conn.execute(
                    "UPDATE dag_nodes SET status = ?, finished_at = ? WHERE id = ?",
                    (status, now, node_id),
                )

    def _reset_to_pending_sync(self, node_id: str, increment_retry: bool) -> None:
        with self._connect() as conn:
            if increment_retry:
                conn.execute(
                    """
                    UPDATE dag_nodes
                    SET status = 'pending',
                        worker_id = NULL,
                        heartbeat_at = NULL,
                        retry_count = retry_count + 1
                    WHERE id = ?
                    """,
                    (node_id,),
                )
            else:
                conn.execute(
                    """
                    UPDATE dag_nodes
                    SET status = 'pending',
                        worker_id = NULL,
                        heartbeat_at = NULL
                    WHERE id = ?
                    """,
                    (node_id,),
                )

    def _find_stale_running_sync(self, threshold_seconds: int) -> list[DagNodeRow]:
        cutoff_dt = datetime.now(timezone.utc).timestamp() - threshold_seconds
        cutoff_iso = datetime.fromtimestamp(cutoff_dt, tz=timezone.utc).isoformat(
            timespec="seconds"
        )
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM dag_nodes
                WHERE status = 'running'
                  AND (heartbeat_at IS NULL OR heartbeat_at < ?)
                """,
                (cutoff_iso,),
            ).fetchall()
        return [_row_to_node(r) for r in rows]

    def _find_done_with_memory_sync(self) -> list[DagNodeRow]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM dag_nodes
                WHERE status = 'done' AND output_memory_id IS NOT NULL
                """
            ).fetchall()
        return [_row_to_node(r) for r in rows]

    def _find_terminal_sync(self) -> list[DagNodeRow]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM dag_nodes
                WHERE status IN ('failed', 'skipped')
                """
            ).fetchall()
        return [_row_to_node(r) for r in rows]
