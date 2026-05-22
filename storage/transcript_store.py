"""对话原文库（spec v4 §3.1，v6 升级）。

v5 之前每个节点只写 ``turn_index=1`` 一条；v6 起多轮 tool_use 的每一步都落
``transcript_turns``，新增 ``turn_kind`` + ``turn_meta`` 字段：

- ``turn_kind`` ∈ {single, user, assistant, tool_call, tool_result, final}
- ``turn_meta`` JSON：``tool_call`` 含 ``{tool_name, tool_use_id, args}``；
  ``tool_result`` 含 ``{tool_use_id, is_error}``；其它可空

接口：

- ``add_turn(...)``：兼容老调用（``turn_kind`` 默认 ``"single"``）
- ``add_tool_loop_turns(loop_result, conv_id, agent_id, packed_text)``：把
  ``ToolLoopResult`` 一次拆成多个 turn 原子写入
- ``get_turns_by_range``：取范围内所有 turn（含中间 tool_call/tool_result）
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS transcript_turns (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    turn_index      INTEGER NOT NULL,
    agent_id        TEXT,
    user_input      TEXT NOT NULL,
    agent_output    TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    turn_kind       TEXT NOT NULL DEFAULT 'single',
    turn_meta       TEXT,
    UNIQUE(conversation_id, turn_index)
);
CREATE INDEX IF NOT EXISTS idx_transcript_conv
    ON transcript_turns(conversation_id, turn_index);
"""

VALID_TURN_KIND = {
    "single",        # 单轮节点（无 tool_use）：user_input + agent_output 都用
    "user",          # 多轮节点的初始用户输入
    "assistant",     # 中间一段 assistant 文本（无 tool）
    "tool_call",     # 工具调用：turn_meta 含 {tool_name, tool_use_id, args}
    "tool_result",   # 工具结果：turn_meta 含 {tool_use_id, is_error}
    "final",         # 最终 assistant 文本（取代 ToolLoopResult.final_text）
}


@dataclass(frozen=True)
class TranscriptTurn:
    id: str
    conversation_id: str
    turn_index: int
    agent_id: str | None
    user_input: str
    agent_output: str
    created_at: str
    turn_kind: str = "single"
    turn_meta: dict[str, Any] | None = None


class TranscriptStore:
    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._init_schema()

    async def add_turn(
        self,
        conversation_id: str,
        turn_index: int,
        user_input: str,
        agent_output: str,
        agent_id: str | None = None,
        *,
        turn_kind: str = "single",
        turn_meta: dict[str, Any] | None = None,
    ) -> str:
        if turn_kind not in VALID_TURN_KIND:
            raise ValueError(f"invalid turn_kind={turn_kind!r}")
        return await asyncio.to_thread(
            self._add_turn_sync,
            conversation_id, turn_index, user_input, agent_output, agent_id,
            turn_kind, turn_meta,
        )

    async def add_tool_loop_turns(
        self,
        *,
        conversation_id: str,
        agent_id: str | None,
        initial_user_input: str,
        loop_result,  # worker.tool_loop.ToolLoopResult
    ) -> list[str]:
        """把 ``ToolLoopResult`` 一次拆成多个 turn 写入；原子事务。

        生成的 turn 序列（按 turn_index 1..N）：

        - 1：``user`` 含 initial_user_input
        - 2..K：每个 tool_call 后跟一个 tool_result（成对出现）
        - K+1：``final`` 含 ``loop_result.final_text``

        ``turn_meta`` 含足够信息让仪表盘还原 agent 思考过程。
        """
        return await asyncio.to_thread(
            self._add_tool_loop_turns_sync,
            conversation_id, agent_id, initial_user_input, loop_result,
        )

    async def get_turns_by_range(
        self,
        conversation_id: str,
        start: int,
        end: int,
    ) -> list[TranscriptTurn]:
        return await asyncio.to_thread(
            self._get_turns_by_range_sync, conversation_id, start, end
        )

    async def count_turns(self, conversation_id: str) -> int:
        """该 conv 当前有多少 turn（供节点接力 turn_range 默认值用）。"""
        return await asyncio.to_thread(self._count_turns_sync, conversation_id)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            # 兼容性迁移：旧 DB 缺新列
            cols = {row[1] for row in conn.execute(
                "PRAGMA table_info(transcript_turns)"
            )}
            if "turn_kind" not in cols:
                conn.execute(
                    "ALTER TABLE transcript_turns ADD COLUMN turn_kind "
                    "TEXT NOT NULL DEFAULT 'single'"
                )
            if "turn_meta" not in cols:
                conn.execute(
                    "ALTER TABLE transcript_turns ADD COLUMN turn_meta TEXT"
                )

    @staticmethod
    def _utcnow() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    def _add_turn_sync(
        self,
        conversation_id: str,
        turn_index: int,
        user_input: str,
        agent_output: str,
        agent_id: str | None,
        turn_kind: str,
        turn_meta: dict | None,
    ) -> str:
        turn_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO transcript_turns
                    (id, conversation_id, turn_index, agent_id,
                     user_input, agent_output, created_at,
                     turn_kind, turn_meta)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn_id, conversation_id, turn_index, agent_id,
                    user_input, agent_output, self._utcnow(),
                    turn_kind,
                    json.dumps(turn_meta, ensure_ascii=False) if turn_meta else None,
                ),
            )
        return turn_id

    def _add_tool_loop_turns_sync(
        self,
        conversation_id: str,
        agent_id: str | None,
        initial_user_input: str,
        loop_result,
    ) -> list[str]:
        now = self._utcnow()
        rows: list[tuple] = []

        def _row(idx: int, kind: str, user_text: str, agent_text: str,
                 meta: dict | None) -> tuple:
            return (
                str(uuid.uuid4()), conversation_id, idx, agent_id,
                user_text, agent_text, now,
                kind,
                json.dumps(meta, ensure_ascii=False) if meta else None,
            )

        idx = 1
        rows.append(_row(idx, "user", initial_user_input, "", None))
        idx += 1
        # 每个 tool_call 后紧跟 tool_result，对齐索引
        for i, tc in enumerate(loop_result.tool_calls):
            rows.append(_row(
                idx, "tool_call", "", "",
                {
                    "tool_use_id": f"tc_{i}",
                    "tool_name": tc.tool_name,
                    "args": tc.args,
                },
            ))
            idx += 1
            rows.append(_row(
                idx, "tool_result", "", tc.result,
                {"tool_use_id": f"tc_{i}", "is_error": tc.is_error},
            ))
            idx += 1
        rows.append(_row(
            idx, "final", "", loop_result.final_text or "",
            {"turns": loop_result.turns, "stop_reason": loop_result.stop_reason},
        ))

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO transcript_turns
                    (id, conversation_id, turn_index, agent_id,
                     user_input, agent_output, created_at,
                     turn_kind, turn_meta)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return [r[0] for r in rows]

    def _get_turns_by_range_sync(
        self,
        conversation_id: str,
        start: int,
        end: int,
    ) -> list[TranscriptTurn]:
        if start > end:
            raise ValueError(f"start({start}) > end({end})")
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, conversation_id, turn_index, agent_id,
                       user_input, agent_output, created_at,
                       turn_kind, turn_meta
                FROM transcript_turns
                WHERE conversation_id = ?
                  AND turn_index BETWEEN ? AND ?
                ORDER BY turn_index ASC
                """,
                (conversation_id, start, end),
            ).fetchall()
        out: list[TranscriptTurn] = []
        for row in rows:
            d = dict(row)
            d["turn_meta"] = json.loads(d["turn_meta"]) if d["turn_meta"] else None
            out.append(TranscriptTurn(**d))
        return out

    def _count_turns_sync(self, conversation_id: str) -> int:
        with self._connect() as conn:
            r = conn.execute(
                "SELECT COUNT(*) FROM transcript_turns WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        return r[0] if r else 0
