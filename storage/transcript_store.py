"""对话原文库（spec v4 §3.1）。

阶段 1 唯一公开接口为 async；内部用 ``asyncio.to_thread`` 包同步 ``sqlite3``
（决策 D-1.2 的样板，其他 store 模块照抄）。
"""

from __future__ import annotations

import asyncio
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS transcript_turns (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    turn_index      INTEGER NOT NULL,
    agent_id        TEXT,
    user_input      TEXT NOT NULL,
    agent_output    TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    UNIQUE(conversation_id, turn_index)
);
CREATE INDEX IF NOT EXISTS idx_transcript_conv
    ON transcript_turns(conversation_id, turn_index);
"""


@dataclass(frozen=True)
class TranscriptTurn:
    id: str
    conversation_id: str
    turn_index: int
    agent_id: str | None
    user_input: str
    agent_output: str
    created_at: str


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
    ) -> str:
        return await asyncio.to_thread(
            self._add_turn_sync,
            conversation_id,
            turn_index,
            user_input,
            agent_output,
            agent_id,
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

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _add_turn_sync(
        self,
        conversation_id: str,
        turn_index: int,
        user_input: str,
        agent_output: str,
        agent_id: str | None,
    ) -> str:
        turn_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO transcript_turns
                    (id, conversation_id, turn_index, agent_id,
                     user_input, agent_output, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn_id,
                    conversation_id,
                    turn_index,
                    agent_id,
                    user_input,
                    agent_output,
                    created_at,
                ),
            )
        return turn_id

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
                       user_input, agent_output, created_at
                FROM transcript_turns
                WHERE conversation_id = ?
                  AND turn_index BETWEEN ? AND ?
                ORDER BY turn_index ASC
                """,
                (conversation_id, start, end),
            ).fetchall()
        return [TranscriptTurn(**dict(row)) for row in rows]
