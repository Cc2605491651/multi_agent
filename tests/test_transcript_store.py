"""transcript_store 单测（阶段 1 任务 1.3）。"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from storage.transcript_store import TranscriptStore


@pytest.fixture
def store(tmp_path: Path) -> TranscriptStore:
    return TranscriptStore(tmp_path / "transcript.db")


async def test_add_and_get_single_turn(store: TranscriptStore) -> None:
    turn_id = await store.add_turn(
        conversation_id="conv1",
        turn_index=1,
        user_input="hi",
        agent_output="hello",
        agent_id="agent_a",
    )
    assert turn_id

    turns = await store.get_turns_by_range("conv1", 1, 1)
    assert len(turns) == 1
    t = turns[0]
    assert t.id == turn_id
    assert t.conversation_id == "conv1"
    assert t.turn_index == 1
    assert t.user_input == "hi"
    assert t.agent_output == "hello"
    assert t.agent_id == "agent_a"
    assert t.created_at.endswith("+00:00")


async def test_range_ordered_by_turn_index(store: TranscriptStore) -> None:
    for i in [3, 1, 2]:
        await store.add_turn("conv1", i, f"u{i}", f"a{i}")
    turns = await store.get_turns_by_range("conv1", 1, 3)
    assert [t.turn_index for t in turns] == [1, 2, 3]


async def test_range_filters_by_conversation(store: TranscriptStore) -> None:
    await store.add_turn("convA", 1, "uA", "aA")
    await store.add_turn("convB", 1, "uB", "aB")
    turns = await store.get_turns_by_range("convA", 1, 10)
    assert len(turns) == 1
    assert turns[0].user_input == "uA"


async def test_range_partial_bounds(store: TranscriptStore) -> None:
    for i in range(1, 6):
        await store.add_turn("conv1", i, f"u{i}", f"a{i}")
    turns = await store.get_turns_by_range("conv1", 2, 4)
    assert [t.turn_index for t in turns] == [2, 3, 4]


async def test_empty_range_returns_empty_list(store: TranscriptStore) -> None:
    turns = await store.get_turns_by_range("ghost", 1, 100)
    assert turns == []


async def test_duplicate_conv_turn_raises(store: TranscriptStore) -> None:
    await store.add_turn("conv1", 1, "u1", "a1")
    with pytest.raises(sqlite3.IntegrityError):
        await store.add_turn("conv1", 1, "u1-dup", "a1-dup")


async def test_invalid_range_raises(store: TranscriptStore) -> None:
    with pytest.raises(ValueError):
        await store.get_turns_by_range("conv1", 5, 1)


async def test_agent_id_optional(store: TranscriptStore) -> None:
    await store.add_turn("conv1", 1, "u", "a")
    turns = await store.get_turns_by_range("conv1", 1, 1)
    assert turns[0].agent_id is None


async def test_store_persists_across_instances(tmp_path: Path) -> None:
    db_path = tmp_path / "persist.db"
    store_a = TranscriptStore(db_path)
    await store_a.add_turn("conv1", 1, "u", "a")

    store_b = TranscriptStore(db_path)
    turns = await store_b.get_turns_by_range("conv1", 1, 1)
    assert len(turns) == 1
    assert turns[0].user_input == "u"
