"""命名空间隔离回归测（spec v4 §3.2，阶段 1 任务 1.10）。

阶段 1 就要落地、后期不能迁的两条硬规则：
- 跨 user 不可见（per-user collection）
- 跨 task 检索默认关闭（``cross_task=False`` 强制 ``where.task_id``）
- ``user_id`` 不合规直接拒绝
"""

from __future__ import annotations

from pathlib import Path

import pytest

from storage.memory_store import MemoryStore


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "chroma")


async def test_default_user_id_pattern_accepted(store: MemoryStore) -> None:
    mid = await store.add("default_user", "首个用户的记忆", {"task_id": "t"})
    assert mid.startswith("mem_")


@pytest.mark.parametrize(
    "user_id",
    ["", "a b", "a/b", "a.b", "a:b", "中文用户", "-" * 33, "a@b"],
)
async def test_namespace_invalid_user_id_rejected(
    store: MemoryStore, user_id: str
) -> None:
    with pytest.raises(ValueError):
        await store.add(user_id, "doc", {"task_id": "t"})
    with pytest.raises(ValueError):
        await store.search("q", user_id, "t")
    with pytest.raises(ValueError):
        await store.get_by_ids(user_id, ["mem_xxx"])
    with pytest.raises(ValueError):
        await store.update_status(user_id, "mem_xxx", "active")


async def test_cross_user_invisible(store: MemoryStore) -> None:
    await store.add("alice", "alice 的秘密", {"task_id": "t1"})
    await store.add("bob", "bob 的秘密", {"task_id": "t1"})

    a_hits = await store.search("秘密", "alice", "t1", k=5)
    b_hits = await store.search("秘密", "bob", "t1", k=5)

    assert all("bob" not in h["document"] for h in a_hits)
    assert all("alice" not in h["document"] for h in b_hits)


async def test_same_user_cross_task_default_off(store: MemoryStore) -> None:
    await store.add("alice", "task1 的结论", {"task_id": "task_1"})
    await store.add("alice", "task2 的结论", {"task_id": "task_2"})

    hits = await store.search("结论", "alice", "task_1", k=5)
    assert hits
    for h in hits:
        assert h["metadata"]["task_id"] == "task_1"


async def test_same_user_cross_task_opt_in(store: MemoryStore) -> None:
    await store.add("alice", "task1 的结论", {"task_id": "task_1"})
    await store.add("alice", "task2 的结论", {"task_id": "task_2"})

    hits = await store.search(
        "结论", "alice", "task_1", k=5, cross_task=True
    )
    tids = {h["metadata"]["task_id"] for h in hits}
    assert tids == {"task_1", "task_2"}
