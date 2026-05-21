"""memory_store 单测（阶段 1 任务 1.4 / 1.5）。

首次运行会下载 bge-small-zh-v1.5（~100MB），后续走 HF 本地缓存。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from storage.memory_store import MemoryStore


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> MemoryStore:
    persist_dir = tmp_path_factory.mktemp("chroma")
    return MemoryStore(persist_dir)


# ---- user_id 校验 ----


@pytest.mark.parametrize(
    "bad",
    [
        "",
        " ",
        "has space",
        "中文",
        "user.id",
        "user@host",
        "a" * 33,
    ],
)
async def test_invalid_user_id_rejected(store: MemoryStore, bad: str) -> None:
    with pytest.raises(ValueError):
        await store.add(bad, "doc", {"task_id": "t1"})


@pytest.mark.parametrize(
    "good",
    ["u", "user_1", "USER-2", "a" * 32, "default_user", "0"],
)
async def test_valid_user_id_accepted(store: MemoryStore, good: str) -> None:
    mem_id = await store.add(good, "测试记忆", {"task_id": "t1"})
    assert mem_id.startswith("mem_")


# ---- add / search / get_by_ids ----


async def test_add_then_search_recall(store: MemoryStore) -> None:
    user = "alice"
    await store.add(
        user,
        "用户有一只橘猫，名字叫米饭，喜欢半夜抓门",
        {"task_id": "task_pets"},
    )
    await store.add(
        user,
        "用户喜欢喝美式咖啡，加一点燕麦奶",
        {"task_id": "task_pets"},
    )

    hits = await store.search("用户的宠物", user, "task_pets", k=2)
    assert hits, "should recall at least one memory"
    top_doc = hits[0]["document"]
    assert "橘猫" in top_doc, f"top hit should be the pet memory, got: {top_doc}"


async def test_get_by_ids_returns_in_request_order(store: MemoryStore) -> None:
    user = "bob"
    a = await store.add(user, "结论 A", {"task_id": "task_x"})
    b = await store.add(user, "结论 B", {"task_id": "task_x"})
    c = await store.add(user, "结论 C", {"task_id": "task_x"})

    out = await store.get_by_ids(user, [c, a, b])
    assert [m["id"] for m in out] == [c, a, b]
    assert [m["document"] for m in out] == ["结论 C", "结论 A", "结论 B"]


async def test_get_by_ids_skips_unknown(store: MemoryStore) -> None:
    user = "bob"
    a = await store.add(user, "存在", {"task_id": "task_y"})
    out = await store.get_by_ids(user, [a, "mem_ghost"])
    assert len(out) == 1
    assert out[0]["id"] == a


# ---- 跨 task 默认关闭 ----


async def test_cross_task_default_off(store: MemoryStore) -> None:
    user = "carol"
    await store.add(user, "task1 的结论", {"task_id": "task_1"})
    await store.add(user, "task2 的结论", {"task_id": "task_2"})

    hits = await store.search("结论", user, "task_1", k=5)
    assert hits
    for h in hits:
        assert h["metadata"]["task_id"] == "task_1"


async def test_cross_task_opt_in(store: MemoryStore) -> None:
    user = "carol"
    hits = await store.search("结论", user, "task_1", k=5, cross_task=True)
    task_ids = {h["metadata"]["task_id"] for h in hits}
    assert "task_2" in task_ids


# ---- status 过滤 / update_status ----


async def test_default_status_is_active_and_filter(store: MemoryStore) -> None:
    user = "dave"
    pending = await store.add(
        user, "悬而未决的结论", {"task_id": "task_s", "status": "pending"}
    )
    active = await store.add(user, "活跃的结论", {"task_id": "task_s"})

    hits = await store.search("结论", user, "task_s", k=5)
    ids = {h["id"] for h in hits}
    assert active in ids
    assert pending not in ids

    pending_hits = await store.search(
        "结论", user, "task_s", k=5, status="pending"
    )
    assert pending in {h["id"] for h in pending_hits}


async def test_update_status_pending_to_active(store: MemoryStore) -> None:
    user = "dave"
    mid = await store.add(
        user, "升级前 pending", {"task_id": "task_u", "status": "pending"}
    )
    assert (await store.get_by_ids(user, [mid]))[0]["metadata"]["status"] == "pending"

    await store.update_status(user, mid, "active")
    assert (await store.get_by_ids(user, [mid]))[0]["metadata"]["status"] == "active"


async def test_update_status_rejects_unknown(store: MemoryStore) -> None:
    with pytest.raises(ValueError):
        await store.update_status("dave", "mem_x", "garbage")


# ---- metadata 预留字段 ----


async def test_metadata_has_expiry_hooks(store: MemoryStore) -> None:
    user = "eve"
    mid = await store.add(user, "记忆 A", {"task_id": "task_m"})
    meta = (await store.get_by_ids(user, [mid]))[0]["metadata"]
    assert "expires_at" in meta
    assert "last_accessed_at" in meta
    assert "access_count" in meta
    assert meta["access_count"] == 0


# ---- per-user collection 隔离 ----


async def test_per_user_collection_isolation(store: MemoryStore) -> None:
    await store.add("alpha", "alpha 私密笔记", {"task_id": "t1"})
    await store.add("beta", "beta 私密笔记", {"task_id": "t1"})

    hits_alpha = await store.search("私密笔记", "alpha", "t1", k=5)
    assert all("beta" not in h["document"] for h in hits_alpha)

    hits_beta = await store.search("私密笔记", "beta", "t1", k=5)
    assert all("alpha" not in h["document"] for h in hits_beta)


# ---- 持久化 ----


async def test_store_persists_across_instances(tmp_path: Path) -> None:
    persist_dir = tmp_path / "chroma_persist"
    s1 = MemoryStore(persist_dir)
    mid = await s1.add("frank", "持久化记忆", {"task_id": "t1"})

    s2 = MemoryStore(persist_dir)
    out = await s2.get_by_ids("frank", [mid])
    assert len(out) == 1
    assert out[0]["document"] == "持久化记忆"
