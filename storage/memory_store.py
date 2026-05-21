"""记忆库（spec v4 §3.2）。

- 每个 user_id 一个独立 Chroma collection（``mem_<user_id>``）。
- ``user_id`` 强校验正则 ``^[a-zA-Z0-9_-]{1,32}$``。
- Embedding 锁 ``BAAI/bge-small-zh-v1.5`` / 512 维，归一化后做余弦检索。
- ``metadata`` 默认补 ``status`` / ``created_at`` / ``expires_at`` /
  ``last_accessed_at`` / ``access_count``（后三个为远期淘汰钩子预留）。
- 接口签名按 spec §3.2 写死，全 async；内部用 ``asyncio.to_thread`` 包同步 chroma。
"""

from __future__ import annotations

import asyncio
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

_USER_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,32}$")
_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
_EMBEDDING_DIM = 512

_ef_lock = threading.Lock()
_embedding_function = None


def _get_embedding_function():
    """模块级单例：bge-small-zh-v1.5 加载一次。"""
    global _embedding_function
    if _embedding_function is None:
        with _ef_lock:
            if _embedding_function is None:
                from chromadb.utils.embedding_functions import (
                    SentenceTransformerEmbeddingFunction,
                )

                _embedding_function = SentenceTransformerEmbeddingFunction(
                    model_name=_EMBEDDING_MODEL,
                    normalize_embeddings=True,
                )
    return _embedding_function


def _validate_user_id(user_id: str) -> None:
    if not isinstance(user_id, str) or not _USER_ID_RE.match(user_id):
        raise ValueError(
            f"invalid user_id={user_id!r}; must match {_USER_ID_RE.pattern}"
        )


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class MemoryStore:
    """spec §3.2 接口签名锁定，下游代码不可再变。"""

    def __init__(self, persist_dir: str | Path) -> None:
        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False, allow_reset=False),
        )
        # 并发安全：缓存 collection；chromadb 0.4.15 的 get_or_create 不是线程安全的
        self._coll_cache: dict[str, object] = {}
        self._coll_lock = threading.Lock()

    async def add(
        self,
        user_id: str,
        doc: str,
        metadata: dict[str, Any],
    ) -> str:
        return await asyncio.to_thread(self._add_sync, user_id, doc, metadata)

    async def search(
        self,
        query: str,
        user_id: str,
        task_id: str,
        k: int = 5,
        cross_task: bool = False,
        status: str = "active",
    ) -> list[dict]:
        return await asyncio.to_thread(
            self._search_sync, query, user_id, task_id, k, cross_task, status
        )

    async def get_by_ids(
        self,
        user_id: str,
        mem_ids: list[str],
    ) -> list[dict]:
        return await asyncio.to_thread(self._get_by_ids_sync, user_id, mem_ids)

    async def update_status(
        self,
        user_id: str,
        mem_id: str,
        status: str,
    ) -> None:
        await asyncio.to_thread(self._update_status_sync, user_id, mem_id, status)

    async def delete(self, user_id: str, mem_ids: list[str]) -> int:
        """硬删一组记忆。recovery 类 1 / 类 3 用。返回删除条数。"""
        return await asyncio.to_thread(self._delete_sync, user_id, mem_ids)

    async def list_pending_for_node(
        self, user_id: str, node_id: str
    ) -> list[dict]:
        """recovery 用：列某节点产出且仍 ``status=pending`` 的记忆。"""
        return await asyncio.to_thread(
            self._list_pending_for_node_sync, user_id, node_id
        )

    async def get_status(self, user_id: str, mem_id: str) -> str | None:
        """recovery 类 2 用：取单条记忆当前 status；记忆不存在返回 None。"""
        return await asyncio.to_thread(self._get_status_sync, user_id, mem_id)

    def _collection(self, user_id: str):
        _validate_user_id(user_id)
        if user_id in self._coll_cache:
            return self._coll_cache[user_id]
        with self._coll_lock:
            if user_id in self._coll_cache:
                return self._coll_cache[user_id]
            coll = self._client.get_or_create_collection(
                name=f"mem_{user_id}",
                embedding_function=_get_embedding_function(),
                metadata={"hnsw:space": "cosine"},
            )
            self._coll_cache[user_id] = coll
            return coll

    def _add_sync(self, user_id: str, doc: str, metadata: dict[str, Any]) -> str:
        if not isinstance(doc, str) or not doc.strip():
            raise ValueError("doc must be a non-empty string")
        if not isinstance(metadata, dict):
            raise TypeError("metadata must be a dict")

        mem_id = f"mem_{uuid.uuid4()}"
        full_metadata: dict[str, Any] = {
            "status": "active",
            "created_at": _utcnow_iso(),
            "expires_at": "",
            "last_accessed_at": "",
            "access_count": 0,
            **metadata,
        }
        # Chroma 不接受 None 值，统一转空串
        full_metadata = {k: ("" if v is None else v) for k, v in full_metadata.items()}

        coll = self._collection(user_id)
        coll.add(ids=[mem_id], documents=[doc], metadatas=[full_metadata])
        return mem_id

    def _search_sync(
        self,
        query: str,
        user_id: str,
        task_id: str,
        k: int,
        cross_task: bool,
        status: str,
    ) -> list[dict]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        coll = self._collection(user_id)
        where: dict[str, Any] = {"status": status}
        if not cross_task:
            where = {"$and": [{"status": status}, {"task_id": task_id}]}

        res = coll.query(
            query_texts=[query],
            n_results=max(1, k),
            where=where,
        )
        out: list[dict] = []
        if not res["ids"] or not res["ids"][0]:
            return out
        for mem_id, doc, meta, dist in zip(
            res["ids"][0],
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
        ):
            out.append(
                {
                    "id": mem_id,
                    "document": doc,
                    "metadata": dict(meta),
                    "distance": float(dist),
                }
            )
        return out

    def _get_by_ids_sync(self, user_id: str, mem_ids: list[str]) -> list[dict]:
        if not mem_ids:
            return []
        coll = self._collection(user_id)
        res = coll.get(ids=list(mem_ids))
        out: list[dict] = []
        # chroma get() 不保证返回顺序与请求一致，手动按请求顺序重排
        index = {mid: i for i, mid in enumerate(res["ids"])}
        for mid in mem_ids:
            if mid not in index:
                continue
            i = index[mid]
            out.append(
                {
                    "id": res["ids"][i],
                    "document": res["documents"][i],
                    "metadata": dict(res["metadatas"][i]),
                }
            )
        return out

    def _update_status_sync(self, user_id: str, mem_id: str, status: str) -> None:
        if status not in {"pending", "active", "superseded", "archived"}:
            raise ValueError(f"invalid status: {status}")
        coll = self._collection(user_id)
        coll.update(ids=[mem_id], metadatas=[{"status": status}])

    def _delete_sync(self, user_id: str, mem_ids: list[str]) -> int:
        if not mem_ids:
            return 0
        coll = self._collection(user_id)
        coll.delete(ids=list(mem_ids))
        return len(mem_ids)

    def _list_pending_for_node_sync(
        self, user_id: str, node_id: str
    ) -> list[dict]:
        coll = self._collection(user_id)
        res = coll.get(
            where={
                "$and": [
                    {"produced_by_node": node_id},
                    {"status": "pending"},
                ]
            }
        )
        out: list[dict] = []
        for mid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"]):
            out.append({"id": mid, "document": doc, "metadata": dict(meta)})
        return out

    def _get_status_sync(self, user_id: str, mem_id: str) -> str | None:
        coll = self._collection(user_id)
        res = coll.get(ids=[mem_id])
        if not res["ids"]:
            return None
        return res["metadatas"][0].get("status")
