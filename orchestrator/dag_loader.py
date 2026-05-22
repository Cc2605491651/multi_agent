"""DAG JSON 加载 + schema 校验 + 实例化（spec v4 §5.4 / 阶段 4a 任务 4a.1）。

DAG 文件结构（参见 ``dags/research_report.json``）::

    {
      "dag_id": "research_report",
      "description": "...",
      "nodes": [
        {"id": "n1", "name": "research_a", "deps": [],
         "failure_policy": "fail_retry", "max_retries": 3},
        ...
      ]
    }

``id`` 是 DAG 文件内的「逻辑 id」（节点引用 deps 用）；实例化时为每个逻辑 id
生成一个 DB 内的 ``node_id``，并把 ``deps`` 翻译成 ``depends_on``。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from storage.state_store import (
    StateStore,
    VALID_FAILURE_POLICY,
    VALID_MEMORY_LEVEL,
)


@dataclass(frozen=True)
class DagNodeDef:
    id: str
    name: str
    deps: list[str]
    failure_policy: str = "fail_retry"
    max_retries: int = 2
    memory_level: str = "node_output"


@dataclass(frozen=True)
class DagDef:
    dag_id: str
    description: str
    nodes: list[DagNodeDef] = field(default_factory=list)


def load_dag(path: str | Path) -> DagDef:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    return _parse_dag(raw)


def parse_dag(raw: dict) -> DagDef:
    """从已解析的 dict 构造 DagDef（测试用）。"""
    return _parse_dag(raw)


def _parse_dag(raw: dict) -> DagDef:
    dag_id = raw.get("dag_id")
    if not isinstance(dag_id, str) or not dag_id:
        raise ValueError("dag_id is required and must be a non-empty string")

    description = raw.get("description", "")
    if not isinstance(description, str):
        raise ValueError("description must be a string")

    nodes_raw = raw.get("nodes")
    if not isinstance(nodes_raw, list) or not nodes_raw:
        raise ValueError("nodes must be a non-empty list")

    seen_ids: set[str] = set()
    parsed_nodes: list[DagNodeDef] = []
    for i, n in enumerate(nodes_raw):
        if not isinstance(n, dict):
            raise ValueError(f"node[{i}] must be an object")
        nid = n.get("id")
        if not isinstance(nid, str) or not nid:
            raise ValueError(f"node[{i}].id missing or not a string")
        if nid in seen_ids:
            raise ValueError(f"duplicate node id: {nid}")
        seen_ids.add(nid)

        name = n.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"node {nid}: name missing or not a string")

        deps = n.get("deps", [])
        if not isinstance(deps, list) or not all(isinstance(d, str) for d in deps):
            raise ValueError(f"node {nid}: deps must be list of strings")

        policy = n.get("failure_policy", "fail_retry")
        if policy not in VALID_FAILURE_POLICY:
            raise ValueError(
                f"node {nid}: invalid failure_policy={policy!r}, "
                f"must be one of {sorted(VALID_FAILURE_POLICY)}"
            )

        max_retries = n.get("max_retries", 2)
        if not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError(
                f"node {nid}: max_retries must be a non-negative int"
            )

        memory_level = n.get("memory_level", "node_output")
        if memory_level not in VALID_MEMORY_LEVEL:
            raise ValueError(
                f"node {nid}: invalid memory_level={memory_level!r}, "
                f"must be one of {sorted(VALID_MEMORY_LEVEL)}"
            )

        parsed_nodes.append(
            DagNodeDef(
                id=nid,
                name=name,
                deps=list(deps),
                failure_policy=policy,
                max_retries=max_retries,
                memory_level=memory_level,
            )
        )

    # 检查 deps 都存在
    for n in parsed_nodes:
        for dep in n.deps:
            if dep not in seen_ids:
                raise ValueError(
                    f"node {n.id}: dep {dep!r} not defined in this DAG"
                )

    # 环检测（拓扑排序 Kahn）
    _check_acyclic(parsed_nodes)

    return DagDef(dag_id=dag_id, description=description, nodes=parsed_nodes)


def _check_acyclic(nodes: list[DagNodeDef]) -> None:
    in_degree: dict[str, int] = {n.id: 0 for n in nodes}
    edges: dict[str, list[str]] = {n.id: [] for n in nodes}
    for n in nodes:
        for dep in n.deps:
            edges[dep].append(n.id)
            in_degree[n.id] += 1

    ready = [nid for nid, d in in_degree.items() if d == 0]
    visited = 0
    while ready:
        nid = ready.pop()
        visited += 1
        for downstream in edges[nid]:
            in_degree[downstream] -= 1
            if in_degree[downstream] == 0:
                ready.append(downstream)
    if visited != len(nodes):
        raise ValueError("DAG contains a cycle")


async def instantiate_dag(
    state_store: StateStore,
    dag: DagDef,
    *,
    user_id: str,
    title: str,
    handoff_conversation_id: str | None = None,
    handoff_turn_range: list[int] | None = None,
) -> tuple[str, dict[str, str]]:
    """把 DAG 定义实例化为 task + dag_nodes。

    返回 ``(task_id, logical_id_to_node_id)``。``depends_on`` 已用实例化后的
    node_id 填好——下游调度直接读 ``state_store`` 即可。
    """
    task_id = await state_store.create_task(
        user_id=user_id,
        title=title,
        dag_id=dag.dag_id,
        handoff_conversation_id=handoff_conversation_id,
        handoff_turn_range=handoff_turn_range,
    )

    # 按 deps 顺序实例化（先实例化无依赖的，确保 depends_on 能填到真实 node_id）
    sorted_nodes = _topological_sort(dag.nodes)
    mapping: dict[str, str] = {}
    for n in sorted_nodes:
        depends_on = [mapping[d] for d in n.deps]
        nid = await state_store.create_dag_node(
            task_id=task_id,
            node_name=n.name,
            depends_on=depends_on,
            failure_policy=n.failure_policy,
            max_retries=n.max_retries,
            memory_level=n.memory_level,
        )
        mapping[n.id] = nid
    return task_id, mapping


def _topological_sort(nodes: list[DagNodeDef]) -> list[DagNodeDef]:
    """Kahn 拓扑排序，返回按依赖顺序排列的节点。环已在 _check_acyclic 排除。"""
    by_id = {n.id: n for n in nodes}
    in_degree: dict[str, int] = {n.id: 0 for n in nodes}
    edges: dict[str, list[str]] = {n.id: [] for n in nodes}
    for n in nodes:
        for dep in n.deps:
            edges[dep].append(n.id)
            in_degree[n.id] += 1

    ready = [nid for nid, d in in_degree.items() if d == 0]
    out: list[DagNodeDef] = []
    while ready:
        ready.sort()  # 稳定输出顺序
        nid = ready.pop(0)
        out.append(by_id[nid])
        for downstream in edges[nid]:
            in_degree[downstream] -= 1
            if in_degree[downstream] == 0:
                ready.append(downstream)
    return out
