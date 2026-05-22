"""dag_loader 单测（阶段 4a 任务 4a.1）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.dag_loader import (
    DagDef,
    DagNodeDef,
    instantiate_dag,
    load_dag,
    parse_dag,
)
from storage.state_store import StateStore


def _base() -> dict:
    return {
        "dag_id": "x",
        "nodes": [
            {"id": "n1", "name": "research", "deps": []},
            {"id": "n2", "name": "writing", "deps": ["n1"]},
        ],
    }


def test_parse_minimal_valid() -> None:
    d = parse_dag(_base())
    assert d.dag_id == "x"
    assert len(d.nodes) == 2
    assert d.nodes[0].failure_policy == "fail_retry"  # default
    assert d.nodes[0].max_retries == 2  # default


def test_parse_full_fields() -> None:
    d = parse_dag(
        {
            "dag_id": "y",
            "description": "示例 DAG",
            "nodes": [
                {
                    "id": "n1", "name": "a", "deps": [],
                    "failure_policy": "fail_fast", "max_retries": 0,
                },
            ],
        }
    )
    assert d.description == "示例 DAG"
    assert d.nodes[0].failure_policy == "fail_fast"
    assert d.nodes[0].max_retries == 0


@pytest.mark.parametrize(
    "patch,err_substr",
    [
        ({"dag_id": ""}, "dag_id"),
        ({"nodes": []}, "nodes"),
        ({"nodes": [{"id": "n1", "name": "a", "deps": ["ghost"]}]}, "not defined"),
        (
            {"nodes": [
                {"id": "a", "name": "a", "deps": ["b"]},
                {"id": "b", "name": "b", "deps": ["a"]},
            ]},
            "cycle",
        ),
        (
            {"nodes": [
                {"id": "n1", "name": "a", "deps": []},
                {"id": "n1", "name": "dup", "deps": []},
            ]},
            "duplicate",
        ),
        (
            {"nodes": [{"id": "n1", "name": "a", "deps": [], "failure_policy": "bogus"}]},
            "failure_policy",
        ),
        (
            {"nodes": [{"id": "n1", "name": "a", "deps": [], "max_retries": -1}]},
            "max_retries",
        ),
    ],
)
def test_parse_invalid(patch: dict, err_substr: str) -> None:
    raw = _base()
    raw.update(patch)
    with pytest.raises(ValueError, match=err_substr):
        parse_dag(raw)


def test_load_spec_5_4_example_dag() -> None:
    """阶段 4a 验收 spec §5.4 示例（已存在于 dags/research_report.json）。"""
    path = Path(__file__).resolve().parent.parent / "dags" / "research_report.json"
    d = load_dag(path)
    assert d.dag_id == "research_report"
    by_id = {n.id: n for n in d.nodes}
    assert by_id["n6"].deps == ["n3", "n4", "n5"]
    assert by_id["n6"].failure_policy == "fail_fast"
    assert by_id["n2"].failure_policy == "fail_skip"
    assert by_id["n1"].max_retries == 3


async def test_instantiate_dag_writes_task_and_nodes(tmp_path: Path) -> None:
    state = StateStore(tmp_path / "state.db")
    d = parse_dag(
        {
            "dag_id": "x",
            "nodes": [
                {"id": "n1", "name": "research_a", "deps": []},
                {"id": "n2", "name": "research_b", "deps": [], "failure_policy": "fail_skip"},
                {"id": "n3", "name": "summarize", "deps": ["n1", "n2"], "failure_policy": "fail_fast"},
            ],
        }
    )
    task_id, mapping = await instantiate_dag(
        state, d, user_id="alice", title="t"
    )
    assert mapping.keys() == {"n1", "n2", "n3"}

    nodes = await state.list_dag_nodes(task_id)
    by_name = {n.node_name: n for n in nodes}
    assert by_name["summarize"].depends_on == [mapping["n1"], mapping["n2"]]
    assert by_name["summarize"].failure_policy == "fail_fast"
    assert by_name["research_b"].failure_policy == "fail_skip"


async def test_instantiate_preserves_max_retries(tmp_path: Path) -> None:
    state = StateStore(tmp_path / "state.db")
    d = parse_dag(
        {
            "dag_id": "x",
            "nodes": [
                {"id": "n1", "name": "a", "deps": [], "max_retries": 5},
            ],
        }
    )
    _, mapping = await instantiate_dag(state, d, user_id="u", title="t")
    nid = mapping["n1"]
    n = await state.get_dag_node(nid)
    assert n.max_retries == 5


# ---- 阶段 5 新字段：model + tools ----


def test_parse_model_and_tools() -> None:
    d = parse_dag(
        {
            "dag_id": "x",
            "nodes": [
                {
                    "id": "n1", "name": "a", "deps": [],
                    "model": "claude-opus-4-7",
                    "tools": ["web_search", "read_file"],
                },
            ],
        }
    )
    assert d.nodes[0].model_name == "claude-opus-4-7"
    assert d.nodes[0].tools == ["web_search", "read_file"]


def test_parse_defaults_when_omitted() -> None:
    d = parse_dag({"dag_id": "x", "nodes": [{"id": "n1", "name": "a", "deps": []}]})
    assert d.nodes[0].model_name is None
    assert d.nodes[0].tools == []


@pytest.mark.parametrize(
    "patch,err",
    [
        ({"nodes": [{"id": "n1", "name": "a", "deps": [], "model": ""}]}, "model"),
        ({"nodes": [{"id": "n1", "name": "a", "deps": [], "tools": "not-a-list"}]}, "tools"),
        ({"nodes": [{"id": "n1", "name": "a", "deps": [], "tools": [123]}]}, "tools"),
        ({"nodes": [{"id": "n1", "name": "a", "deps": [], "tools": [""]}]}, "tools"),
    ],
)
def test_parse_invalid_model_or_tools(patch: dict, err: str) -> None:
    raw = _base()
    raw.update(patch)
    with pytest.raises(ValueError, match=err):
        parse_dag(raw)


async def test_instantiate_writes_model_and_tools(tmp_path: Path) -> None:
    state = StateStore(tmp_path / "state.db")
    d = parse_dag(
        {
            "dag_id": "x",
            "nodes": [
                {
                    "id": "n1", "name": "research", "deps": [],
                    "model": "claude-haiku-4-5",
                    "tools": ["web_search"],
                },
                {
                    "id": "n2", "name": "writing", "deps": ["n1"],
                    "model": "claude-sonnet-4-6",
                },
            ],
        }
    )
    _, mapping = await instantiate_dag(state, d, user_id="u", title="t")
    r = await state.get_dag_node(mapping["n1"])
    w = await state.get_dag_node(mapping["n2"])
    assert r.model_name == "claude-haiku-4-5"
    assert r.tools == ["web_search"]
    assert w.model_name == "claude-sonnet-4-6"
    assert w.tools == []


async def test_loaded_5_4_dag_has_per_node_models(tmp_path: Path) -> None:
    """阶段 5 起 dags/research_report.json 每节点都标注 model + tools。"""
    state = StateStore(tmp_path / "state.db")
    path = Path(__file__).resolve().parent.parent / "dags" / "research_report.json"
    d = load_dag(path)
    _, mapping = await instantiate_dag(state, d, user_id="u", title="t")
    summarize = await state.get_dag_node(mapping["n6"])
    assert summarize.model_name == "claude-opus-4-7"
    assert summarize.memory_level == "task_conclusion"
    assert "write_file" in summarize.tools


# ---- ABC.A 新增：harness 字段 ----


def test_parse_full_harness_field() -> None:
    d = parse_dag(
        {
            "dag_id": "x",
            "nodes": [
                {
                    "id": "n1", "name": "a", "deps": [],
                    "harness": {
                        "model": "deepseek-chat",
                        "provider": "deepseek",
                        "system_prompt": "你是专业研究员",
                        "tools": [
                            {"name": "web_search", "description": "联网"},
                            "read_file",
                        ],
                        "skills": [{"name": "fact-check"}],
                        "mcp_servers": [
                            {"name": "fs", "command": "npx", "args": ["@mcp/fs"]}
                        ],
                    },
                },
            ],
        }
    )
    h = d.nodes[0].harness
    assert h.model == "deepseek-chat"
    assert h.provider == "deepseek"
    assert h.system_prompt == "你是专业研究员"
    assert [t.name for t in h.tools] == ["web_search", "read_file"]
    assert h.skills[0].name == "fact-check"
    assert h.mcp_servers[0].command == "npx"
    # effective_model 落到节点字段
    assert d.nodes[0].model_name == "deepseek-chat"
    assert d.nodes[0].tools == ["web_search", "read_file"]


def test_legacy_flat_fields_still_work() -> None:
    """老式 model/tools 平铺写法 → 自动 fold 进 harness。"""
    d = parse_dag(
        {
            "dag_id": "x",
            "nodes": [
                {
                    "id": "n1", "name": "a", "deps": [],
                    "model": "gpt-4o-mini",
                    "tools": ["web_search"],
                },
            ],
        }
    )
    h = d.nodes[0].harness
    assert h.model == "gpt-4o-mini"
    assert [t.name for t in h.tools] == ["web_search"]


async def test_instantiate_writes_harness_json(tmp_path: Path) -> None:
    state = StateStore(tmp_path / "state.db")
    d = parse_dag(
        {
            "dag_id": "x",
            "nodes": [
                {
                    "id": "n1", "name": "a", "deps": [],
                    "harness": {
                        "model": "claude-opus-4-7",
                        "provider": "anthropic",
                        "tools": ["read_file"],
                    },
                },
            ],
        }
    )
    _, mapping = await instantiate_dag(state, d, user_id="u", title="t")
    n = await state.get_dag_node(mapping["n1"])
    assert n.harness_json is not None
    import json as _json

    parsed = _json.loads(n.harness_json)
    assert parsed["model"] == "claude-opus-4-7"
    assert parsed["provider"] == "anthropic"
    assert parsed["tools"] == [{"name": "read_file", "description": "", "params": {}}]


def test_invalid_harness_raises() -> None:
    with pytest.raises(ValueError, match="harness"):
        parse_dag(
            {
                "dag_id": "x",
                "nodes": [
                    {"id": "n1", "name": "a", "deps": [], "harness": 42},
                ],
            }
        )
