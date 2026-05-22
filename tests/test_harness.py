"""AgentHarness + ToolSpec/SkillSpec/MCPServerSpec 单测（阶段 ABC.A.2）。"""

from __future__ import annotations

import json

import pytest

from worker.harness import (
    AgentHarness,
    MCPServerSpec,
    SkillSpec,
    ToolSpec,
)


# ---- ToolSpec ----


def test_tool_from_string_shorthand() -> None:
    t = ToolSpec.from_obj("read_file")
    assert t.name == "read_file"
    assert t.description == ""
    assert t.params == {}


def test_tool_from_full_dict() -> None:
    t = ToolSpec.from_obj({
        "name": "web_search", "description": "做联网搜索",
        "params": {"max_results": 5},
    })
    assert t.name == "web_search"
    assert t.description == "做联网搜索"
    assert t.params == {"max_results": 5}


def test_tool_invalid_type_raises() -> None:
    with pytest.raises(TypeError):
        ToolSpec.from_obj(123)


# ---- SkillSpec ----


def test_skill_string_and_dict() -> None:
    assert SkillSpec.from_obj("frontend-design").name == "frontend-design"
    s = SkillSpec.from_obj({
        "name": "code-review", "description": "代码审计",
        "instructions_path": "skills/code-review.md",
        "invoke_keywords": ["review", "审"],
    })
    assert s.instructions_path == "skills/code-review.md"
    assert "review" in s.invoke_keywords


# ---- MCPServerSpec ----


def test_mcp_dict_full() -> None:
    m = MCPServerSpec.from_obj({
        "name": "fs", "command": "npx",
        "args": ["@mcp/filesystem", "/workdir"],
        "env": {"NODE_ENV": "production"},
    })
    assert m.name == "fs"
    assert m.command == "npx"
    assert m.args == ["@mcp/filesystem", "/workdir"]
    assert m.env == {"NODE_ENV": "production"}


# ---- AgentHarness ----


def test_harness_empty() -> None:
    h = AgentHarness()
    assert h.is_empty() is True
    assert h.tools == []


def test_harness_round_trip_full() -> None:
    raw = {
        "model": "claude-opus-4-7",
        "provider": "anthropic",
        "system_prompt": "你是高级研究员",
        "tools": ["read_file", {"name": "web_search", "params": {"k": 5}}],
        "skills": ["code-review"],
        "mcp_servers": [{"name": "fs", "command": "npx", "args": ["@mcp/filesystem"]}],
    }
    h = AgentHarness.from_dict(raw)
    assert h.model == "claude-opus-4-7"
    assert h.provider == "anthropic"
    assert h.system_prompt == "你是高级研究员"
    assert [t.name for t in h.tools] == ["read_file", "web_search"]
    assert h.tools[1].params == {"k": 5}
    assert h.skills[0].name == "code-review"
    assert h.mcp_servers[0].args == ["@mcp/filesystem"]

    # round-trip
    d = h.to_dict()
    h2 = AgentHarness.from_dict(d)
    assert h2 == h


def test_harness_from_json_string() -> None:
    s = json.dumps({"model": "gpt-4o-mini", "tools": ["read_file"]})
    h = AgentHarness.from_dict(s)
    assert h.model == "gpt-4o-mini"
    assert h.tools[0].name == "read_file"


def test_harness_from_legacy_fields() -> None:
    h = AgentHarness.from_legacy(
        model_name="claude-sonnet-4-6",
        tools=["read_file", "web_search"],
    )
    assert h.model == "claude-sonnet-4-6"
    assert [t.name for t in h.tools] == ["read_file", "web_search"]
    assert h.skills == [] and h.mcp_servers == []


def test_harness_from_none_returns_empty() -> None:
    assert AgentHarness.from_dict(None).is_empty()


def test_harness_invalid_input_raises() -> None:
    with pytest.raises(TypeError):
        AgentHarness.from_dict(42)


def test_harness_immutable() -> None:
    h = AgentHarness(model="x")
    with pytest.raises(Exception):
        h.model = "y"  # type: ignore[misc]
