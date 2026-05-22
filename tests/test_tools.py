"""内置工具单测（阶段 ABC.B.1）。

所有工具用 LocalBackend 真跑（沙箱够轻）；web_search 用 mock sandbox 避免联网。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from worker.harness import ToolSpec
from worker.sandbox import LocalBackend
from worker.tools import (
    BUILTIN_TOOLS,
    ExecCommandTool,
    ReadFileTool,
    RunCodeTool,
    ToolRegistry,
    ToolResult,
    WebSearchTool,
    WriteFileTool,
)


@pytest.fixture
async def env(tmp_path: Path):
    sandbox = LocalBackend(root_dir=tmp_path / "sb")
    handle = await sandbox.create("test ctx")
    yield sandbox, handle
    await sandbox.destroy(handle)


# ---- ToolResult ----


def test_result_truncates_long_content() -> None:
    r = ToolResult("x" * 9000)
    assert "[truncated]" in r.content
    assert len(r.content) <= 8100


# ---- read/write/exec/run_code via LocalBackend ----


async def test_write_then_read(env) -> None:
    sandbox, handle = env
    w = WriteFileTool()
    r = ReadFileTool()
    out = await w.run({"path": "a.txt", "content": "hello"}, sandbox=sandbox, handle=handle)
    assert not out.is_error
    out = await r.run({"path": "a.txt"}, sandbox=sandbox, handle=handle)
    assert out.content == "hello"


async def test_exec_command(env) -> None:
    sandbox, handle = env
    out = await ExecCommandTool().run(
        {"cmd": "echo hi"}, sandbox=sandbox, handle=handle
    )
    assert "hi" in out.content


async def test_run_code(env) -> None:
    sandbox, handle = env
    out = await RunCodeTool().run(
        {"code": "print(2+3)"}, sandbox=sandbox, handle=handle
    )
    assert "5" in out.content


async def test_invalid_args(env) -> None:
    sandbox, handle = env
    out = await ReadFileTool().run({"path": ""}, sandbox=sandbox, handle=handle)
    assert out.is_error


async def test_read_nonexistent(env) -> None:
    sandbox, handle = env
    out = await ReadFileTool().run(
        {"path": "ghost.txt"}, sandbox=sandbox, handle=handle
    )
    assert out.is_error
    assert "failed" in out.content


# ---- ToolRegistry ----


def test_registry_from_specs_skips_unknown() -> None:
    reg = ToolRegistry.from_specs(
        [ToolSpec(name="read_file"), ToolSpec(name="bogus_unknown")]
    )
    assert "read_file" in reg.names()
    assert "bogus_unknown" not in reg.names()


def test_registry_overrides_description() -> None:
    reg = ToolRegistry.from_specs(
        [ToolSpec(name="read_file", description="自定义读文件描述")]
    )
    schema = reg.to_anthropic_schema()
    assert schema[0]["description"] == "自定义读文件描述"
    assert schema[0]["name"] == "read_file"


def test_registry_anthropic_vs_openai_schema() -> None:
    reg = ToolRegistry.from_specs(
        [ToolSpec(name="read_file"), ToolSpec(name="write_file")]
    )
    a = reg.to_anthropic_schema()
    o = reg.to_openai_schema()
    assert len(a) == 2 and len(o) == 2
    assert all("input_schema" in t for t in a)
    assert all(t["type"] == "function" for t in o)
    assert o[0]["function"]["name"] == "read_file"


async def test_registry_call_unknown_returns_error(env) -> None:
    sandbox, handle = env
    reg = ToolRegistry.from_specs([ToolSpec(name="read_file")])
    out = await reg.call("ghost", {}, sandbox=sandbox, handle=handle)
    assert out.is_error
    assert "未知工具" in out.content


async def test_registry_routes_to_correct_tool(env) -> None:
    sandbox, handle = env
    reg = ToolRegistry.from_specs(
        [ToolSpec(name="write_file"), ToolSpec(name="read_file")]
    )
    await reg.call(
        "write_file", {"path": "x.txt", "content": "abc"},
        sandbox=sandbox, handle=handle,
    )
    out = await reg.call("read_file", {"path": "x.txt"}, sandbox=sandbox, handle=handle)
    assert out.content == "abc"


# ---- WebSearchTool: 用 mock sandbox 避免联网 ----


class _FakeHandle:
    def __init__(self):
        self.sandbox_id = "fake"


class _FakeSandbox:
    """记录写入的脚本 + 命令；exec_command 返回 fake JSON。"""

    def __init__(self, exec_output: str):
        self._exec_output = exec_output
        self.written: dict[str, str] = {}
        self.commands: list[str] = []

    async def write_file(self, handle, path, content):
        self.written[path] = content

    async def exec_command(self, handle, cmd):
        self.commands.append(cmd)
        return self._exec_output


async def test_web_search_returns_parsed_results() -> None:
    fake_json = json.dumps({
        "results": [
            {"title": "Hello", "url": "https://example.com", "snippet": "world"},
        ]
    }, ensure_ascii=False)
    sb = _FakeSandbox(exec_output=fake_json)
    out = await WebSearchTool().run(
        {"query": "Hello world", "max_results": 1},
        sandbox=sb, handle=_FakeHandle(),
    )
    assert not out.is_error
    parsed = json.loads(out.content)
    assert parsed["results"][0]["title"] == "Hello"
    # 验证脚本被写入沙箱
    assert "_search.py" in sb.written


async def test_web_search_network_error_returns_error() -> None:
    fake_json = json.dumps({"error": "timeout"})
    sb = _FakeSandbox(exec_output=fake_json)
    out = await WebSearchTool().run(
        {"query": "x"}, sandbox=sb, handle=_FakeHandle()
    )
    assert out.is_error
    assert "timeout" in out.content


async def test_web_search_bad_json_returns_error() -> None:
    sb = _FakeSandbox(exec_output="not even json")
    out = await WebSearchTool().run(
        {"query": "x"}, sandbox=sb, handle=_FakeHandle()
    )
    assert out.is_error


# ---- 内置工具表完整性 ----


def test_all_5_builtin_tools_present() -> None:
    expected = {"read_file", "write_file", "exec_command", "run_code", "web_search"}
    assert set(BUILTIN_TOOLS) == expected
