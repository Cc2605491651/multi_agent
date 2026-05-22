"""MCP client 单测（阶段 ABC.C.2）。

不依赖 npm / 真实 MCP server：用 ``tests/_mcp_fake_server.py``。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from worker.harness import MCPServerSpec
from worker.mcp_client import MCPClient, MCPError, MCPTool, connect_all, close_all


FAKE_SERVER = Path(__file__).resolve().parent / "_mcp_fake_server.py"


def _fake_spec(name: str = "fake") -> MCPServerSpec:
    return MCPServerSpec(name=name, command=sys.executable, args=[str(FAKE_SERVER)])


async def test_connect_initialize_and_discover_tools() -> None:
    client = await MCPClient.connect(_fake_spec(), timeout=5.0)
    try:
        tools = client.discovered_tools()
        names = {t["name"] for t in tools}
        assert names == {"echo", "add"}
    finally:
        await client.close()


async def test_call_tool_echo() -> None:
    client = await MCPClient.connect(_fake_spec(), timeout=5.0)
    try:
        out = await client.call_tool("echo", {"text": "hello mcp"})
        assert out == "hello mcp"
    finally:
        await client.close()


async def test_call_tool_add() -> None:
    client = await MCPClient.connect(_fake_spec(), timeout=5.0)
    try:
        out = await client.call_tool("add", {"a": 3, "b": 4})
        assert out == "7"
    finally:
        await client.close()


async def test_unknown_tool_returns_error_content() -> None:
    client = await MCPClient.connect(_fake_spec(), timeout=5.0)
    try:
        out = await client.call_tool("bogus", {})
        assert "unknown tool" in out
    finally:
        await client.close()


async def test_mcp_tool_wrap() -> None:
    client = await MCPClient.connect(_fake_spec("fs"), timeout=5.0)
    try:
        tdef = next(t for t in client.discovered_tools() if t["name"] == "echo")
        mcp_tool = MCPTool(client, tdef)
        assert mcp_tool.name == "mcp_fs_echo"
        # sandbox/handle 在 MCP 路径下不用
        result = await mcp_tool.run({"text": "wrapped"}, sandbox=None, handle=None)
        assert not result.is_error
        assert result.content == "wrapped"
    finally:
        await client.close()


async def test_connect_all_multi_server_tolerates_failure() -> None:
    good = _fake_spec("good")
    bad = MCPServerSpec(name="missing", command="/no/such/binary")
    clients, tools = await connect_all([good, bad], timeout=5.0)
    try:
        assert len(clients) == 1
        assert clients[0].server_name == "good"
        assert any(t.name == "mcp_good_echo" for t in tools)
    finally:
        await close_all(clients)


async def test_connect_missing_command_raises() -> None:
    with pytest.raises(MCPError, match="command not found"):
        await MCPClient.connect(
            MCPServerSpec(name="x", command="/definitely/not/exists"),
            timeout=2.0,
        )


async def test_empty_command_raises() -> None:
    with pytest.raises(MCPError, match="no command"):
        await MCPClient.connect(MCPServerSpec(name="x", command=""), timeout=2.0)


async def test_close_is_idempotent() -> None:
    client = await MCPClient.connect(_fake_spec(), timeout=5.0)
    await client.close()
    await client.close()  # 第二次不应该报错
