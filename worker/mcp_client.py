"""MCP server stdio 客户端 + tool 包装（阶段 ABC.C.2）。

实现 Anthropic Model Context Protocol（MCP）的最小 client：

- 启动 ``MCPServerSpec.command + args`` 作为子进程
- stdio JSON-RPC 2.0：``initialize`` → ``notifications/initialized`` → ``tools/list``
- 把发现的 tool 包装成 ``worker.tools.Tool``，name 加前缀 ``mcp_{server}_{tool}``
  避免与内置工具冲突
- 节点结束时 ``close()`` 终止进程

**实施抉择**（spec §7.2 注脚）：MCP server 进程跑在主机而非 sandbox 内——MCP
协议要求长期 stdio 连接，沙箱内反复 exec_command 不可行。这意味着 MCP server
的安全责任由 DAG 作者承担：声明的 ``command + args`` 等同被 ``npx`` 之类直接执行。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from worker.harness import MCPServerSpec
from worker.tools import Tool, ToolResult

_log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30.0
PROTOCOL_VERSION = "2024-11-05"


class MCPError(RuntimeError):
    pass


@dataclass
class _PendingRequest:
    future: asyncio.Future


class MCPClient:
    def __init__(self, server_name: str, process: asyncio.subprocess.Process) -> None:
        self.server_name = server_name
        self._process = process
        self._next_id = 0
        self._tools: list[dict[str, Any]] = []
        self._pending: dict[int, _PendingRequest] = {}
        self._reader_task: asyncio.Task | None = None
        self._closed = False

    @classmethod
    async def connect(
        cls,
        spec: MCPServerSpec,
        *,
        env_overrides: dict[str, str] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> "MCPClient":
        if not spec.command:
            raise MCPError(f"MCP server {spec.name!r} has no command")
        env = {**os.environ, **(env_overrides or {}), **spec.env}
        try:
            proc = await asyncio.create_subprocess_exec(
                spec.command, *spec.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except FileNotFoundError as e:
            raise MCPError(
                f"MCP server {spec.name!r} command not found: {spec.command!r}"
            ) from e
        client = cls(server_name=spec.name, process=proc)
        client._reader_task = asyncio.create_task(
            client._read_loop(), name=f"mcp:{spec.name}:reader"
        )
        try:
            await asyncio.wait_for(client._handshake(), timeout=timeout)
        except Exception:
            await client.close()
            raise
        return client

    async def _handshake(self) -> None:
        await self._request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "multi_agent", "version": "0.5.0"},
        })
        await self._notify("notifications/initialized", {})
        resp = await self._request("tools/list", {})
        self._tools = list(resp.get("tools", []))
        _log.info(
            "MCP server %r connected, discovered %d tools",
            self.server_name, len(self._tools),
        )

    async def call_tool(self, name: str, args: dict[str, Any]) -> str:
        resp = await self._request(
            "tools/call",
            {"name": name, "arguments": args},
            timeout=DEFAULT_TIMEOUT,
        )
        # MCP content list: [{type: "text", text: "..."}, ...]
        parts: list[str] = []
        for c in resp.get("content", []):
            if c.get("type") == "text":
                parts.append(c.get("text", ""))
        return "".join(parts)

    def discovered_tools(self) -> list[dict[str, Any]]:
        return list(self._tools)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._reader_task:
            self._reader_task.cancel()
        if self._process.returncode is None:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=2.0)
            except (asyncio.TimeoutError, ProcessLookupError):
                try:
                    self._process.kill()
                    await self._process.wait()
                except ProcessLookupError:
                    pass
        # 唤醒所有 pending request
        for pending in self._pending.values():
            if not pending.future.done():
                pending.future.set_exception(MCPError("MCP client closed"))
        self._pending.clear()

    async def _request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> dict[str, Any]:
        self._next_id += 1
        msg_id = self._next_id
        future: asyncio.Future[dict[str, Any]] = asyncio.get_event_loop().create_future()
        self._pending[msg_id] = _PendingRequest(future=future)
        await self._write({
            "jsonrpc": "2.0", "id": msg_id,
            "method": method, "params": params,
        })
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError as e:
            self._pending.pop(msg_id, None)
            raise MCPError(
                f"MCP server {self.server_name!r} {method} timed out after {timeout}s"
            ) from e

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        await self._write({"jsonrpc": "2.0", "method": method, "params": params})

    async def _write(self, msg: dict[str, Any]) -> None:
        if self._process.stdin is None:
            raise MCPError("stdin closed")
        data = (json.dumps(msg) + "\n").encode("utf-8")
        self._process.stdin.write(data)
        await self._process.stdin.drain()

    async def _read_loop(self) -> None:
        assert self._process.stdout is not None
        try:
            while True:
                line = await self._process.stdout.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    _log.warning(
                        "MCP %r: non-JSON line: %r", self.server_name, line[:200]
                    )
                    continue
                msg_id = msg.get("id")
                if msg_id is not None and msg_id in self._pending:
                    pending = self._pending.pop(msg_id)
                    if "error" in msg:
                        pending.future.set_exception(
                            MCPError(f"{self.server_name}: {msg['error']}")
                        )
                    else:
                        pending.future.set_result(msg.get("result", {}))
                # 服务端 notifications：当前忽略
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            _log.warning("MCP %r reader exit: %s", self.server_name, e)


# ---------- 把 MCP tools 包装成 Tool ----------


class MCPTool:
    """单个 MCP server 暴露的 tool 的本地 Tool 实现。"""

    def __init__(
        self,
        client: MCPClient,
        tool_def: dict[str, Any],
    ) -> None:
        self._client = client
        self._raw_name = tool_def.get("name", "")
        self.name = f"mcp_{client.server_name}_{self._raw_name}"
        self.description = tool_def.get("description", "")
        self.input_schema = tool_def.get("inputSchema") or {"type": "object", "properties": {}}

    async def run(self, args, *, sandbox, handle) -> ToolResult:
        # sandbox / handle 在 MCP 路径下不使用（MCP 进程在主机）
        try:
            text = await self._client.call_tool(self._raw_name, args or {})
            return ToolResult(text)
        except Exception as e:  # noqa: BLE001
            return ToolResult(f"MCP {self.name} failed: {e}", is_error=True)


async def connect_all(
    specs: list[MCPServerSpec],
    *,
    timeout: float = DEFAULT_TIMEOUT,
) -> tuple[list[MCPClient], list[MCPTool]]:
    """启动所有 MCP server，返回 (clients, flattened_tools)。

    单个 server 失败 → log warning 后跳过，不阻塞其他 server。
    """
    clients: list[MCPClient] = []
    tools: list[MCPTool] = []
    for spec in specs:
        try:
            client = await MCPClient.connect(spec, timeout=timeout)
        except Exception as e:  # noqa: BLE001
            _log.warning("MCP server %r connect failed: %s", spec.name, e)
            continue
        clients.append(client)
        for tdef in client.discovered_tools():
            tools.append(MCPTool(client, tdef))
    return clients, tools


async def close_all(clients: list[MCPClient]) -> None:
    for c in clients:
        try:
            await c.close()
        except Exception as e:  # noqa: BLE001
            _log.warning("MCP %r close failed: %s", c.server_name, e)
