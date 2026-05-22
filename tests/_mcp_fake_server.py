"""仅供测试用的极简 MCP server（JSON-RPC 2.0 stdio）。

实现：
- initialize → 返回基本 capabilities
- notifications/initialized → 忽略
- tools/list → 返回 2 个 toy tool（echo, add）
- tools/call → 路由到 echo/add 的实现

启动：python tests/_mcp_fake_server.py
"""

from __future__ import annotations

import json
import sys


TOOLS = [
    {
        "name": "echo",
        "description": "把输入的 text 原样返回",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "add",
        "description": "返回 a + b 的和（int）",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    },
]


def _send(msg: dict) -> None:
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def _handle_tools_call(params: dict) -> dict:
    name = params.get("name", "")
    args = params.get("arguments", {}) or {}
    if name == "echo":
        return {"content": [{"type": "text", "text": str(args.get("text", ""))}]}
    if name == "add":
        try:
            s = int(args.get("a", 0)) + int(args.get("b", 0))
            return {"content": [{"type": "text", "text": str(s)}]}
        except (TypeError, ValueError):
            return {
                "content": [{"type": "text", "text": "invalid args"}],
                "isError": True,
            }
    return {
        "content": [{"type": "text", "text": f"unknown tool {name!r}"}],
        "isError": True,
    }


def main() -> None:
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            continue

        method = msg.get("method")
        msg_id = msg.get("id")
        if method == "initialize":
            _send({"jsonrpc": "2.0", "id": msg_id, "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "fake-mcp", "version": "0.0.1"},
                "capabilities": {"tools": {}},
            }})
        elif method == "notifications/initialized":
            # notification 无 id 无响应
            pass
        elif method == "tools/list":
            _send({"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}})
        elif method == "tools/call":
            _send({
                "jsonrpc": "2.0", "id": msg_id,
                "result": _handle_tools_call(msg.get("params") or {}),
            })
        elif msg_id is not None:
            _send({
                "jsonrpc": "2.0", "id": msg_id,
                "error": {"code": -32601, "message": f"method not found: {method}"},
            })


if __name__ == "__main__":
    main()
