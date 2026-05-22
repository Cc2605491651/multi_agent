"""内置工具实现（spec v4 §7，阶段 ABC.B.1）。

每个 tool 实现 ``Tool`` 协议，由 ``ToolRegistry`` 按 ``name`` 路由。
所有真实 IO 走传入的 ``SandboxBackend``（spec §7.1 抽象层），切 e2b/Local 零改动。

5 个内置工具（spec §5.4 示例 DAG 用到的全在）：

- ``read_file(path)``
- ``write_file(path, content)``
- ``exec_command(cmd)``
- ``run_code(code, language="python")``
- ``web_search(query, max_results=5)``：在沙箱里跑 Python 调 DuckDuckGo HTML 端点
"""

from __future__ import annotations

import json
import logging
import textwrap
from dataclasses import dataclass
from typing import Any, Protocol

from worker.harness import ToolSpec
from worker.sandbox import SandboxBackend, SandboxHandle

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolResult:
    content: str
    is_error: bool = False

    def __post_init__(self) -> None:
        # 防御性：截到 8K 字符，避免单次工具返回炸 LLM context
        if len(self.content) > 8000:
            object.__setattr__(
                self, "content", self.content[:8000] + "\n…[truncated]"
            )


class Tool(Protocol):
    name: str
    description: str
    input_schema: dict[str, Any]

    async def run(
        self,
        args: dict[str, Any],
        *,
        sandbox: SandboxBackend,
        handle: SandboxHandle,
    ) -> ToolResult: ...


# ---------- 工具实现 ----------


class ReadFileTool:
    name = "read_file"
    description = "读取沙箱内的文件。path 可以是相对路径（基于 workdir）或绝对路径。"
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }

    async def run(self, args, *, sandbox, handle):
        path = args.get("path")
        if not isinstance(path, str) or not path:
            return ToolResult("path 必须是非空字符串", is_error=True)
        try:
            content = await sandbox.read_file(handle, path)
            return ToolResult(content)
        except Exception as e:  # noqa: BLE001
            return ToolResult(f"read_file failed: {e}", is_error=True)


class WriteFileTool:
    name = "write_file"
    description = "写入沙箱内的文件，覆盖已有内容。"
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    }

    async def run(self, args, *, sandbox, handle):
        path = args.get("path")
        content = args.get("content")
        if not isinstance(path, str) or not path:
            return ToolResult("path 必须是非空字符串", is_error=True)
        if not isinstance(content, str):
            return ToolResult("content 必须是字符串", is_error=True)
        try:
            await sandbox.write_file(handle, path, content)
            return ToolResult(f"wrote {len(content)} chars to {path}")
        except Exception as e:  # noqa: BLE001
            return ToolResult(f"write_file failed: {e}", is_error=True)


class ExecCommandTool:
    name = "exec_command"
    description = "在沙箱里跑 shell 命令；返回 stdout+stderr 合并文本。"
    input_schema = {
        "type": "object",
        "properties": {"cmd": {"type": "string"}},
        "required": ["cmd"],
    }

    async def run(self, args, *, sandbox, handle):
        cmd = args.get("cmd")
        if not isinstance(cmd, str) or not cmd:
            return ToolResult("cmd 必须是非空字符串", is_error=True)
        try:
            out = await sandbox.exec_command(handle, cmd)
            return ToolResult(out)
        except Exception as e:  # noqa: BLE001
            return ToolResult(f"exec_command failed: {e}", is_error=True)


class RunCodeTool:
    name = "run_code"
    description = "在沙箱里跑 Python 代码（写成 .py 后 python3 执行）。"
    input_schema = {
        "type": "object",
        "properties": {"code": {"type": "string"}},
        "required": ["code"],
    }

    async def run(self, args, *, sandbox, handle):
        code = args.get("code")
        if not isinstance(code, str) or not code:
            return ToolResult("code 必须是非空字符串", is_error=True)
        try:
            out = await sandbox.run_code(handle, code)
            return ToolResult(out)
        except Exception as e:  # noqa: BLE001
            return ToolResult(f"run_code failed: {e}", is_error=True)


_WEB_SEARCH_SCRIPT = textwrap.dedent('''\
import sys, json, urllib.parse, urllib.request, html, re
q = sys.argv[1]
max_results = int(sys.argv[2]) if len(sys.argv) > 2 else 5
url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": q})
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
try:
    with urllib.request.urlopen(req, timeout=15) as r:
        body = r.read().decode("utf-8", "replace")
except Exception as e:
    print(json.dumps({"error": str(e)})); sys.exit(0)
# 极简解析：抓 <a class="result__a" href="...">title</a> 和 <a class="result__snippet">
pat = re.compile(r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.S)
snip = re.compile(r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>', re.S)
links = pat.findall(body)[:max_results]
snippets = snip.findall(body)[:max_results]
def clean(s):
    s = re.sub(r'<[^>]+>', '', s)
    return html.unescape(s).strip()
results = []
for i, (u, t) in enumerate(links):
    results.append({
        "title": clean(t),
        "url": u,
        "snippet": clean(snippets[i]) if i < len(snippets) else "",
    })
print(json.dumps({"results": results}, ensure_ascii=False))
''')


class WebSearchTool:
    name = "web_search"
    description = "在沙箱内联网搜索 DuckDuckGo，返回 JSON 数组：[{title, url, snippet}, ...]"
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer", "default": 5},
        },
        "required": ["query"],
    }

    async def run(self, args, *, sandbox, handle):
        query = args.get("query")
        if not isinstance(query, str) or not query:
            return ToolResult("query 必须是非空字符串", is_error=True)
        max_results = args.get("max_results", 5)
        if not isinstance(max_results, int) or max_results <= 0:
            max_results = 5
        try:
            await sandbox.write_file(handle, "_search.py", _WEB_SEARCH_SCRIPT)
            out = await sandbox.exec_command(
                handle,
                f"python3 _search.py {_shell_quote(query)} {max_results}",
            )
            # out 期望是单行 JSON；如果有 stderr 噪音取最后一行
            line = out.strip().splitlines()[-1] if out.strip() else "{}"
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                return ToolResult(
                    f"web_search 返回非 JSON：{out[:500]}", is_error=True
                )
            if "error" in data:
                return ToolResult(
                    f"web_search 网络失败：{data['error']}", is_error=True
                )
            return ToolResult(json.dumps(data, ensure_ascii=False))
        except Exception as e:  # noqa: BLE001
            return ToolResult(f"web_search exception: {e}", is_error=True)


def _shell_quote(s: str) -> str:
    import shlex

    return shlex.quote(s)


# ---------- registry ----------


BUILTIN_TOOLS: dict[str, type[Tool]] = {
    "read_file": ReadFileTool,
    "write_file": WriteFileTool,
    "exec_command": ExecCommandTool,
    "run_code": RunCodeTool,
    "web_search": WebSearchTool,
}


@dataclass
class ToolRegistry:
    tools: dict[str, Tool]

    @classmethod
    def from_specs(cls, specs: list[ToolSpec]) -> "ToolRegistry":
        instances: dict[str, Tool] = {}
        for spec in specs:
            tool_cls = BUILTIN_TOOLS.get(spec.name)
            if tool_cls is None:
                _log.warning(
                    "unknown tool %r in harness; skipping (will appear in dashboard "
                    "but LLM can't call it)",
                    spec.name,
                )
                continue
            tool = tool_cls()
            # 允许 spec.description 覆盖内置 description（节点级定制）
            if spec.description:
                tool = _override_description(tool, spec.description)
            instances[spec.name] = tool
        return cls(tools=instances)

    def names(self) -> list[str]:
        return list(self.tools.keys())

    def get(self, name: str) -> Tool | None:
        return self.tools.get(name)

    async def call(
        self,
        name: str,
        args: dict[str, Any],
        *,
        sandbox: SandboxBackend,
        handle: SandboxHandle,
    ) -> ToolResult:
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult(
                f"未知工具 {name!r}；可用：{self.names()}", is_error=True
            )
        return await tool.run(args, sandbox=sandbox, handle=handle)

    def to_anthropic_schema(self) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self.tools.values()
        ]

    def to_openai_schema(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in self.tools.values()
        ]


def _override_description(tool: Tool, desc: str) -> Tool:
    """简单 wrap：返回带新 description 的 tool（不动原类）。"""

    class _Wrapped:
        name = tool.name
        description = desc
        input_schema = tool.input_schema

        async def run(self, args, *, sandbox, handle):
            return await tool.run(args, sandbox=sandbox, handle=handle)

    return _Wrapped()
