"""LLM tool-use 多轮循环（阶段 ABC.B.2）。

抽出来不放进 ``llm_clients.py`` / ``agent.py`` 是因为 Anthropic 和 OpenAI
两家的协议形态差异较大，单独一个模块清晰：

- Anthropic：``messages.create(tools=[...])`` 返回 content blocks，
  含 ``tool_use`` 块时调用工具并把 ``tool_result`` 块塞回 messages，再调一次
- OpenAI compat：``chat/completions`` 返回 ``message.tool_calls`` 列表，
  执行后把 ``role=tool`` 消息塞回 messages

两家共享：``ToolRegistry`` 提供 schema 转换 + 实际调用。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from worker.sandbox import SandboxBackend, SandboxHandle
from worker.tools import ToolRegistry, ToolResult

_log = logging.getLogger(__name__)

DEFAULT_MAX_TURNS = 10


@dataclass
class ToolCallRecord:
    tool_name: str
    args: dict[str, Any]
    result: str
    is_error: bool


@dataclass
class ToolLoopResult:
    final_text: str
    turns: int
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    stop_reason: str = "end_turn"


async def run_anthropic_tool_loop(
    *,
    anthropic_client,  # AnthropicClient（持有 self._client = AsyncAnthropic）
    model: str,
    system: str,
    initial_user: str,
    registry: ToolRegistry,
    sandbox: SandboxBackend,
    handle: SandboxHandle,
    max_tokens: int = 2048,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> ToolLoopResult:
    """spec §7：Anthropic 协议下的 tool-use loop。"""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": initial_user}
    ]
    tools_schema = registry.to_anthropic_schema()
    tool_calls: list[ToolCallRecord] = []

    for turn in range(1, max_turns + 1):
        msg = await anthropic_client._client.messages.create(
            model=model,
            system=system or "",
            messages=messages,
            max_tokens=max_tokens,
            tools=tools_schema if tools_schema else None,
        )
        stop = getattr(msg, "stop_reason", "end_turn")
        # 拼出本轮 assistant 的完整 content（含文本块 + tool_use 块）
        assistant_content: list[Any] = []
        text_pieces: list[str] = []
        tool_uses: list[dict[str, Any]] = []
        for block in msg.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                assistant_content.append({"type": "text", "text": block.text})
                text_pieces.append(block.text)
            elif btype == "tool_use":
                tu = {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": dict(block.input or {}),
                }
                assistant_content.append(tu)
                tool_uses.append(tu)
        messages.append({"role": "assistant", "content": assistant_content})

        if stop != "tool_use" or not tool_uses:
            return ToolLoopResult(
                final_text="".join(text_pieces).strip(),
                turns=turn,
                tool_calls=tool_calls,
                stop_reason=stop,
            )

        # 执行所有 tool_use，把 tool_result 块拼成下一条 user
        tool_results: list[Any] = []
        for tu in tool_uses:
            r = await registry.call(
                tu["name"], tu["input"], sandbox=sandbox, handle=handle
            )
            tool_calls.append(
                ToolCallRecord(
                    tool_name=tu["name"], args=tu["input"],
                    result=r.content, is_error=r.is_error,
                )
            )
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu["id"],
                "content": r.content,
                "is_error": r.is_error,
            })
        messages.append({"role": "user", "content": tool_results})

    _log.warning("tool loop hit max_turns=%d without end_turn", max_turns)
    # 把当前已知文本作为 final（即便不完整）
    last_text = ""
    for m in reversed(messages):
        if m["role"] == "assistant":
            for block in m["content"]:
                if isinstance(block, dict) and block.get("type") == "text":
                    last_text = block["text"]
                    break
            if last_text:
                break
    return ToolLoopResult(
        final_text=last_text, turns=max_turns,
        tool_calls=tool_calls, stop_reason="max_turns",
    )


async def run_openai_tool_loop(
    *,
    base_url: str,
    api_key: str | None,
    extra_headers: dict[str, str],
    model: str,
    system: str,
    initial_user: str,
    registry: ToolRegistry,
    sandbox: SandboxBackend,
    handle: SandboxHandle,
    max_tokens: int = 2048,
    max_turns: int = DEFAULT_MAX_TURNS,
    timeout: float = 180.0,
) -> ToolLoopResult:
    """OpenAI 兼容协议下的 tool-use loop（function calling）。"""
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": initial_user})

    tools_schema = registry.to_openai_schema()
    tool_calls: list[ToolCallRecord] = []

    headers = {"Content-Type": "application/json", **extra_headers}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async with httpx.AsyncClient(timeout=timeout) as client:
        for turn in range(1, max_turns + 1):
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
            }
            if tools_schema:
                payload["tools"] = tools_schema
                payload["tool_choice"] = "auto"
            resp = await client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers=headers, json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            choice = data.get("choices", [{}])[0]
            msg = choice.get("message", {})
            finish_reason = choice.get("finish_reason", "stop")

            content = msg.get("content") or ""
            if isinstance(content, list):
                content = "".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") in ("text", None)
                )
            tcs = msg.get("tool_calls") or []

            # 把 assistant 消息原样塞回
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": content}
            if tcs:
                assistant_msg["tool_calls"] = tcs
            messages.append(assistant_msg)

            if not tcs or finish_reason != "tool_calls":
                return ToolLoopResult(
                    final_text=(content or "").strip(), turns=turn,
                    tool_calls=tool_calls,
                    stop_reason=finish_reason or "stop",
                )

            # 执行每个 tool_call，把结果作为 role=tool 的消息塞回
            for tc in tcs:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args_raw = fn.get("arguments", "{}")
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else dict(args_raw)
                except json.JSONDecodeError:
                    args = {}
                r = await registry.call(
                    name, args, sandbox=sandbox, handle=handle
                )
                tool_calls.append(
                    ToolCallRecord(
                        tool_name=name, args=args,
                        result=r.content, is_error=r.is_error,
                    )
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": r.content,
                })

    _log.warning("openai tool loop hit max_turns=%d", max_turns)
    last_text = ""
    for m in reversed(messages):
        if m.get("role") == "assistant" and m.get("content"):
            last_text = m["content"]
            break
    return ToolLoopResult(
        final_text=last_text, turns=max_turns,
        tool_calls=tool_calls, stop_reason="max_turns",
    )
