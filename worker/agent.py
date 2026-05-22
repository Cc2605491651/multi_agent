"""Agent + LLMClient 协议（spec v5 §9）。

- 主对话模型默认 ``claude-sonnet-4-6``，记忆提炼默认 ``claude-haiku-4-5``（D-1.1）；
  节点级 ``harness.model`` 覆盖
- 单轮 ``respond`` 给 mock 测试 / 无 tool 节点用；多轮 ``run_with_tools``
  在 harness 声明 tools 时自动启用（按 client 类型分派到 Anthropic / OpenAI tool loop）
- ``LLMClient`` 协议供 ``worker.llm_clients`` 各 provider 实现 + 测试 stub
- ``default_client()`` 按 ``LLM_PROVIDER`` env 工厂选 client
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol


class LLMClient(Protocol):
    async def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict],
        max_tokens: int = 1024,
    ) -> str: ...


class AnthropicClient:
    """Anthropic 官方 SDK 的薄包装。"""

    def __init__(self, api_key: str | None = None) -> None:
        from anthropic import AsyncAnthropic

        self._client = AsyncAnthropic(api_key=api_key) if api_key else AsyncAnthropic()

    async def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict],
        max_tokens: int = 1024,
    ) -> str:
        msg = await self._client.messages.create(
            model=model,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
        )
        # SDK 返回 content blocks 列表，目前只用文本块
        parts: list[str] = []
        for block in msg.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "".join(parts).strip()


_DEFAULT_SYSTEM = (
    "你是一个有用的中文助手。回复尽量简洁、聚焦事实，避免空话。"
)

_DISTILL_SYSTEM = (
    "你是「对话提炼员」。读完一轮用户输入 + Agent 输出，写一句不超过 60 字的中文结论，"
    "概括这一轮里**值得后续 Agent 复用**的事实/决定/偏好。"
    "格式：直接输出一句话，不要加任何前缀、引号或解释。"
    "若这一轮没有值得记的实质信息（如纯客套），输出空字符串。"
)


@dataclass
class Agent:
    agent_id: str
    client: LLMClient
    chat_model: str = "claude-sonnet-4-6"
    distill_model: str = "claude-haiku-4-5"
    system_prompt: str = _DEFAULT_SYSTEM

    async def respond(
        self,
        history: list[dict],
        user_input: str,
        extra_context: str = "",
    ) -> str:
        """根据历史 + 当前用户输入生成回复。

        ``history`` 元素结构：``{"role": "user"|"assistant", "content": str}``。
        """
        system = self.system_prompt
        if extra_context:
            system = f"{system}\n\n## 已知上下文\n{extra_context}"
        messages = list(history) + [{"role": "user", "content": user_input}]
        return await self.client.complete(
            model=self.chat_model,
            system=system,
            messages=messages,
            max_tokens=1024,
        )

    async def run_with_tools(
        self,
        packed_text: str,
        *,
        registry,  # ToolRegistry
        sandbox,  # SandboxBackend
        handle,  # SandboxHandle
        max_tokens: int = 2048,
        max_turns: int = 10,
    ):
        """多轮 tool-use loop（spec §7，阶段 ABC.B.3）。

        根据 client 类型分派：
        - ``AnthropicClient`` → ``run_anthropic_tool_loop``
        - ``OpenAICompatibleClient`` → ``run_openai_tool_loop``

        返回 ``ToolLoopResult``（含 final_text + tool_calls 列表）。
        """
        from worker.llm_clients import OpenAICompatibleClient
        from worker.tool_loop import (
            run_anthropic_tool_loop,
            run_openai_tool_loop,
        )

        if isinstance(self.client, AnthropicClient):
            return await run_anthropic_tool_loop(
                anthropic_client=self.client,
                model=self.chat_model,
                system=self.system_prompt,
                initial_user=packed_text,
                registry=registry,
                sandbox=sandbox, handle=handle,
                max_tokens=max_tokens, max_turns=max_turns,
            )
        if isinstance(self.client, OpenAICompatibleClient):
            return await run_openai_tool_loop(
                base_url=self.client._base,
                api_key=self.client._key,
                extra_headers=self.client._extra_headers,
                model=self.chat_model,
                system=self.system_prompt,
                initial_user=packed_text,
                registry=registry,
                sandbox=sandbox, handle=handle,
                max_tokens=max_tokens, max_turns=max_turns,
            )
        # 不识别的 client（如 mock）→ 退化到无工具单轮
        from worker.tool_loop import ToolLoopResult

        out = await self.respond([], packed_text)
        return ToolLoopResult(final_text=out, turns=1, tool_calls=[])

    async def distill(self, user_input: str, agent_output: str) -> str:
        """从单轮对话里提炼一句记忆。空字符串表示「不值得记」。"""
        prompt = (
            f"【用户输入】\n{user_input}\n\n"
            f"【Agent 输出】\n{agent_output}\n\n"
            f"请按要求输出一句结论。"
        )
        out = await self.client.complete(
            model=self.distill_model,
            system=_DISTILL_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        return out.strip().strip("\"'「」“”")


def default_client() -> "LLMClient":
    """按 ``LLM_PROVIDER`` env 工厂选 LLM；缺 key 抛错。"""
    from worker.llm_clients import make_llm_client

    return make_llm_client()
