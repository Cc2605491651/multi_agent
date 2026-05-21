"""最简 Agent（spec v4 §2 / 阶段 1 任务 1.6）。

- 主对话模型默认 ``claude-sonnet-4-6``；
- 记忆提炼默认 ``claude-haiku-4-5``（决策 D-1.1）；
- 留出 ``LLMClient`` 协议供测试 stub。
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


def default_client() -> AnthropicClient:
    """要求 ``ANTHROPIC_API_KEY`` 已配置；缺则抛错。"""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set; export it or pass api_key explicitly"
        )
    return AnthropicClient()
