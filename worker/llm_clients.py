"""多 LLM provider 客户端 + 工厂（spec v4 §9，阶段 ABC.A.1）。

支持：

- ``anthropic``（默认）—— 走 ``worker.agent.AnthropicClient``，Claude
- ``openai``    —— OpenAI 官方
- ``deepseek``  —— https://api.deepseek.com/v1，兼容 OpenAI Chat Completions
- ``openrouter``—— https://openrouter.ai/api/v1，一个 key 通 100+ 模型
- ``ollama``    —— http://127.0.0.1:11434/v1，本地大模型，无需 key

环境变量：

- ``LLM_PROVIDER``：上面任一字符串；不设默认 ``anthropic``
- ``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY`` / ``DEEPSEEK_API_KEY`` /
  ``OPENROUTER_API_KEY``：各家专属 key
- ``LLM_BASE_URL``：可选，覆盖默认 base_url（自部署兼容服务用）
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol

import httpx


class LLMClient(Protocol):
    async def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict],
        max_tokens: int = 1024,
    ) -> str: ...


@dataclass(frozen=True)
class ProviderConfig:
    """每家 provider 的默认设定。"""

    name: str
    base_url: str | None  # None → 走 Anthropic 自己的 SDK
    env_key: str | None  # 该 provider 的 API key env 名；None 表示本地无需
    default_model: str
    extra_headers: dict[str, str] = field(default_factory=dict)


PROVIDERS: dict[str, ProviderConfig] = {
    "anthropic": ProviderConfig(
        name="anthropic", base_url=None, env_key="ANTHROPIC_API_KEY",
        default_model="claude-sonnet-4-6",
    ),
    "openai": ProviderConfig(
        name="openai", base_url="https://api.openai.com/v1",
        env_key="OPENAI_API_KEY", default_model="gpt-4o-mini",
    ),
    "deepseek": ProviderConfig(
        name="deepseek", base_url="https://api.deepseek.com/v1",
        env_key="DEEPSEEK_API_KEY", default_model="deepseek-chat",
    ),
    "openrouter": ProviderConfig(
        name="openrouter", base_url="https://openrouter.ai/api/v1",
        env_key="OPENROUTER_API_KEY",
        default_model="anthropic/claude-sonnet-4",
        extra_headers={"HTTP-Referer": "https://github.com/cuiyuntao/multi_agent"},
    ),
    "ollama": ProviderConfig(
        name="ollama", base_url="http://127.0.0.1:11434/v1",
        env_key=None, default_model="qwen2.5",
    ),
}


class OpenAICompatibleClient:
    """所有 OpenAI Chat Completions 兼容服务的通用 client，用 httpx 直调。"""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        default_model: str | None = None,
        extra_headers: dict[str, str] | None = None,
        timeout: float = 180.0,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._key = api_key
        self._default_model = default_model
        self._extra_headers = dict(extra_headers or {})
        self._timeout = timeout

    async def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict],
        max_tokens: int = 1024,
    ) -> str:
        # OpenAI 风格 messages：把 system 作为第一条 role=system
        full_msgs: list[dict] = []
        if system:
            full_msgs.append({"role": "system", "content": system})
        full_msgs.extend(messages)

        headers = {"Content-Type": "application/json", **self._extra_headers}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base}/chat/completions",
                headers=headers,
                json={
                    "model": model or self._default_model,
                    "messages": full_msgs,
                    "max_tokens": max_tokens,
                },
            )
            resp.raise_for_status()
            data = resp.json()
        # 取第一个 choice
        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        content = msg.get("content", "")
        # 某些 provider（OpenRouter 部分模型）返回 content 是 list of blocks
        if isinstance(content, list):
            text_parts = [
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") in ("text", None)
            ]
            content = "".join(text_parts)
        return (content or "").strip()


def make_llm_client(provider: str | None = None) -> LLMClient:
    """按 ``LLM_PROVIDER`` env / 入参选 LLM client。

    Anthropic 走原生 SDK；其它走 ``OpenAICompatibleClient``。
    """
    provider = (provider or os.environ.get("LLM_PROVIDER", "anthropic")).strip().lower()
    cfg = PROVIDERS.get(provider)
    if cfg is None:
        raise ValueError(
            f"unknown LLM_PROVIDER={provider!r}; supported: {sorted(PROVIDERS)}"
        )

    if provider == "anthropic":
        # 用原生 SDK，避免 OpenAI 兼容层的损耗
        from worker.agent import AnthropicClient

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set; export it or switch to another LLM_PROVIDER"
            )
        return AnthropicClient(api_key=api_key)

    # OpenAI 兼容：从专属 env 读 key；本地 ollama 无需
    api_key: str | None = None
    if cfg.env_key:
        api_key = os.environ.get(cfg.env_key)
        if not api_key:
            raise RuntimeError(
                f"{cfg.env_key} not set for provider={provider!r}"
            )

    base_url = os.environ.get("LLM_BASE_URL") or cfg.base_url
    if base_url is None:
        raise RuntimeError(f"provider {provider!r} has no base_url configured")

    return OpenAICompatibleClient(
        base_url=base_url,
        api_key=api_key,
        default_model=cfg.default_model,
        extra_headers=cfg.extra_headers,
    )
