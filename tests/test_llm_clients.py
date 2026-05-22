"""LLM provider 工厂 + OpenAI 兼容 client 单测（阶段 ABC.A.1）。

httpx 用 MockTransport 拦截网络调用，不打真实 API。
"""

from __future__ import annotations

import json

import httpx
import pytest

from worker.llm_clients import (
    OpenAICompatibleClient,
    PROVIDERS,
    make_llm_client,
)


# ---- 工厂 ----


def test_factory_anthropic_requires_key(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        make_llm_client()


def test_factory_default_provider_is_anthropic(monkeypatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    client = make_llm_client()
    # 不打 API，只检查类型
    assert client.__class__.__name__ == "AnthropicClient"


@pytest.mark.parametrize(
    "provider,env_var",
    [
        ("openai", "OPENAI_API_KEY"),
        ("deepseek", "DEEPSEEK_API_KEY"),
        ("openrouter", "OPENROUTER_API_KEY"),
    ],
)
def test_factory_openai_compatible_picks_correct_env(
    provider: str, env_var: str, monkeypatch
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv(env_var, "fake-key")
    client = make_llm_client()
    assert isinstance(client, OpenAICompatibleClient)
    assert client._key == "fake-key"
    expected_base = PROVIDERS[provider].base_url
    assert client._base == expected_base.rstrip("/")


def test_factory_ollama_needs_no_key(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = make_llm_client()
    assert isinstance(client, OpenAICompatibleClient)
    assert client._key is None
    assert client._base.startswith("http://127.0.0.1:11434")


def test_factory_unknown_provider_raises(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "bogus")
    with pytest.raises(ValueError, match="unknown LLM_PROVIDER"):
        make_llm_client()


def test_factory_llm_base_url_override(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("LLM_BASE_URL", "https://my.self/v1")
    client = make_llm_client()
    assert client._base == "https://my.self/v1"


def test_factory_missing_env_key_for_provider(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        make_llm_client()


# ---- OpenAICompatibleClient.complete ----


def _mock_transport(captured: dict, response_content: str = "ok"):
    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": response_content}}
                ]
            },
        )
    return httpx.MockTransport(handler)


@pytest.fixture
def patched_client(monkeypatch):
    captured = {}

    real_async = httpx.AsyncClient

    class _Patched(real_async):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = _mock_transport(captured)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _Patched)
    return captured


async def test_openai_compat_complete_basic(patched_client) -> None:
    client = OpenAICompatibleClient(
        base_url="https://x/v1", api_key="k", default_model="gpt-4o-mini",
    )
    out = await client.complete(
        model="gpt-4o-mini", system="be helpful",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert out == "ok"
    body = patched_client["body"]
    # system 应该在 messages 第一条
    assert body["messages"][0] == {"role": "system", "content": "be helpful"}
    assert body["messages"][1] == {"role": "user", "content": "hi"}
    assert body["model"] == "gpt-4o-mini"
    # 默认 max_tokens 1024
    assert body["max_tokens"] == 1024
    # Authorization 头
    assert patched_client["headers"]["authorization"] == "Bearer k"


async def test_openai_compat_no_system(patched_client) -> None:
    client = OpenAICompatibleClient(base_url="https://x/v1", api_key="k")
    await client.complete(
        model="m", system="", messages=[{"role": "user", "content": "hi"}],
    )
    body = patched_client["body"]
    assert body["messages"] == [{"role": "user", "content": "hi"}]


async def test_openai_compat_block_list_content(monkeypatch) -> None:
    """OpenRouter 某些模型返回 content 是 block list；要正确拼回字符串。"""
    # 自定义 transport 返回 block-list content
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "片段一"},
                            {"type": "text", "text": "片段二"},
                        ],
                    }
                }]
            },
        )

    real_async = httpx.AsyncClient

    class _Patched(real_async):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)
    monkeypatch.setattr(httpx, "AsyncClient", _Patched)

    client = OpenAICompatibleClient(base_url="https://x/v1", api_key="k")
    out = await client.complete(
        model="m", system="", messages=[{"role": "user", "content": "hi"}],
    )
    assert out == "片段一片段二"


async def test_openai_compat_no_api_key_omits_auth(monkeypatch) -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
        )

    real_async = httpx.AsyncClient

    class _Patched(real_async):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)
    monkeypatch.setattr(httpx, "AsyncClient", _Patched)

    client = OpenAICompatibleClient(
        base_url="http://localhost:11434/v1", api_key=None,
    )
    await client.complete(
        model="qwen2.5", system="", messages=[{"role": "user", "content": "hi"}],
    )
    assert "authorization" not in captured["headers"]


async def test_openai_compat_http_error_propagates(monkeypatch) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "unauthorized"})

    real_async = httpx.AsyncClient

    class _Patched(real_async):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)
    monkeypatch.setattr(httpx, "AsyncClient", _Patched)

    client = OpenAICompatibleClient(base_url="https://x/v1", api_key="bad")
    with pytest.raises(httpx.HTTPStatusError):
        await client.complete(
            model="m", system="",
            messages=[{"role": "user", "content": "hi"}],
        )


async def test_openrouter_sends_extra_headers(monkeypatch) -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "ok"}}]}
        )

    real_async = httpx.AsyncClient

    class _Patched(real_async):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)
    monkeypatch.setattr(httpx, "AsyncClient", _Patched)

    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    client = make_llm_client()
    await client.complete(
        model="anthropic/claude-sonnet-4", system="",
        messages=[{"role": "user", "content": "hi"}],
    )
    # OpenRouter 推荐带 HTTP-Referer
    assert "http-referer" in captured["headers"]
