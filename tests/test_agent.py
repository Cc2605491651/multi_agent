"""Agent 单测（用 stub LLMClient，不打真实 API）。"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from worker.agent import Agent


@dataclass
class _StubClient:
    """记录调用并返回预设响应。"""

    responses: dict[str, str] = field(default_factory=dict)
    calls: list[dict] = field(default_factory=list)

    async def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict],
        max_tokens: int = 1024,
    ) -> str:
        self.calls.append(
            {
                "model": model,
                "system": system,
                "messages": list(messages),
                "max_tokens": max_tokens,
            }
        )
        return self.responses.get(model, "默认回复")


async def test_respond_uses_chat_model_and_history() -> None:
    client = _StubClient(responses={"claude-sonnet-4-6": "好的，已收到"})
    agent = Agent(agent_id="a1", client=client)

    history = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好，有什么可以帮你？"},
    ]
    out = await agent.respond(history, "我想要一只猫")
    assert out == "好的，已收到"

    call = client.calls[-1]
    assert call["model"] == "claude-sonnet-4-6"
    assert call["messages"][-1] == {"role": "user", "content": "我想要一只猫"}
    assert len(call["messages"]) == 3
    assert "中文助手" in call["system"]


async def test_respond_inlines_extra_context() -> None:
    client = _StubClient(responses={"claude-sonnet-4-6": "ok"})
    agent = Agent(agent_id="a1", client=client)
    await agent.respond([], "go", extra_context="task=plan_pets")
    assert "task=plan_pets" in client.calls[-1]["system"]


async def test_distill_uses_haiku_model() -> None:
    client = _StubClient(
        responses={"claude-haiku-4-5": "用户决定养橘猫，命名为米饭"}
    )
    agent = Agent(agent_id="a1", client=client)
    memory = await agent.distill("我决定养一只橘猫，叫米饭", "好的，米饭是个好名字")
    assert memory == "用户决定养橘猫，命名为米饭"
    assert client.calls[-1]["model"] == "claude-haiku-4-5"
    assert client.calls[-1]["max_tokens"] == 200


async def test_distill_strips_quotes() -> None:
    client = _StubClient(responses={"claude-haiku-4-5": "「带引号的结论」"})
    agent = Agent(agent_id="a1", client=client)
    out = await agent.distill("u", "a")
    assert out == "带引号的结论"


async def test_distill_empty_means_nothing_to_remember() -> None:
    client = _StubClient(responses={"claude-haiku-4-5": ""})
    agent = Agent(agent_id="a1", client=client)
    out = await agent.distill("你好", "你好")
    assert out == ""
