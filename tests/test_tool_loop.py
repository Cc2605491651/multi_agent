"""LLM tool-use loop 单测（阶段 ABC.B.2）。

不打真实 API：
- Anthropic loop：mock ``AsyncAnthropic.messages.create`` 模拟多轮 tool_use → end_turn
- OpenAI loop：httpx MockTransport 拦截 chat/completions
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

from worker.harness import ToolSpec
from worker.sandbox import LocalBackend
from worker.tool_loop import (
    DEFAULT_MAX_TURNS,
    run_anthropic_tool_loop,
    run_openai_tool_loop,
)
from worker.tools import ToolRegistry


@pytest.fixture
async def env(tmp_path: Path):
    sandbox = LocalBackend(root_dir=tmp_path / "sb")
    handle = await sandbox.create("ctx")
    registry = ToolRegistry.from_specs(
        [ToolSpec(name="read_file"), ToolSpec(name="write_file")]
    )
    yield sandbox, handle, registry
    await sandbox.destroy(handle)


# ---------- Anthropic loop ----------


@dataclass
class _AnthroBlock:
    type: str
    text: str | None = None
    id: str | None = None
    name: str | None = None
    input: dict | None = None


@dataclass
class _AnthroMsg:
    content: list[_AnthroBlock]
    stop_reason: str = "end_turn"


class _AnthroMessagesAPI:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def create(self, **kwargs):
        # 深拷贝 messages 避免后续 append 污染快照
        import copy

        self.calls.append({
            **kwargs,
            "messages": copy.deepcopy(kwargs.get("messages")),
        })
        if not self.responses:
            raise RuntimeError("no more scripted responses")
        return self.responses.pop(0)


class _AnthroClient:
    def __init__(self, msgs_api):
        self._client = type("X", (), {"messages": msgs_api})()


async def test_anthropic_loop_single_turn_no_tools(env) -> None:
    sandbox, handle, registry = env
    msg = _AnthroMsg(
        content=[_AnthroBlock(type="text", text="直接回答没工具调用")],
        stop_reason="end_turn",
    )
    api = _AnthroMessagesAPI([msg])
    client = _AnthroClient(api)
    res = await run_anthropic_tool_loop(
        anthropic_client=client, model="claude-test",
        system="你是测试 agent", initial_user="hi",
        registry=registry, sandbox=sandbox, handle=handle,
    )
    assert res.final_text == "直接回答没工具调用"
    assert res.turns == 1
    assert res.tool_calls == []
    assert res.stop_reason == "end_turn"


async def test_anthropic_loop_one_tool_then_end(env) -> None:
    sandbox, handle, registry = env
    # 先写一个文件供工具调用读
    await sandbox.write_file(handle, "data.txt", "FILE_CONTENT_42")

    # 第 1 轮：模型说要调 read_file
    turn1 = _AnthroMsg(
        content=[
            _AnthroBlock(type="text", text="让我看看文件"),
            _AnthroBlock(
                type="tool_use", id="tu_1", name="read_file",
                input={"path": "data.txt"},
            ),
        ],
        stop_reason="tool_use",
    )
    # 第 2 轮：拿到结果后给最终答案
    turn2 = _AnthroMsg(
        content=[_AnthroBlock(type="text", text="读到 FILE_CONTENT_42，结论 X")],
        stop_reason="end_turn",
    )
    api = _AnthroMessagesAPI([turn1, turn2])
    client = _AnthroClient(api)
    res = await run_anthropic_tool_loop(
        anthropic_client=client, model="m",
        system="", initial_user="读 data.txt",
        registry=registry, sandbox=sandbox, handle=handle,
    )
    assert res.turns == 2
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0].tool_name == "read_file"
    assert res.tool_calls[0].result == "FILE_CONTENT_42"
    assert "FILE_CONTENT_42" in res.final_text
    # 第 2 轮的 messages 里应当含 tool_result
    last_call = api.calls[-1]
    msgs = last_call["messages"]
    # 倒数第二条是 assistant；倒数第一条是 user 含 tool_result
    assert msgs[-1]["role"] == "user"
    tool_result_blocks = [
        b for b in msgs[-1]["content"]
        if isinstance(b, dict) and b.get("type") == "tool_result"
    ]
    assert tool_result_blocks
    assert tool_result_blocks[0]["tool_use_id"] == "tu_1"


async def test_anthropic_loop_records_tool_error(env) -> None:
    sandbox, handle, registry = env
    turn1 = _AnthroMsg(
        content=[_AnthroBlock(
            type="tool_use", id="tu_1", name="read_file",
            input={"path": ""},  # 空 path → 工具报错
        )],
        stop_reason="tool_use",
    )
    turn2 = _AnthroMsg(
        content=[_AnthroBlock(type="text", text="无法读取")],
        stop_reason="end_turn",
    )
    api = _AnthroMessagesAPI([turn1, turn2])
    res = await run_anthropic_tool_loop(
        anthropic_client=_AnthroClient(api), model="m",
        system="", initial_user="x",
        registry=registry, sandbox=sandbox, handle=handle,
    )
    assert res.tool_calls[0].is_error is True


async def test_anthropic_loop_max_turns(env) -> None:
    sandbox, handle, registry = env
    # 一直返回 tool_use 永远不终止
    infinite = [
        _AnthroMsg(
            content=[_AnthroBlock(
                type="tool_use", id=f"tu_{i}", name="read_file",
                input={"path": "data.txt"},
            )],
            stop_reason="tool_use",
        )
        for i in range(20)
    ]
    await sandbox.write_file(handle, "data.txt", "x")
    res = await run_anthropic_tool_loop(
        anthropic_client=_AnthroClient(_AnthroMessagesAPI(infinite)),
        model="m", system="", initial_user="loop",
        registry=registry, sandbox=sandbox, handle=handle,
        max_turns=3,
    )
    assert res.stop_reason == "max_turns"
    assert res.turns == 3
    assert len(res.tool_calls) == 3


# ---------- OpenAI loop ----------


def _openai_mock(monkeypatch, scripted_responses):
    """scripted_responses: list[dict]，按调用顺序返回对应 chat/completions JSON。"""

    state = {"idx": 0, "requests": []}

    def handler(request: httpx.Request) -> httpx.Response:
        state["requests"].append(json.loads(request.content.decode()))
        if state["idx"] >= len(scripted_responses):
            return httpx.Response(500, json={"error": "no more"})
        resp = scripted_responses[state["idx"]]
        state["idx"] += 1
        return httpx.Response(200, json=resp)

    real_async = httpx.AsyncClient

    class _Patched(real_async):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _Patched)
    return state


async def test_openai_loop_no_tools(env, monkeypatch) -> None:
    sandbox, handle, registry = env
    _openai_mock(monkeypatch, [
        {"choices": [{
            "message": {"role": "assistant", "content": "直接回答"},
            "finish_reason": "stop",
        }]},
    ])
    res = await run_openai_tool_loop(
        base_url="https://x/v1", api_key="k", extra_headers={},
        model="m", system="测试", initial_user="hi",
        registry=registry, sandbox=sandbox, handle=handle,
    )
    assert res.final_text == "直接回答"
    assert res.tool_calls == []


async def test_openai_loop_one_tool_then_end(env, monkeypatch) -> None:
    sandbox, handle, registry = env
    await sandbox.write_file(handle, "data.txt", "AAA")
    state = _openai_mock(monkeypatch, [
        {"choices": [{
            "message": {
                "role": "assistant", "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "data.txt"}',
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }]},
        {"choices": [{
            "message": {"role": "assistant", "content": "看到 AAA"},
            "finish_reason": "stop",
        }]},
    ])
    res = await run_openai_tool_loop(
        base_url="https://x/v1", api_key="k", extra_headers={},
        model="m", system="", initial_user="读 data.txt",
        registry=registry, sandbox=sandbox, handle=handle,
    )
    assert res.turns == 2
    assert res.tool_calls[0].tool_name == "read_file"
    assert res.tool_calls[0].result == "AAA"
    assert "AAA" in res.final_text
    # 第 2 轮 messages 里应有 role=tool
    second_req = state["requests"][1]
    roles = [m["role"] for m in second_req["messages"]]
    assert "tool" in roles


async def test_openai_loop_max_turns(env, monkeypatch) -> None:
    sandbox, handle, registry = env
    await sandbox.write_file(handle, "x", "x")
    forever = [
        {"choices": [{
            "message": {
                "role": "assistant", "content": "",
                "tool_calls": [{
                    "id": f"c{i}", "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "x"}'},
                }],
            },
            "finish_reason": "tool_calls",
        }]}
        for i in range(20)
    ]
    _openai_mock(monkeypatch, forever)
    res = await run_openai_tool_loop(
        base_url="https://x/v1", api_key="k", extra_headers={},
        model="m", system="", initial_user="loop",
        registry=registry, sandbox=sandbox, handle=handle,
        max_turns=2,
    )
    assert res.stop_reason == "max_turns"
    assert res.turns == 2
