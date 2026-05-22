"""Planner 单测（自然语言 → DAG JSON）。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from orchestrator.planner import (
    Planner,
    PlannerError,
    _extract_json,
)


_VALID_DAG = {
    "dag_id": "test_plan",
    "description": "测试 DAG",
    "nodes": [
        {
            "id": "n1", "name": "research", "deps": [],
            "failure_policy": "fail_retry",
            "harness": {
                "model": "deepseek-chat", "provider": "deepseek",
                "system_prompt": "你是调研 Agent",
                "tools": ["web_search"],
            },
        },
        {
            "id": "n2", "name": "summarize", "deps": ["n1"],
            "failure_policy": "fail_fast",
            "memory_level": "task_conclusion",
            "harness": {
                "model": "deepseek-chat", "provider": "deepseek",
                "system_prompt": "你是汇总 Agent",
                "tools": ["write_file"],
            },
        },
    ],
}


_INVALID_DAG = {
    "dag_id": "bad",
    "nodes": [
        {"id": "n1", "name": "x", "deps": ["ghost_dep"]},
    ],
}


@dataclass
class _ScriptedLLM:
    responses: list[str]
    calls: list[dict] = field(default_factory=list)

    async def complete(self, *, model, system, messages, max_tokens=1024):
        self.calls.append({
            "model": model, "system": system,
            "messages": list(messages), "max_tokens": max_tokens,
        })
        if not self.responses:
            raise RuntimeError("no scripted response left")
        return self.responses.pop(0)


# ---- _extract_json ----


def test_extract_plain_json() -> None:
    out = _extract_json('{"a": 1}')
    assert out == {"a": 1}


def test_extract_strips_markdown_fence_with_lang() -> None:
    raw = "```json\n{\"a\": 1}\n```"
    assert _extract_json(raw) == {"a": 1}


def test_extract_strips_markdown_fence_no_lang() -> None:
    raw = "```\n{\"a\": 1}\n```"
    assert _extract_json(raw) == {"a": 1}


def test_extract_handles_preamble_chatter() -> None:
    raw = "好的，这是 DAG：\n{\"a\": 1}\n以上。"
    assert _extract_json(raw) == {"a": 1}


def test_extract_bad_json_raises() -> None:
    with pytest.raises(json.JSONDecodeError):
        _extract_json("not even json")


# ---- Planner.plan ----


async def test_plan_succeeds_first_try() -> None:
    client = _ScriptedLLM(responses=[json.dumps(_VALID_DAG)])
    planner = Planner(client, max_retries=2)
    result = await planner.plan("做个调研报告")
    assert result.attempts == 1
    assert result.dag_def.dag_id == "test_plan"
    assert len(result.dag_def.nodes) == 2


async def test_plan_succeeds_after_one_retry() -> None:
    client = _ScriptedLLM(responses=[
        json.dumps(_INVALID_DAG),       # 第 1 次：deps 引用不存在节点
        json.dumps(_VALID_DAG),         # 第 2 次：合规
    ])
    planner = Planner(client, max_retries=2)
    result = await planner.plan("xxx")
    assert result.attempts == 2
    assert len(client.calls) == 2
    # 第二次 prompt 应当包含上次错误信息
    second_user = client.calls[1]["messages"][0]["content"]
    assert "未通过 schema 校验" in second_user


async def test_plan_strips_markdown_fence() -> None:
    client = _ScriptedLLM(responses=[
        f"```json\n{json.dumps(_VALID_DAG)}\n```",
    ])
    planner = Planner(client, max_retries=0)
    result = await planner.plan("xxx")
    assert result.dag_def.dag_id == "test_plan"


async def test_plan_exhausts_retries_then_raises() -> None:
    client = _ScriptedLLM(responses=[
        "not json",
        json.dumps(_INVALID_DAG),
        json.dumps({"dag_id": "still_bad"}),  # 缺 nodes
    ])
    planner = Planner(client, max_retries=2)
    with pytest.raises(PlannerError, match="failed after 3 attempts"):
        await planner.plan("xxx")
    assert len(client.calls) == 3


async def test_plan_passes_skills_into_system_prompt(tmp_path) -> None:
    # 临时 skills 目录
    sk = tmp_path / "skills"
    sk.mkdir()
    (sk / "alpha.md").write_text("# alpha")
    (sk / "beta.md").write_text("# beta")

    client = _ScriptedLLM(responses=[json.dumps(_VALID_DAG)])
    planner = Planner(client, max_retries=0, skill_dir=sk)
    await planner.plan("xxx")
    system = client.calls[0]["system"]
    assert "alpha" in system
    assert "beta" in system


async def test_plan_includes_dag_id_hint_in_user_prompt() -> None:
    client = _ScriptedLLM(responses=[json.dumps(_VALID_DAG)])
    planner = Planner(client, max_retries=0)
    await planner.plan("做调研", dag_id_hint="my_special_dag")
    user_prompt = client.calls[0]["messages"][0]["content"]
    assert "my_special_dag" in user_prompt


async def test_default_model_and_provider_in_system() -> None:
    """system prompt 默认建议 deepseek + deepseek-chat。"""
    client = _ScriptedLLM(responses=[json.dumps(_VALID_DAG)])
    planner = Planner(client, max_retries=0)
    await planner.plan("xxx")
    system = client.calls[0]["system"]
    assert "deepseek" in system
    assert "deepseek-chat" in system


async def test_default_uses_deepseek_chat_as_model_arg() -> None:
    """Planner 调 LLM 时 model 参数默认 deepseek-chat。"""
    client = _ScriptedLLM(responses=[json.dumps(_VALID_DAG)])
    planner = Planner(client, max_retries=0)
    await planner.plan("xxx")
    assert client.calls[0]["model"] == "deepseek-chat"


async def test_explicit_model_override() -> None:
    client = _ScriptedLLM(responses=[json.dumps(_VALID_DAG)])
    planner = Planner(client, model="custom-model", max_retries=0)
    await planner.plan("xxx")
    assert client.calls[0]["model"] == "custom-model"
