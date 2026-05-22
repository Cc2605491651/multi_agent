"""Planner Agent：自然语言目标 → 合规 DAG JSON（spec v5 §9.7 落地）。

走 ``LLMClient.complete``，按 §3.5 AgentHarness schema 生成节点：

1. 把 schema 描述 + 可用 providers/tools/skills 注入 system prompt
2. 把用户 goal + 可选 dag_id 提示作 user prompt
3. LLM 输出 → 容忍 markdown 代码块包裹 → ``json.loads`` → ``dag_loader.parse_dag``
   严格校验
4. 校验失败把错误回灌给 LLM，重试至多 ``max_retries`` 次
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orchestrator.dag_loader import DagDef, parse_dag
from worker.agent import LLMClient
from worker.skills import PROJECT_SKILLS_DIR

_log = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 2
DEFAULT_MAX_TOKENS = 4096


class PlannerError(RuntimeError):
    """Planner 多次尝试后仍无法生成合规 DAG。"""


@dataclass
class PlannerResult:
    dag_dict: dict[str, Any]
    dag_def: DagDef
    attempts: int
    raw_outputs: list[str]


class Planner:
    def __init__(
        self,
        llm_client: LLMClient,
        *,
        model: str | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        skill_dir: Path | None = None,
        default_provider: str = "deepseek",
        default_model: str = "deepseek-chat",
    ) -> None:
        self._client = llm_client
        self._model = model or default_model
        self._max_retries = max_retries
        self._skill_dir = skill_dir or PROJECT_SKILLS_DIR
        self._default_provider = default_provider
        self._default_model = default_model

    async def plan(
        self,
        goal: str,
        *,
        dag_id_hint: str | None = None,
    ) -> PlannerResult:
        skills_available = self._list_skills()
        system = self._build_system_prompt(skills_available)
        prompt = self._build_user_prompt(goal, dag_id_hint)

        errors: list[str] = []
        raw_outputs: list[str] = []
        for attempt in range(1, self._max_retries + 2):
            raw = await self._client.complete(
                model=self._model, system=system,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=DEFAULT_MAX_TOKENS,
            )
            raw_outputs.append(raw)
            try:
                dag_dict = _extract_json(raw)
                dag_def = parse_dag(dag_dict)
                return PlannerResult(
                    dag_dict=dag_dict, dag_def=dag_def,
                    attempts=attempt, raw_outputs=raw_outputs,
                )
            except (ValueError, json.JSONDecodeError) as e:
                err_msg = f"attempt {attempt}: {e}"
                errors.append(err_msg)
                _log.warning("Planner attempt %d failed: %s", attempt, e)
                if attempt > self._max_retries:
                    break
                prompt = self._build_retry_prompt(goal, raw, str(e))

        raise PlannerError(
            f"Planner failed after {self._max_retries + 1} attempts. "
            f"Errors: {errors}"
        )

    # ---- helpers ----

    def _list_skills(self) -> list[str]:
        if not self._skill_dir.exists():
            return []
        return sorted(p.stem for p in self._skill_dir.glob("*.md"))

    def _build_system_prompt(self, skills: list[str]) -> str:
        skill_listing = ", ".join(skills) if skills else "（暂无）"
        return f"""你是「DAG 设计师」，负责把用户的自然语言目标转换为 multi_agent 系统可执行的 DAG JSON。

## 输出 JSON schema（严格遵守）

{{
  "dag_id": "snake_case_unique_id",
  "description": "一句话描述任务",
  "nodes": [
    {{
      "id": "n1",                              // 节点在 DAG 内的唯一 id
      "name": "research_market",               // 节点角色名（describes what it does）
      "deps": [],                              // 依赖节点 id 列表（必须是已声明的 id）
      "failure_policy": "fail_retry",          // fail_retry | fail_skip | fail_fast
      "max_retries": 2,                        // 可选，默认 2
      "memory_level": "node_output",           // 可选，汇总节点用 "task_conclusion"
      "harness": {{
        "model": "{self._default_model}",
        "provider": "{self._default_provider}",
        "system_prompt": "你是 ... Agent。负责 ...。产出格式：...",
        "tools": ["web_search", "read_file"],
        "skills": [{{"name": "fact-check"}}]
      }}
    }}
  ]
}}

## 可用 provider 与模型

- **deepseek**（默认主用，最便宜）：`deepseek-chat` / `deepseek-reasoner`
- **openrouter**（备选，多模型路由）：`anthropic/claude-sonnet-4` / `openai/gpt-4o-mini` 等
- **anthropic**（仅在用户明确要求 Claude 时用）：`claude-sonnet-4-6` / `claude-opus-4-7`
- **openai**：`gpt-4o-mini` / `gpt-4o`
- **ollama**（本地）：`qwen2.5` 等

**默认建议**：所有节点都用 `deepseek` + `deepseek-chat`；只有当用户明确说要 Claude / GPT 时才改 provider。

## 可用工具（节点声明 `tools` 字段时按需选）

- `read_file(path)`：读沙箱文件
- `write_file(path, content)`：写沙箱文件
- `exec_command(cmd)`：执行 shell 命令
- `run_code(code)`：跑 Python 代码
- `web_search(query, max_results=5)`：联网搜 DuckDuckGo，返回 [{{title, url, snippet}}]

## 可用 skills（项目 skills/ 内置；按需在节点声明 `skills`）

{skill_listing}

## 设计原则

1. 节点数 3-7 个为宜；不要超过 10 个
2. 通常结构：N 个并发调研节点 → M 个写作/分析节点 → 1 个汇总节点
3. 汇总节点：`failure_policy=fail_fast` + `memory_level=task_conclusion`
4. 调研节点：`failure_policy=fail_retry`；可选辅助节点用 `failure_policy=fail_skip`
5. 每个节点的 `system_prompt` 必须写明：身份 + 任务 + 产出格式
6. `tools` 按节点实际需求选，不要全堆上；`web_search` 只给调研类节点
7. `skills` 按 skill 描述选，如汇总节点配 `structured-output`，调研节点配 `fact-check`
8. `deps` 必须是上面已声明的节点 id（不能有循环）

## 输出要求

- **只输出一个 JSON 对象**，不要解释、不要 markdown 代码块包裹
- 所有字段名严格按 schema；所有引号用双引号
- JSON 必须能通过严格 schema 校验
"""

    def _build_user_prompt(self, goal: str, dag_id_hint: str | None) -> str:
        hint = f"\n（建议 dag_id：{dag_id_hint}）" if dag_id_hint else ""
        return f"用户目标：{goal}{hint}\n\n按 schema 输出 DAG JSON："

    def _build_retry_prompt(self, goal: str, prev_output: str, error: str) -> str:
        snippet = prev_output[:600] + ("..." if len(prev_output) > 600 else "")
        return (
            f"用户目标：{goal}\n\n"
            f"上一次输出未通过 schema 校验，错误：{error}\n\n"
            f"上一次输出（前 600 字）：\n{snippet}\n\n"
            f"请重新输出修正后的完整 DAG JSON（仍然只输出 JSON 对象，无任何解释）："
        )


_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL | re.IGNORECASE)


def _extract_json(raw: str) -> dict[str, Any]:
    """容忍 LLM 在 JSON 外面套 markdown 代码块。"""
    text = (raw or "").strip()
    m = _FENCE_RE.match(text)
    if m:
        text = m.group(1).strip()
    # 兜底：找第一个 { 到最后一个 } 之间的子串
    if not text.startswith("{"):
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last > first:
            text = text[first : last + 1]
    return json.loads(text)
