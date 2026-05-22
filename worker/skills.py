"""Skills 加载 + 按需注入（Claude Code 风格指令包，阶段 ABC.C.1）。

每个 ``SkillSpec`` 指向一个 markdown 文件（``instructions_path``），内容是
该技能的执行指引——加载后注入到 agent 的 ``system_prompt``：

- 若 ``invoke_keywords`` 非空：仅当 ``sub_task_description`` 命中任一关键词时注入
- 若 ``invoke_keywords`` 为空：始终注入（被视为节点级强制技能）

instructions_path 解析顺序：

1. 绝对路径直接读
2. 项目根 ``skills/<name>.md``
3. 用户全局 ``~/.claude/skills/<name>/SKILL.md``（Claude Code 习惯）
4. 全找不到 → log warning，注入一个占位「skill <name>: instructions 未找到」

加载结果是 markdown 原文（不解析、不裁剪），由 LLM 在 system prompt 里读。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from worker.harness import SkillSpec

_log = logging.getLogger(__name__)

PROJECT_SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"
USER_SKILLS_DIR = Path.home() / ".claude" / "skills"


@dataclass(frozen=True)
class LoadedSkill:
    name: str
    description: str
    instructions: str
    source_path: str | None  # 加载来源；None 表示用了占位


class SkillLoader:
    def __init__(
        self,
        *,
        project_dir: Path | None = None,
        user_dir: Path | None = None,
    ) -> None:
        self._project_dir = project_dir or PROJECT_SKILLS_DIR
        self._user_dir = user_dir or USER_SKILLS_DIR

    def load(self, spec: SkillSpec) -> LoadedSkill:
        instructions: str | None = None
        source: str | None = None

        # 1. 绝对路径
        if spec.instructions_path:
            p = Path(spec.instructions_path)
            if p.is_absolute() and p.is_file():
                instructions = p.read_text(encoding="utf-8")
                source = str(p)
            elif not p.is_absolute():
                # 相对：先项目再用户
                candidates = [
                    self._project_dir / spec.instructions_path,
                    self._user_dir / spec.instructions_path,
                ]
                for c in candidates:
                    if c.is_file():
                        instructions = c.read_text(encoding="utf-8")
                        source = str(c)
                        break

        # 2. 按 name 找
        if instructions is None:
            candidates = [
                self._project_dir / f"{spec.name}.md",
                self._user_dir / spec.name / "SKILL.md",
                self._user_dir / f"{spec.name}.md",
            ]
            for c in candidates:
                if c.is_file():
                    instructions = c.read_text(encoding="utf-8")
                    source = str(c)
                    break

        if instructions is None:
            _log.warning(
                "skill %r 未找到 instructions（searched: project=%s user=%s）",
                spec.name, self._project_dir, self._user_dir,
            )
            instructions = (
                f"# skill: {spec.name}\n\n"
                f"（instructions 未找到，仅有声明）"
                + (f"\n\n{spec.description}" if spec.description else "")
            )

        return LoadedSkill(
            name=spec.name,
            description=spec.description,
            instructions=instructions,
            source_path=source,
        )

    def apply(
        self,
        specs: list[SkillSpec],
        sub_task_description: str,
        base_system_prompt: str | None,
    ) -> tuple[str | None, list[LoadedSkill]]:
        """返回 (final_system_prompt, applied_skills)。

        - sub_task_description 匹配规则：含任一 invoke_keyword（大小写不敏感）即触发
        - 无 invoke_keywords 的 skill 总是触发
        """
        if not specs:
            return base_system_prompt, []

        applied: list[LoadedSkill] = []
        lower_task = (sub_task_description or "").lower()

        for spec in specs:
            triggered = (
                not spec.invoke_keywords
                or any(kw.lower() in lower_task for kw in spec.invoke_keywords)
            )
            if not triggered:
                continue
            try:
                loaded = self.load(spec)
            except Exception as e:  # noqa: BLE001
                _log.warning("skill %r 加载失败: %s", spec.name, e)
                continue
            applied.append(loaded)

        if not applied:
            return base_system_prompt, []

        sections = []
        if base_system_prompt:
            sections.append(base_system_prompt)
        for s in applied:
            sections.append(
                f"## 技能：{s.name}\n"
                + (f"_{s.description}_\n\n" if s.description else "")
                + s.instructions.strip()
            )
        return "\n\n".join(sections), applied
