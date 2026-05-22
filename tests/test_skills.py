"""Skills loader 单测（阶段 ABC.C.1）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from worker.harness import SkillSpec
from worker.skills import SkillLoader


@pytest.fixture
def loader(tmp_path: Path) -> SkillLoader:
    project = tmp_path / "skills"
    user = tmp_path / "user_skills"
    project.mkdir()
    user.mkdir()
    (project / "code-review.md").write_text("# code-review\n\n仔细看每行")
    (user / "frontend-design").mkdir()
    (user / "frontend-design" / "SKILL.md").write_text("# frontend-design\n\n高品质 UI")
    return SkillLoader(project_dir=project, user_dir=user)


def test_load_from_project_dir(loader) -> None:
    s = loader.load(SkillSpec(name="code-review"))
    assert "code-review" in s.instructions
    assert "仔细看" in s.instructions
    assert s.source_path is not None


def test_load_from_user_dir(loader) -> None:
    s = loader.load(SkillSpec(name="frontend-design"))
    assert "frontend-design" in s.instructions
    assert "高品质 UI" in s.instructions


def test_load_missing_returns_placeholder(loader) -> None:
    s = loader.load(SkillSpec(name="ghost-skill"))
    assert s.source_path is None
    assert "未找到" in s.instructions


def test_apply_without_invoke_keywords_always_triggers(loader) -> None:
    final, applied = loader.apply(
        [SkillSpec(name="code-review")],
        sub_task_description="任意子任务",
        base_system_prompt="你是 reviewer",
    )
    assert len(applied) == 1
    assert "你是 reviewer" in final
    assert "code-review" in final
    assert "仔细看" in final


def test_apply_with_invoke_keywords_filters(loader) -> None:
    spec = SkillSpec(name="code-review", invoke_keywords=["audit", "审计"])
    # 不匹配
    final, applied = loader.apply(
        [spec], sub_task_description="写一个新功能",
        base_system_prompt="base",
    )
    assert applied == []
    assert final == "base"
    # 匹配中文关键词
    final, applied = loader.apply(
        [spec], sub_task_description="请做安全审计",
        base_system_prompt="base",
    )
    assert len(applied) == 1


def test_apply_combines_multiple_skills(loader) -> None:
    final, applied = loader.apply(
        [SkillSpec(name="code-review"), SkillSpec(name="frontend-design")],
        sub_task_description="任意",
        base_system_prompt="基础提示",
    )
    assert len(applied) == 2
    assert "code-review" in final
    assert "frontend-design" in final
    assert final.index("基础提示") < final.index("code-review")


def test_apply_empty_specs_returns_base(loader) -> None:
    final, applied = loader.apply([], "x", "base")
    assert final == "base"
    assert applied == []


def test_real_project_skills_load() -> None:
    """项目内置 skills 真实加载（structured-output / fact-check）。"""
    real_loader = SkillLoader()
    so = real_loader.load(SkillSpec(name="structured-output"))
    assert "structured-output" in so.instructions
    fc = real_loader.load(SkillSpec(name="fact-check"))
    assert "fact-check" in fc.instructions
