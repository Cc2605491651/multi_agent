"""交互向导：``multi-agent`` 不带参数时进 TUI 菜单。

设计目标：零命令记忆。新用户敲一个 ``multi-agent`` 就能上路。

主菜单：
- 🚀 新建任务（自然语言）
- 📋 看历史任务
- 🖥  打开仪表盘
- ⚙  配置 / 状态自检
- 退出
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sqlite3
import subprocess
import sys
import webbrowser
from pathlib import Path

try:
    import questionary
except ImportError:  # pragma: no cover
    questionary = None


def _need_questionary() -> bool:
    if questionary is None:
        print(
            "[wizard] 缺 questionary。请装：\n"
            "  pip install questionary\n"
            "或使用 CLI 命令模式（例如 multi-agent plan-task --goal ...）",
            file=sys.stderr,
        )
        return False
    return True


def _data_dir() -> Path:
    from orchestrator.main import DATA_DIR

    return DATA_DIR


def _check_status() -> dict[str, str]:
    """收集运行环境状态，给「⚙ 配置」菜单用。"""
    data_dir = _data_dir()
    keys = {
        "ANTHROPIC_API_KEY": "Anthropic",
        "DEEPSEEK_API_KEY": "DeepSeek",
        "OPENROUTER_API_KEY": "OpenRouter",
        "OPENAI_API_KEY": "OpenAI",
        "E2B_API_KEY": "E2B 沙箱",
        "HTTP_PROXY": "HTTP 代理",
    }
    info: dict[str, str] = {}
    info["数据目录"] = str(data_dir) + (" ✓" if data_dir.exists() else " （首次跑时自动建）")
    info["LLM_PROVIDER"] = os.environ.get("LLM_PROVIDER") or "anthropic（默认）"
    info["SANDBOX_BACKEND"] = os.environ.get("SANDBOX_BACKEND") or "local（默认）"
    for k, label in keys.items():
        info[label] = "✓ 已配置" if os.environ.get(k) else "✗ 未填"
    return info


def _list_tasks_sync(limit: int = 20) -> list[dict]:
    db = _data_dir() / "state.db"
    if not db.is_file():
        return []
    try:
        with sqlite3.connect(db) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, title, dag_id, status, created_at FROM tasks "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        return []


async def _do_plan_task(goal: str, workdir: str | None, provider: str | None) -> None:
    from orchestrator.main import run_plan_task

    if provider:
        os.environ["LLM_PROVIDER"] = provider
    await run_plan_task(
        goal=goal,
        out_path=None,
        mock=False,
        reset=False,
        max_concurrent=3,
        user_id="default_user",
        workdir=workdir,
    )


def _menu_new_task() -> None:
    goal = questionary.text(
        "目标是？（用自然语言描述要让多 agent 做什么）",
        validate=lambda x: True if x.strip() else "不能为空",
    ).ask()
    if not goal:
        return
    workdir = questionary.text(
        "让 agent 在哪个目录工作？（留空 = 用 sandbox 临时目录，不读你的项目文件）",
        default="",
    ).ask()
    workdir = workdir.strip() or None
    if workdir:
        wd = Path(workdir).expanduser().resolve()
        if not wd.exists():
            create = questionary.confirm(
                f"目录 {wd} 不存在，要现在创建吗？", default=False
            ).ask()
            if not create:
                return
            wd.mkdir(parents=True, exist_ok=True)
        # 安全提示
        is_git = (wd / ".git").exists()
        if not is_git:
            ok = questionary.confirm(
                f"⚠️  {wd} 不是 git 仓库，agent 改了文件没法回滚。继续？",
                default=False,
            ).ask()
            if not ok:
                return
        workdir = str(wd)

    provider_choices = [
        questionary.Choice(title="DeepSeek（便宜，国内稳）", value="deepseek"),
        questionary.Choice(title="OpenRouter（一 key 通多家）", value="openrouter"),
        questionary.Choice(title="Anthropic（Claude，最强但贵）", value="anthropic"),
        questionary.Choice(title="OpenAI（GPT）", value="openai"),
        questionary.Choice(title="Ollama（本地）", value="ollama"),
        questionary.Choice(title="保持当前 LLM_PROVIDER env", value=None),
    ]
    provider = questionary.select(
        f"用哪个 provider？（当前 env: {os.environ.get('LLM_PROVIDER') or 'anthropic'}）",
        choices=provider_choices,
    ).ask()

    print()
    print(f"🚀 启动任务：{goal[:60]}{'…' if len(goal) > 60 else ''}")
    if workdir:
        print(f"   工作目录：{workdir}")
    if provider:
        print(f"   provider：{provider}")
    print(
        f"   仪表盘：可另开终端跑 `multi-agent dashboard-serve --port 8765` 然后开浏览器看"
    )
    print()
    asyncio.run(_do_plan_task(goal, workdir, provider))


def _menu_history() -> None:
    tasks = _list_tasks_sync()
    if not tasks:
        print("（暂无历史任务；按任意键返回）")
        input()
        return
    choices = [
        questionary.Choice(
            title=f"{t['title'][:50]}  ·  {t['status']}  ·  {t['created_at']}",
            value=t["id"],
        )
        for t in tasks
    ]
    choices.append(questionary.Choice(title="返回", value=None))
    tid = questionary.select("选一个任务看详情", choices=choices).ask()
    if not tid:
        return
    t = next((x for x in tasks if x["id"] == tid), None)
    if not t:
        return
    print()
    print(f"任务 ID：{t['id']}")
    print(f"  标题：{t['title']}")
    print(f"  DAG ：{t['dag_id']}")
    print(f"  状态：{t['status']}")
    print(f"  创建：{t['created_at']}")
    print()
    if questionary.confirm("打开仪表盘看这个任务？", default=True).ask():
        _menu_dashboard(open_browser=True)


def _menu_dashboard(open_browser: bool = True, port: int = 8765) -> None:
    from orchestrator.main import run_dashboard_serve

    url = f"http://127.0.0.1:{port}"
    print(f"🖥  启动仪表盘 {url}（Ctrl+C 退出）")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        run_dashboard_serve(host="127.0.0.1", port=port)
    except KeyboardInterrupt:
        print("\n仪表盘已停。")


def _menu_status() -> None:
    print()
    print("=== 运行环境状态 ===")
    for k, v in _check_status().items():
        print(f"  {k:15s} : {v}")
    print()
    print("=== 配置入口 ===")
    print("  复制 .env.example 到本目录 .env 然后填 KEY，重启即可生效。")
    print(f"  当前数据目录: {_data_dir()}")
    print(f"  环境变量 MA_DATA_DIR 可覆盖数据目录")
    print(f"  环境变量 MA_WORKDIR 可设默认 agent 工作目录")
    print()
    input("按任意键返回主菜单…")


def run_wizard() -> int:
    """主菜单循环。返回 exit code。"""
    if not _need_questionary():
        return 1

    print()
    print("┌──────────────────────────────────────────┐")
    print("│   多 Agent 协作运行时 · multi-agent-tool  │")
    print("└──────────────────────────────────────────┘")
    print(f"数据目录：{_data_dir()}")
    print()

    while True:
        try:
            choice = questionary.select(
                "想做什么？",
                choices=[
                    questionary.Choice("🚀 新建任务（自然语言描述）", value="new"),
                    questionary.Choice("📋 看历史任务", value="hist"),
                    questionary.Choice("🖥  打开仪表盘（浏览器）", value="dash"),
                    questionary.Choice("⚙  环境状态 / 配置帮助", value="cfg"),
                    questionary.Choice("退出", value="quit"),
                ],
            ).ask()
        except KeyboardInterrupt:
            print("\n再见。")
            return 0
        if choice is None or choice == "quit":
            print("再见。")
            return 0
        if choice == "new":
            _menu_new_task()
        elif choice == "hist":
            _menu_history()
        elif choice == "dash":
            _menu_dashboard(open_browser=True)
        elif choice == "cfg":
            _menu_status()
