# multi_agent

多 Agent 协作系统（spec v4）。

## 当前状态

- **阶段 1（单 Agent + 记忆库）**：✅ 完成
  - `storage/transcript_store.py`：SQLite 对话原文库（async / `asyncio.to_thread` 包同步）
  - `storage/memory_store.py`：Chroma per-user collection + `bge-small-zh-v1.5` / 512 维 + status / cross_task 过滤
  - `worker/sandbox.py`：`SandboxBackend` 抽象 + `LocalBackend`（阶段 4 加 E2BBackend 零改动）
  - `worker/agent.py`：Claude Sonnet 主对话 / Haiku 提炼 + 可 stub 的 `LLMClient` 协议
  - `worker/writeback.py`：阶段 1 简化版（直接 active；阶段 2 升级为 pending → active 三步）
  - `orchestrator/main.py`：CLI `demo-phase1` / `recall-baseline`
  - 测试：60 个 case 全过；召回基线 P@5 = 1.00 / MRR = 0.90（45 query × 20 docs）

## 运行

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 端到端 demo（mock，不打 API）
python -m orchestrator.main demo-phase1 --mock --reset

# 真实 Claude API
export ANTHROPIC_API_KEY=...
python -m orchestrator.main demo-phase1 --reset

# 召回质量基线
python -m orchestrator.main recall-baseline

# 测试
pytest -v
```

## 文档

- `multi-agent-architecture-spec-v4.md` — 架构规范
- `project-development-plan-v1.md` — 6 阶段开发计划
- `runtime-dashboard-prototype-v2.html` — 仪表盘原型
