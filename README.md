# multi_agent

多 Agent 协作系统（spec v4）。

## 当前状态

- **阶段 1（单 Agent + 记忆库）**：✅ 完成
  - `storage/transcript_store.py`：SQLite 对话原文库（async / `asyncio.to_thread` 包同步）
  - `storage/memory_store.py`：Chroma per-user collection + `bge-small-zh-v1.5` / 512 维 + status / cross_task 过滤
  - `worker/sandbox.py`：`SandboxBackend` 抽象 + `LocalBackend`（阶段 4 加 E2BBackend 零改动）
  - `worker/agent.py`：Claude Sonnet 主对话 / Haiku 提炼 + 可 stub 的 `LLMClient` 协议
  - 召回基线 P@5 = 1.00 / MRR = 0.90（45 query × 20 docs）

- **阶段 2（状态库 + 回写原子性 + 崩溃恢复）**：✅ 完成
  - `storage/state_store.py`：tasks + dag_nodes（字段一次到位），WAL 模式，三类 recovery 查询入口
  - `worker/writeback.py` v2：spec §6.2 三步顺序——transcript → pending memory → 状态库事务（唯一提交点）→ Chroma update active
  - `worker/heartbeat.py`：`HeartbeatTask` async context manager，每 30s 上报
  - `orchestrator/recovery.py`：spec §6.3 三类扫描全部实现 + 幂等性
  - `orchestrator/scheduler.py`：串行拓扑调度（阶段 4a 升级为并发 + 失败模型）
  - 覆盖故障注入（writeback 第 2 步后崩、第 3 步 Chroma 失败、终态节点 pending 残留）

- **阶段 3（双 Agent + 精确接力）**：✅ 完成
  - `orchestrator/context_packer.py` 早期版：task.title + 接力点原文（transcript range）+ input_memory_ids 精确产出，按 depends_on 顺序对齐，上游 skipped 显式注明（不做语义补充检索 / token budget，阶段 4c 落地）
  - scheduler 改用 context_packer 打包 prompt
  - **对比实验数据**（`recall-drift`）：input_memory_ids 精确取 = 100%；语义召回 top-1 = 29%（7 个 query 5 个飘走），top-3 = 86%。验证 spec §3.3 P0 判断
  - 测试：105 个 case 全过（19s）

## 运行

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 阶段 1 端到端 demo（mock，不打 API）
python -m orchestrator.main demo-phase1 --mock --reset

# 阶段 2 串行 2 节点 DAG demo
python -m orchestrator.main demo-phase2 --mock --reset

# 阶段 3 双 Agent + 精确接力 demo
python -m orchestrator.main demo-phase3 --mock --reset

# 真实 Claude API
export ANTHROPIC_API_KEY=...
python -m orchestrator.main demo-phase3 --reset

# 召回质量基线 / 飘移对比
python -m orchestrator.main recall-baseline
python -m orchestrator.main recall-drift

# 测试
pytest -v
```

## 文档

- `multi-agent-architecture-spec-v4.md` — 架构规范
- `project-development-plan-v1.md` — 6 阶段开发计划
- `runtime-dashboard-prototype-v2.html` — 仪表盘原型
