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

- **阶段 4a（DAG 编排 + 失败模型 + 并发）**：✅ 完成
  - `orchestrator/dag_loader.py`：DAG JSON 加载 + schema 校验 + 环检测 + 实例化（逻辑 id → DB node_id）
  - `orchestrator/failure_handler.py`：spec §5.2 表落地——retry 路径清 pending → 重置 pending + `retry_count+1`；耗尽后按 policy 终态（`fail_retry`→failed，`fail_skip`→skipped，`fail_fast`→failed+取消兄弟）
  - `orchestrator/scheduler.py` 重写：`asyncio.Semaphore(MAX_CONCURRENT_WORKERS=5)` 并发，节点失败走 FailureHandler；`fail_fast` 调 `sandbox.cancel`，5s 超时改 `destroy` 强杀；任务级失败后 cascade `skipped` 下游
  - memory_store 加 collection 缓存 + lock（修并发 Chroma `get_or_create` 竞态）
  - demo-phase4a：跑 spec §5.4 完整 DAG（3 并发 research → 2 writing → summarize，含 fail_skip / fail_retry / fail_fast 三种 policy）

- **阶段 4c（context_packer 完整版 + 接力点 + token budget）**：✅ 完成
  - `orchestrator/context_packer.py` 完整版：
    - 四个来源齐全：task.title / 接力原文 / `input_memory_ids` 精确产出 / **语义补充检索（新增）**
    - spec §8.2 query 构造：`title + sub_task + 上游摘要(≤50字)`，总长 ≤ 200 token，超长按 30/20/0 阶梯截断
    - spec §8.2 token budget：默认 2K token；超出按相关度从低到高裁语义补充；底线保留 `task.title + 子任务`，必要时硬截接力原文
    - spec §8.3 memory_level 排序：语义检索时 `task_conclusion` 优先于 `node_output`
  - `state_store` + `dag_loader` 加 `memory_level` 字段（兼容性 ALTER 迁移）
  - CLI `run-task --dag --title --handoff-conv --handoff-range`：通用 DAG 入口 + 接力点参数化（spec §11 阶段 4c 任务 4c.3）
  - **token budget 实测**（run-task 跑 spec §5.4 6 节点 DAG）：所有节点 context 在 137-200 token 之间，远低于 2K 上限
  - **召回基线 v2**（`recall-baseline-v2`）：P@5 = 0.978 / MRR = 0.828；vs v1 的 1.00 / 0.90 略降，是 spec §8.2 把 `title` 拼进 query 的预期 trade-off（真实多 agent 场景需要 title 解决「子任务描述太短」的召回噪音）
  - 测试：133 个 case 全过（19s）

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

# 阶段 4a 完整 DAG demo（3 并发 → 2 写作 → 汇总）
python -m orchestrator.main demo-phase4a --mock --reset
python -m orchestrator.main demo-phase4a --mock --reset --fail-b  # 演示 fail_skip

# 阶段 4c 通用 DAG 入口（含接力点）
python -m orchestrator.main run-task --dag dags/research_report.json \
    --title "选型决策任务" --mock --reset
python -m orchestrator.main run-task --dag dags/research_report.json \
    --title "..." --handoff-conv conv_abc --handoff-range 1,5 --mock

# 真实 Claude API
export ANTHROPIC_API_KEY=...
python -m orchestrator.main demo-phase4a --reset

# 召回质量基线 / 飘移对比
python -m orchestrator.main recall-baseline       # 1.11 基线（query 直接搜）
python -m orchestrator.main recall-baseline-v2    # 4c.5 packer 路径
python -m orchestrator.main recall-drift          # 3.6 id vs 语义对比

# 测试
pytest -v
```

## 文档

- `multi-agent-architecture-spec-v4.md` — 架构规范
- `project-development-plan-v1.md` — 6 阶段开发计划
- `runtime-dashboard-prototype-v2.html` — 仪表盘原型
