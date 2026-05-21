# 多 Agent 协作系统 · 开发计划 v1

> 基于 `multi-agent-architecture-spec-v4.md` 与 `runtime-dashboard-prototype-v2.html`，对 spec §11 的 6 阶段做任务级细化。
>
> 本计划不重复 spec 内容（架构、字段、接口签名见 spec），只回答「下一步具体做什么、怎么验收、风险在哪」。

---

## 0. 假设与口径

- **开发模式**：单人全栈，业余时间，每周 10-15h（其他场景按比例换算）。
- **估时口径**：工时（小时），含写代码 + 单测 + 自测，不含 LLM prompt 摸索的意外耗时。
- **任务发起入口**：阶段 1-3 用 CLI（`python -m orchestrator.main ...`），阶段 4a 起加 FastAPI `POST /tasks`。
- **目录结构**：严格按 spec §12，不重新设计。

---

## 1. 总体策略

- 严格按 spec §11 推进，每阶段必须有可演示 demo 再进下一阶段。
- **对 §11 的调整**：
  - 把阶段 4 拆成 **4a（DAG + 失败模型）/ 4b（沙箱后端可插拔）/ 4c（context_packer 完整版 + 接力点）**，每段 15-40h，独立可验证。
  - 阶段 1 writeback 用简化版（直接 active），阶段 2 升级到 spec §6.2 完整 pending 顺序——避免阶段 1 就实现全套，让阶段 2 没事做。
- 关键路径：阶段 1 → 2 → 3 → 4a → 4b → 4c → 5。阶段 6 独立分支，不在主路径。

---

## 2. 阶段总览

| 阶段 | 目标 | 工时 | 里程碑 demo |
|---|---|---|---|
| 1 | 单 Agent + 记忆库 | 30-40h | CLI 跑对话 → 提炼 → 检索 |
| 2 | 状态库 + 回写原子性 + 崩溃恢复 | 40-50h | 串行 2 节点 DAG + `kill -9` 后清理重跑 |
| 3 | 双 Agent + 精确接力 | 20-30h | 调研 → 写作，验证按 input_memory_ids 取而非靠召回 |
| 4a | DAG 编排 + 失败模型 + 并发 | 30-40h | spec §5.4 示例 DAG 跑通 + fail_fast 取消兄弟节点 |
| 4b | 沙箱后端可插拔（Local + E2B） | 15-25h | 同 DAG 在 Local 和 E2B 上一致 |
| 4c | context_packer 完整版 + 接力点 | 15-25h | 复杂 DAG 每个 Worker 收到的 context 都受 token budget 约束 |
| 5 | 运行时仪表盘接入真实数据 | 15-25h | 浏览器实时看 DAG 节点状态变化 |
| 6（可选） | CubeSandbox POC + 生产化 | 30-60h | KVM 环境跑、E2B_API_URL 切换零改动 |

**总工时：195-285h**，业余 5-8 个月。

---

## 3. 阶段 1 · 单 Agent + 记忆库

**目标**：打通「对话 → 提炼记忆 → 写入 Chroma → 下次检索」全链路。Worker 用 LocalBackend。

### 3.1 任务清单

| # | 任务 | 工时 |
|---|---|---|
| 1.1 | 项目脚手架：git init / pyproject.toml / 目录（按 spec §12） | 2h |
| 1.2 | 依赖锁定：`chromadb==0.4.15`、`sentence-transformers`、`pytest`、`anthropic` | 1h |
| 1.3 | `storage/transcript_store.py`：建表 + add_turn / get_turns_by_range + 单测 | 4h |
| 1.4 | `storage/memory_store.py`：per-user collection、user_id 正则校验、add / search / get_by_ids / update_status + 单测 | 8h |
| 1.5 | Embedding 函数：`bge-small-zh-v1.5` 512 维，封装在 memory_store 内 | 2h |
| 1.6 | `worker/agent.py`：最简 Agent，调用 LLM 产生回复 + 提炼记忆的 prompt | 4h |
| 1.7 | `worker/sandbox.py`：抽象接口骨架（含 cancel / read_file / write_file，spec §7.2）+ LocalBackend 最简实现 | 3h |
| 1.8 | `worker/writeback.py` 简化版：直接写 active 记忆，不带 pending（阶段 2 升级） | 2h |
| 1.9 | 端到端 demo：CLI 跑 5 轮对话 → 提炼 → 检索 | 3h |
| 1.10 | 验证：跨 task 检索默认关闭、user_id 非法被拒（spec §3.2 命名空间规则） | 2h |
| 1.11 | 召回质量摸底：手造 20 条 query 看 P@5（用于阶段 4c 对比基线） | 3h |

### 3.2 验收 demo

```
$ python -m orchestrator.main demo-phase1
[round 1] user: 我有只橘猫叫米饭
[round 1] agent: 好的，米饭...
[memory extracted] "用户有只橘猫，名字叫米饭"
[query: "用户的宠物"] -> 召回: "用户有只橘猫..." (sim=0.83)
```

### 3.3 决策点（阶段 1 启动前拍板）

| ID | 决策 | 推荐 |
|---|---|---|
| D-1.1 | LLM 提供商 | Claude（与现有订阅协同；记忆提炼用 Haiku 省钱，主 Agent 用 Sonnet） |
| D-1.2 | async 接口包同步实现的样板 | `asyncio.to_thread()` 包 chromadb / sentence-transformers，写一个示例文件让其他模块抄 |
| D-1.3 | conversation_id 生成 | `uuid4()` |

### 3.4 风险

- bge 模型首次下载约 100MB，国内可能慢——提前用 Clash 代理拉好，写进 README。
- `chromadb` 锁 `==0.4.15` 而非 `>=`，避免后期同步 API 行为变化。

---

## 4. 阶段 2 · 状态库 + 回写原子性 + 崩溃恢复

**目标**：state_store + spec §6.2 完整 pending 回写顺序 + §6.3 三类崩溃恢复扫描。

### 4.1 任务清单

| # | 任务 | 工时 |
|---|---|---|
| 2.1 | `storage/state_store.py`：tasks / dag_nodes 表（**字段一次到位**，含 failure_policy / retry_count / max_retries / heartbeat_at / input_memory_ids，spec §13 强调） | 6h |
| 2.2 | 开启 SQLite WAL 模式（`PRAGMA journal_mode=WAL`），为阶段 4 并发铺路 | 1h |
| 2.3 | `state_store.list_dag_nodes(task_id)` 实现（阶段 5 仪表盘要用） | 2h |
| 2.4 | writeback v2：严格按 spec §6.2 三步顺序 + Chroma update_status(pending→active) | 8h |
| 2.5 | 心跳上报：Worker 执行中每 30s 更新 heartbeat_at | 4h |
| 2.6 | `orchestrator/recovery.py` 三类扫描全部实现（spec §6.3 类 1/2/3，缺一不可） + 单测 | 12h |
| 2.7 | 故障注入测试：故意 kill -9 / 模拟 Chroma update 失败 | 6h |
| 2.8 | 串行 2 节点 DAG 跑通：node_a → node_b，node_b 用 node_a 的 output_memory_id | 4h |

### 4.2 验收 demo

- 2 节点串行 DAG 正常完成。
- 在 node_a writeback 第 2 步（pending 记忆已写）后 `kill -9` 编排器，重启后 recovery 类 1 扫描清理 pending 记忆 + 节点退回 pending → 自动重跑。
- 模拟 spec §6.2 第 3 步 Chroma update_status 失败，recovery 类 2 扫描发现并修复（pending → active）。

### 4.3 决策点

| ID | 决策 | 推荐 |
|---|---|---|
| D-2.1 | 心跳频率 / 超时阈值 | 30s 心跳 / 5min 超时 |
| D-2.2 | 事务粒度 | 单 writeback 一个事务，避免长事务 |

### 4.4 风险

- recovery 三类扫描幂等性必须经测试覆盖——同时跑两遍不能出错（spec §6.3 末尾强调）。
- SQLite 默认 journal 模式在阶段 4 并发下会撞墙——**阶段 2 就要 WAL**，不要等。

---

## 5. 阶段 3 · 双 Agent + 精确接力

**目标**：验证 input_memory_ids 是确定的指针，不依赖召回质量。

### 5.1 任务清单

| # | 任务 | 工时 |
|---|---|---|
| 3.1 | 编排器调度逻辑：调度某节点前，把上游 done 节点的 output_memory_id 填进 input_memory_ids | 4h |
| 3.2 | `orchestrator/context_packer.py` 早期版：只取 task.title + 接力点原文 + input_memory_ids 精确产出，不做语义补充检索（阶段 4c 再加） | 6h |
| 3.3 | 第二个 Agent（写作 prompt）：拿调研 Agent 的产出写报告 | 4h |
| 3.4 | 端到端 demo：调研 → 写作 | 4h |
| 3.5 | 单测：input_memory_ids 为空、为多个、对应记忆 status 不是 active 的边界 | 4h |
| 3.6 | 对比实验：关掉 input_memory_ids 改成纯语义搜，看召回飘移程度（验证 spec §3.3 「P0 级」的判断） | 4h |

### 5.2 验收 demo

- 调研 Agent 产出"方案 X 因为 Y"。
- 写作 Agent 收到的 context 里**直接出现**"方案 X 因为 Y"原文（按 id 取），不是靠 search 召回的近似词。

### 5.3 风险

- 上游 `fail_skip` 导致 input_memory_ids 出现 null 的情况阶段 3 还不处理（没失败模型），到阶段 4a 再补。本阶段 demo 不构造这种边界。

---

## 6. 阶段 4a · DAG 编排 + 失败模型 + 并发

**目标**：JSON DAG 定义 + spec §5 完整失败模型 + asyncio 并发调度（MAX_CONCURRENT_WORKERS=5）。

### 6.1 任务清单

| # | 任务 | 工时 |
|---|---|---|
| 4a.1 | DAG JSON 加载器：`orchestrator/dag_loader.py` + schema 校验 | 4h |
| 4a.2 | `orchestrator/failure_handler.py`：三种 policy + 重试 + **重试耗尽终态按 spec §5.2 表格落地**（fail_skip → skipped，其他 → failed） | 8h |
| 4a.3 | asyncio 主循环：扫所有 ready 节点 → 并发拉起（受 MAX_CONCURRENT_WORKERS=5 约束） | 8h |
| 4a.4 | 并发节点失败隔离 + fail_fast 取消信号传递（调 SandboxBackend.cancel，超时 5s 后强 destroy） | 6h |
| 4a.5 | Recovery 类 3 扫描：取消信号下的 pending 记忆清理 | 4h |
| 4a.6 | context_packer 升级：上游 skipped 时 input_memory_ids 缺失的显式标注（spec §8.1） | 4h |
| 4a.7 | 完整 DAG demo：spec §5.4 示例（3 并发调研 → 写作 → 汇总，含 1 个 fail_skip 节点和 1 个 fail_retry 节点） | 6h |

### 6.2 验收 demo

- 跑 spec §5.4 示例 DAG，重试到位、跳过到位、终态语义正确。
- 故意让某并发节点 fail_fast，看其他并发兄弟被正确取消（5s 内调 cancel，超时则强杀）。

---

## 7. 阶段 4b · 沙箱后端可插拔

**目标**：SandboxBackend 抽象稳固 + LocalBackend 完善 + E2BBackend 接入。

### 7.1 任务清单

| # | 任务 | 工时 |
|---|---|---|
| 4b.1 | LocalBackend 完善：cancel（asyncio.CancelledError）/ exec_command / read_file / write_file | 4h |
| 4b.2 | E2B 账号申请 + API key 配置 + `pip install e2b-code-interpreter` | 1h |
| 4b.3 | E2BBackend 实现：用 e2b SDK async 版本，把 5 个抽象方法都映射上 | 8h |
| 4b.4 | 配置开关：环境变量 `SANDBOX_BACKEND=local\|e2b` 切换 | 2h |
| 4b.5 | 同一个 DAG 在 Local 和 E2B 各跑一遍，结果一致 | 4h |
| 4b.6 | E2B 成本监控：跑一次 demo 看花了多少美元 | 2h |

### 7.2 决策点

| ID | 决策 | 推荐 |
|---|---|---|
| D-4b.1 | E2B 成本控制 | 默认 Local，重要 demo / 强隔离需求才切 E2B |

### 7.3 风险

- E2B 偶有 cold start 延迟（数秒），影响调试体验——预热脚本可后补。
- E2B 按用量计费，开发期不小心跑死循环会烧钱——加每日预算上限（手动监控即可）。

---

## 8. 阶段 4c · context_packer 完整版 + 接力点

**目标**：spec §8.2 query 拼装 + token budget + 接力点选择。

### 8.1 任务清单

| # | 任务 | 工时 |
|---|---|---|
| 4c.1 | context_packer 升级：上游产出摘要拼到 query（每条 ≤50 字，spec §8.2）+ query 总长 ≤200 token + 截断策略 | 6h |
| 4c.2 | token budget 实现：tiktoken 估算 + 超出按相关度截，底线保留 task.title + 接力原文 + input_memory_ids | 4h |
| 4c.3 | 接力点功能：CLI 选 conversation_id + turn_range，编排器按选定点打包接力原文 | 4h |
| 4c.4 | memory_level 字段启用：node_output / task_conclusion 区分，语义检索按层级排序（spec §8.3） | 4h |
| 4c.5 | 召回质量评测 v2：跑阶段 1 同样的 20 条 query，看 P@5 / MRR 是否提升 | 4h |

### 8.2 验收 demo

- 跑 5+ 节点 DAG，每个 Worker 收到的 context 都在 2K token 内，且包含正确的接力原文 + input_memory_ids 精确产出。

---

## 9. 阶段 5 · 运行时仪表盘

**目标**：把 `runtime-dashboard-prototype-v2.html` 接到真实状态库。

### 9.1 任务清单

| # | 任务 | 工时 |
|---|---|---|
| 5.1 | `orchestrator/api.py`：FastAPI 启动 + `/api/dag-status?task_id=` 走 state_store（**不直连 sqlite3**，spec §10.2） | 4h |
| 5.2 | `dashboard/index.html`：复制原型，删 setTimeout 模拟逻辑，换 `setInterval(poll, 2000)` 调真实接口（原型文件末尾注释已写明改造步骤） | 4h |
| 5.3 | 状态映射：原型已支持 wait/running/done/failed/retrying/skipped 6 态，与 spec §5 状态取值一一对应，直接复用 | 1h |
| 5.4 | 端到端：跑一个真实 DAG，浏览器实时看节点状态变化 | 3h |
| 5.5 | CORS + 仅本地访问 | 2h |
| 5.6 | （可选）节点数 >20 时接 Cytoscape.js 自动布局，原型文件已注明 | 5h |

### 9.2 风险

- `setInterval(2s)` 在节点多 + 任务多时每次全量返回会变慢——原型阶段不优化，记下来。生产化时考虑增量推送（WebSocket）。

---

## 10. 阶段 6（可选）· CubeSandbox POC

**触发条件**：E2B 成本撑不住、或需要更强隔离 / 启动速度。

### 10.1 前置

- 拿到支持 KVM 的 Linux 环境（Mac mini 本身不能直接跑，需 Linux 物理机 / 裸金属 / 经 PVM 的云 VM，spec §7.1）。

### 10.2 任务清单

| # | 任务 | 工时 |
|---|---|---|
| 6.1 | 部署 CubeSandbox 服务端 | 8h |
| 6.2 | POC 验证 spec §7.1 官方宣称数据（< 60ms 冷启动 / < 5MB 内存 / 单机数千实例）在你的硬件上是否真实 | 8h |
| 6.3 | 切 `E2B_API_URL` 指向 CubeSandbox，验证业务代码零改动 | 4h |
| 6.4 | 压力测试：100 并发 Worker | 8h |

---

## 11. 关键风险登记表

| ID | 风险 | 影响阶段 | 缓解 |
|---|---|---|---|
| R-1 | LLM API 限流 / 成本超预期 | 1+ | 阶段 1 埋用量计数器；阶段 4 并发后定每日预算 |
| R-2 | bge embedding 召回质量不达标 | 1, 4c | 阶段 1 末尾跑 20 条 query 摸底；不行就考虑换 bge-base 或 bge-m3 |
| R-3 | SQLite 在阶段 4 并发下性能撞墙 | 4a | 阶段 2 就开 WAL；阶段 4 压测，到瓶颈考虑 Postgres |
| R-4 | recovery 三类扫描有边界没覆盖 | 2 | 单测 + 至少 3 次 kill -9 故障注入 |
| R-5 | E2B 成本失控 | 4b | 默认 Local，重要 demo 才切 E2B |
| R-6 | context_packer 超 budget 时截到关键信息 | 4c | 单测覆盖"截后必须保留 task.title + 接力原文 + input_memory_ids" |
| R-7 | CubeSandbox 官方数据虚标 | 6 | 自有硬件实测决策 |

---

## 12. 决策点登记表（你需要拍板）

| ID | 决策 | 推荐 | 截止 |
|---|---|---|---|
| D-1.1 | LLM 提供商 | Claude（提炼用 Haiku，主 Agent 用 Sonnet） | 阶段 1 启动前 |
| D-1.2 | async 包同步实现样板 | `asyncio.to_thread()` 包 chromadb | 阶段 1 第 1 周 |
| D-1.3 | conversation_id 生成 | `uuid4()` | 阶段 1 |
| D-2.1 | 心跳频率 / 超时阈值 | 30s / 5min | 阶段 2 |
| D-4b.1 | E2B 成本控制 | 默认 Local，按需切 | 阶段 4b |

---

## 13. 测试策略

### 13.1 单测
- 每个 `*_store.py` 独立可测，用 in-memory SQLite + Chroma temp 目录
- `failure_handler.py` 用 mocked sandbox 跑三种 policy

### 13.2 集成测
- writeback + recovery 联合测：故意 kill 进程，看清理到位
- 全链路 DAG 测：固定输入 → 跑完 → 断言每节点最终状态

### 13.3 召回质量评测
- 阶段 1 末（基线）+ 阶段 4c 末（升级后），用同一份 20+ query 集，看 P@5 / MRR

### 13.4 故障注入清单（recovery 必过）
- Worker 写 transcript 后 kill
- Worker 写 pending 记忆后 kill
- 状态库事务成功但 Chroma update 失败（模拟网络抖动）
- 并发节点中其一 fail_fast，验证兄弟被正确取消

---

## 14. 下一步

1. **拍板 D-1.1 ~ D-1.3** 三个决策点。
2. 启动阶段 1 任务 1.1（项目脚手架）。
3. 把本计划做成 GitHub Project 或 Linear，按工时打卡，每完成一个阶段对照"验收 demo"自检。
