# multi_agent

多 Agent 协作系统。架构按 **spec v5**（v4 留作初心档案）。

## 安装

```bash
# 推荐（CLI 隔离，不污染全局 Python）
pipx install multi-agent-tool

# 或一次性临时跑（uv 用户）
uvx --from multi-agent-tool multi-agent

# 装完直接敲：进交互向导，零命令记忆
multi-agent
```

可选环境变量：
- `DEEPSEEK_API_KEY`（默认 LLM，1 块钱跑几十次）
- `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `OPENROUTER_API_KEY`（任选其一）
- `MA_DATA_DIR`（数据落地目录，默认 `~/.multi_agent_tool`）
- `MA_WORKDIR`（agent 工作目录，让它直接读写你的项目）

PyPI 页面：https://pypi.org/project/multi-agent-tool/

## 当前状态

主线 plan §1-§5 + ABC 完整 Harness 体系 + **Planner Agent**（自然语言 → DAG）+ **v6 真协作**（多轮 transcript + 节点级接力）+ **蜡笔小新风 UI**（GSAP + rough.js + dagre）全部交付，**268 测试全过（~25s）**。

- **阶段 1（单 Agent + 记忆库）**：✅
  - `storage/transcript_store.py` SQLite 对话原文（async / `asyncio.to_thread`）
  - `storage/memory_store.py` Chroma per-user collection + `bge-small-zh-v1.5` / 512 维 + status / cross_task 过滤
  - `worker/sandbox.py` `SandboxBackend` 抽象 + `LocalBackend`
  - `worker/agent.py` `LLMClient` 协议 + `Agent` 类
  - 召回基线 P@5 = 1.00 / MRR = 0.90（45 query × 20 docs）

- **阶段 2（状态库 + 回写原子性 + 崩溃恢复）**：✅
  - `storage/state_store.py` tasks + dag_nodes（字段一次到位）+ WAL
  - `worker/writeback.py` v2 spec §6.2 三步顺序
  - `worker/heartbeat.py` 30s 心跳
  - `orchestrator/recovery.py` spec §6.3 三类扫描 + 幂等
  - `orchestrator/scheduler.py` 串行拓扑调度

- **阶段 3（双 Agent + 精确接力）**：✅
  - `orchestrator/context_packer.py` 早期版（task.title + 接力原文 + `input_memory_ids` 精确产出）
  - **对比实验**（`recall-drift`）：id 取 100%；语义召回 top-1 仅 29%（7 query 5 飘）；top-3 = 86%。验证 spec §3.3 「P0 级」判断

- **阶段 4a（DAG 编排 + 失败模型 + 并发）**：✅
  - `orchestrator/dag_loader.py` JSON 加载 + 校验 + 实例化
  - `orchestrator/failure_handler.py` 三 policy + 重试（耗尽前清 pending）
  - scheduler 重写为 `asyncio.Semaphore(MAX_CONCURRENT_WORKERS=5)` 并发；fail_fast 取消信号 + 5s 超时改 destroy
  - memory_store 加 collection 缓存 + Lock 修并发 chroma 竞态

- **阶段 4b（E2B 沙箱后端可插拔）**：✅
  - `worker/sandbox_e2b.py` E2BBackend，6 个方法对齐 e2b 官方 `AsyncSandbox`
  - `worker/sandbox.py` 加 `make_sandbox()` 工厂；`SANDBOX_BACKEND=local|e2b` 切换；业务代码零改动

- **阶段 4c（context_packer 完整版 + token budget）**：✅
  - 四个来源齐全（新增语义补充检索）
  - spec §8.2 query 构造：`title + sub_task + 上游摘要(≤50字)`，≤ 200 token 阶梯截断
  - spec §8.2 token budget ≤ 2K，超出按 distance 从大到小裁；底线保 task.title + sub_task
  - spec §8.3 `memory_level` 排序：task_conclusion 优先
  - 实测 spec §5.4 6 节点 DAG 每节点 context 137-200 token，远低 2K

- **阶段 5（运行时仪表盘）**：✅
  - `dag_nodes` 扩 `model_name` / `tools` 列；DAG JSON 节点可声明
  - `orchestrator/api.py` FastAPI 只读 API（走 `state_store` 不直连 sqlite3）
  - `dashboard/index.html` Cytoscape.js + dagre 自动布局；1.5s 轮询；节点详情卡片

- **ABC 完整 Agent Harness 体系**：✅
  - **A 段**（commit `8f262a6`）：`AgentHarness {model, provider, system_prompt, tools, skills, mcp_servers}` schema 一次到位；5 家 provider 切换（anthropic / openai / deepseek / openrouter / ollama）；dashboard 展示完整 harness
  - **B 段**（commit `f18565a`）：5 个内置 tool（read_file / write_file / exec_command / run_code / web_search）走 SandboxBackend；Anthropic + OpenAI 双家 tool_use loop（max_turns 保护、tool_result 回填、错误记录）
  - **C 段**（commit `0adb887`）：`SkillLoader` 加载 markdown 指令包（项目 `skills/` + 用户 `~/.claude/skills/` 双查找）；`MCPClient` stdio JSON-RPC 2.0；MCP tools 自动 prefix 合入 ToolRegistry；单 server 失败容忍

- **Planner Agent（spec v5 §9.7）**：✅
  - `orchestrator/planner.py`：把自然语言目标转成合规 DAG JSON
  - system prompt 注入 schema + 可用 providers/tools/skills；输出严格 parse_dag 校验；不合规把错误回灌重试 ≤2 次
  - 默认走 `deepseek-chat`（DAG 设计不需要 opus，便宜大碗）
  - CLI `plan-task --goal "..."`：plan → 写 data/planned_<ts>.json → 直接 run-task 一条龙
  - 配套修了 4 个真实跑端到端发现的 bug（详见 spec v5 §14.3 后段）

## 运行

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # 按需填 ANTHROPIC_API_KEY / DEEPSEEK_API_KEY / E2B_API_KEY ...
# `python -m orchestrator.main` 启动时会自动加载 .env（已 export 的优先级更高）

# === mock 模式（不打外网；CI / 自测）===
python -m orchestrator.main demo-phase1 --mock --reset
python -m orchestrator.main demo-phase2 --mock --reset
python -m orchestrator.main demo-phase3 --mock --reset
python -m orchestrator.main demo-phase4a --mock --reset
python -m orchestrator.main demo-phase4a --mock --reset --fail-b   # 演示 fail_skip
python -m orchestrator.main run-task --dag dags/research_report.json \
    --title "选型决策任务" --mock --reset
python -m orchestrator.main run-task --dag dags/research_report.json \
    --title "..." --handoff-conv conv_abc --handoff-range 1,5 --mock

# === Planner Agent：自然语言 → DAG 一条龙（spec v5 §9.7）===
python -m orchestrator.main plan-task --goal "调研 3 个国内开源 RAG 框架并选型" \
    --reset                # 真实模式调 LLM 生成 DAG
python -m orchestrator.main plan-task --goal "随便什么" --mock --reset
                          # mock 模式用 fixture DAG（不调 LLM）

# === 真实 LLM 模式 ===
export ANTHROPIC_API_KEY=...           # 或换 LLM_PROVIDER=deepseek + DEEPSEEK_API_KEY
python -m orchestrator.main demo-phase4a --reset

# === 切换沙箱后端 ===
export SANDBOX_BACKEND=e2b
export E2B_API_KEY=...                 # 从 https://e2b.dev/dashboard 拿
python -m orchestrator.main demo-phase4a --reset   # 业务代码零改动

# === 仪表盘（先跑 run-task 落数据，再起服务）===
python -m orchestrator.main run-task --dag dags/research_report.json \
    --title "演示任务" --mock --reset
python -m orchestrator.main dashboard-serve         # http://127.0.0.1:8000

# === 评估 ===
python -m orchestrator.main recall-baseline        # 1.11 基线（query 直搜）
python -m orchestrator.main recall-baseline-v2     # 4c.5 packer 路径
python -m orchestrator.main recall-drift           # 3.6 id 取 vs 语义召回

# === 测试 ===
pytest -v
```

## 文档

- **`multi-agent-architecture-spec-v6.md`** — 当前架构实施手册（推荐先读，含 v6 真协作）
- `multi-agent-architecture-spec-v5.md` — ABC 段实施手册（v6 之前的最后稳定版）
- `multi-agent-architecture-spec-v4.md` — 初心档案（不再更新）
- `project-development-plan-v1.md` — 6 阶段开发计划
- `runtime-dashboard-prototype-v2.html` — 阶段 5 之前的原型（已被真实 dashboard 替代）
