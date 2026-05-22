# 多 Agent 协作系统 · 架构规范 v6

> v4 = 初心档案；v5 = ABC 段（5 家 LLM provider + tool-use + skills + MCP + Planner）的实施手册；
> v6 = **真正落地「多轮 transcript + 节点级接力」**之后的实施手册（268 测试全过）。
>
> v6 相对 v5 的差别集中在 §3.1（transcript schema 扩列）/ §3.6（**新增**多轮模型）/
> §9.8（**新增**节点级接力 HandoffSpec）/ §14（v5 → v6 变更）。

---

## 0. v6 的定位 —— 真协作

v5 之前的"协作"= **通过提炼后的 memory 精确接力**（spec v4 §3.3 P0 判断）。
有效，但**每个节点的 transcript 只有 turn 1 一条**，下游节点拿不到上游节点的
**多轮思考链路**（含每次 tool_use 的搜索词 / 工具结果），只能看到提炼后的一句话。

用户最初提的诉求："每一个 agent 的 session 都拆分开，不同的 agent 能够从之前已经
执行完成的 agent 的某一个会话介入继而继续或反馈问题"—— v5 没落地。

v6 解决：

1. **节点内多轮 transcript**（§3.6）：tool-use loop 的每一步（user/tool_call/
   tool_result/final）都按独立 turn 落 ``transcript_turns``，含完整 ``turn_meta``
2. **节点级接力点 HandoffSpec**（§9.8）：B 节点 ``harness.handoff`` 声明
   ``{from_node, turn_range}``，context_packer 从上游节点 conv 抓多轮原文拼进 B 的 prompt
3. **协作语义**：下游 agent 看到的不是"上游一句结论"，而是"上游怎么搜的 / 搜到什么 /
   怎么推理的 / 最终给的结论"—— 可以**反审上游、追问、给反馈**

读 v6 的顺序建议：

1. §0：理解 v6 解决了什么协作问题
2. §3.6 + §9.8：新增的两个机制
3. §14：v5 → v6 变更对照表
4. 其它章节与 v5 一致

---

## 1. 系统目标（与 v4 一致）

让多个 Agent 在「共享记忆 + 精确接力 + 可插拔沙箱」之上协作完成复杂任务。强调：

1. **三种数据分开存**（spec v4 §3 哲学，没变）
2. **精确接力优先于语义召回**（v4 §3.3、阶段 3 实测 v4 判断成立：id 取 100%，
   语义召回 top-1 在模糊 query 下飘到 29%；见 `data/recall_drift.json`）
3. **业务代码面向抽象编程**——LLM provider、沙箱后端、工具实现都可换
4. **节点级 Harness 完整配置**（v5 新增哲学，见 §9）

---

## 2. 整体架构

```
                    ┌─────────────────────┐
                    │  DAG JSON 定义       │
                    │  含完整节点 Harness  │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │  Orchestrator        │
                    │  - Scheduler        │
                    │  - context_packer   │
                    │  - failure_handler  │
                    │  - recovery         │
                    │  - dag_loader       │
                    └──────────┬──────────┘
                               ↓
        ┌─────────────────────┼─────────────────────┐
        ↓                     ↓                     ↓
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Worker     │    │  Worker      │    │  Worker      │
│  +Agent     │    │  +Agent      │    │  +Agent      │
│  +Tools     │    │  +Tools      │    │  +Tools      │
│  +Skills    │    │  +MCP        │    │  +Skills+MCP │
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       ↓                    ↓                    ↓
┌──────────────────────────────────────────────────────┐
│  SandboxBackend（Local / E2B / CubeSandbox）         │
└──────────────────────────────────────────────────────┘
              ↓                  ↓                ↓
   ┌─────────────────┐  ┌──────────────┐  ┌──────────────┐
   │ transcript_store │  │ memory_store │  │ state_store  │
   │ (SQLite)         │  │ (Chroma)     │  │ (SQLite)     │
   └─────────────────┘  └──────────────┘  └──────────────┘
                                                  ↓
                                          ┌──────────────┐
                                          │ Dashboard    │
                                          │ (FastAPI +   │
                                          │  Cytoscape)  │
                                          └──────────────┘
```

变化点（vs v4）：

- Worker 内多了 **Tools / Skills / MCP** 三个能力来源（§9）
- Sandbox 从单 Local 扩为可插拔（§7 已实施）
- 多了 Dashboard 层（§11 已实施）

---

## 3. 三层存储 + Harness 数据模型

哲学不变（v4 §3）：对话原文 / 提炼记忆 / 任务状态分库；本节只补 v4 之后新增的字段
和接口。

### 3.1 对话原文库 transcript（**v6 多轮模型**）

模块：`storage/transcript_store.py`。v6 扩两列：

| 字段 | 类型 | 说明 |
|---|---|---|
| `turn_kind` | TEXT NOT NULL DEFAULT 'single' | `single` / `user` / `assistant` / `tool_call` / `tool_result` / `final`（详见 §3.6）|
| `turn_meta` | TEXT JSON | `tool_call` 含 `{tool_name, tool_use_id, args}`；`tool_result` 含 `{tool_use_id, is_error}` |

接口：

```python
class TranscriptStore:
    async def add_turn(*, conversation_id, turn_index,
                       user_input, agent_output, agent_id=None,
                       turn_kind="single", turn_meta=None) -> str
    async def add_tool_loop_turns(*, conversation_id, agent_id,
                                   initial_user_input, loop_result) -> list[str]
    async def get_turns_by_range(conversation_id, start, end) -> list[TranscriptTurn]
    async def count_turns(conversation_id) -> int
```

v5 老 `add_turn` 兼容；新代码建议走 `add_tool_loop_turns`，详见 §3.6。

### 3.2 记忆库 memory_store

模块：`storage/memory_store.py`。v4 §3.2 的核心约束全部保留（per-user collection、
`user_id` 正则 `^[a-zA-Z0-9_-]{1,32}$`、bge-small-zh-v1.5 / 512 维 / cosine、
跨 task 默认关闭）。

v5 新增/修正：

- 新增接口（recovery 需要）：
  ```python
  async def delete(user_id, mem_ids: list[str]) -> int
  async def list_pending_for_node(user_id, node_id) -> list[dict]
  async def get_status(user_id, mem_id) -> str | None
  ```
- 内部加 `_coll_cache + threading.Lock` 修复 chromadb 0.4.15 的 `get_or_create_collection`
  并发竞态（阶段 4a 并发跑碰到，commit `fc633f7`）

### 3.3 状态库 state_store（**字段补全**）

模块：`storage/state_store.py`。**v4 §3.3 的表结构不够，v5 一次到位（含 ABC 字段）**：

```sql
CREATE TABLE tasks (
    id                       TEXT PRIMARY KEY,
    user_id                  TEXT NOT NULL,
    title                    TEXT NOT NULL,
    dag_id                   TEXT NOT NULL,
    handoff_conversation_id  TEXT,
    handoff_turn_range       TEXT,     -- JSON [start, end]
    status                   TEXT NOT NULL,    -- pending/running/done/failed
    created_at               TEXT NOT NULL
);

CREATE TABLE dag_nodes (
    id                TEXT PRIMARY KEY,
    task_id           TEXT NOT NULL,
    node_name         TEXT NOT NULL,
    depends_on        TEXT,            -- JSON 数组（节点 id 列表）
    status            TEXT NOT NULL,   -- pending/running/done/failed/skipped
    failure_policy    TEXT NOT NULL DEFAULT 'fail_retry',
    retry_count       INTEGER NOT NULL DEFAULT 0,
    max_retries       INTEGER NOT NULL DEFAULT 2,
    worker_id         TEXT,
    input_memory_ids  TEXT,            -- JSON 数组（含 null 占位，spec §8.1）
    output_memory_id  TEXT,
    heartbeat_at      TEXT,
    started_at        TEXT,
    finished_at       TEXT,
    memory_level      TEXT NOT NULL DEFAULT 'node_output',  -- node_output | task_conclusion
    model_name        TEXT,            -- v4 没有，阶段 5 加
    tools             TEXT,            -- JSON 数组（兼容老式平铺写法）
    harness           TEXT,            -- JSON：完整 AgentHarness（ABC.A 加）
    FOREIGN KEY(task_id) REFERENCES tasks(id)
);

CREATE INDEX idx_dag_nodes_task   ON dag_nodes(task_id);
CREATE INDEX idx_dag_nodes_status ON dag_nodes(status);
```

`_init_schema` 自带兼容性迁移（ALTER TABLE ADD COLUMN），早期建的库自动补列。

WAL 模式启用，应对阶段 4a 起的并发。

`_utcnow()` 用 milliseconds 精度（spec v4 用 seconds，并发心跳同一秒会被合并，
阶段 2 修正）。

### 3.4 运行日志层（v4 §3.4 沿用）

原型阶段不实现；可选未来。

### 3.6 多轮 transcript 模型（**v6 新增**）

`worker/tool_loop.py` 一次 `run_*_tool_loop` 可能跑 1-10 轮 LLM 调用，每轮含
`assistant text` + 若干 `tool_use` → `tool_result`。v5 之前整段当 1 个 turn 存
(`user_input=packed.text`, `agent_output=loop.final_text`)，下游看不到中间过程。

v6 一次 `ToolLoopResult` 拆成 **2K+2** 个 turn（K = `tool_calls` 数）：

```
turn_index | turn_kind     | user_input          | agent_output         | turn_meta
-----------+---------------+---------------------+----------------------+------------------------
1          | user          | <初始 packed.text>  | ""                   | null
2          | tool_call     | ""                  | ""                   | {tool_name, tool_use_id, args}
3          | tool_result   | ""                  | <stdout 文本>        | {tool_use_id, is_error}
...
2K+2       | final         | ""                  | <loop.final_text>    | {turns, stop_reason}
```

`writeback.writeback_turn` 接到 `loop_result` 参数时自动调
`add_tool_loop_turns` 落多轮；不传则维持单轮 `turn_kind="single"`（向后兼容）。

**对接力的意义**：下游节点 B 通过 §9.8 `HandoffSpec` 取 A 的 conv，能逐 turn
看到 A 的 `web_search` 关键词 / 搜到啥 / 怎么推理。这是 v6 真协作的物理基础。

### 3.5 AgentHarness 数据模型（v5 起；**v6 加 `handoff` 字段**）

模块：`worker/harness.py`。Harness = 让 LLM 真正能干活的全套运行时配置：

```python
@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str = ""
    params: dict = field(default_factory=dict)

@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str = ""
    instructions_path: str | None = None
    invoke_keywords: list[str] = field(default_factory=list)

@dataclass(frozen=True)
class MCPServerSpec:
    name: str
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)

@dataclass(frozen=True)
class HandoffSpec:                       # v6 新增（详见 §9.8）
    from_node: str
    turn_range: list[int] | None = None  # [start, end]；None = 全部 turn

@dataclass(frozen=True)
class AgentHarness:
    model: str | None = None         # 模型名，如 "claude-opus-4-7"
    provider: str | None = None      # anthropic | openai | deepseek | openrouter | ollama
    system_prompt: str | None = None # 节点级 system 覆盖
    tools: list[ToolSpec] = field(default_factory=list)
    skills: list[SkillSpec] = field(default_factory=list)
    mcp_servers: list[MCPServerSpec] = field(default_factory=list)
    handoff: HandoffSpec | None = None    # v6 新增
```

序列化：每个 Spec 支持字符串简写 + 完整 dict 两种 JSON 表达，方便 DAG JSON 手写：

```json
"harness": {
  "model": "claude-sonnet-4-6",
  "provider": "anthropic",
  "system_prompt": "你是「调研 C」Agent",
  "tools": ["web_search", {"name": "exec_command", "description": "..."}],
  "skills": [{"name": "fact-check"}],
  "mcp_servers": [
    {"name": "github", "command": "npx", "args": ["@modelcontextprotocol/server-github"]}
  ]
}
```

`AgentHarness.from_legacy(model_name, tools)` 用于读旧 DAG JSON（平铺
`model` + `tools: list[str]`），自动 fold 为 harness。

---

## 4. 编排器 Orchestrator 职责（v4 §4 沿用 + 加 Harness 阶段）

机械可靠不做创造性判断的原则不变。每个调度循环：

1. 读 DAG（`dag_loader.load_dag` + `instantiate_dag`）
2. 找 ready 节点（依赖已 done/skipped）
3. **解析节点 Harness**（v5 新增）：从 `dag_nodes.harness` 反序列化 `AgentHarness`
4. **填充 `input_memory_ids`**：按 `depends_on` 顺序，含 `None` 占位（spec v4 §8.1）
5. **打包上下文**（`context_packer.pack`，§8 完整版）
6. **拉起 Worker**（`make_sandbox()` + `sandbox.create(context_package=...)`）
7. **应用 Harness**（v5 新增）：
   - 按 `harness.provider` 选 LLMClient（缓存）
   - 按 `harness.model` 实例化 Agent
   - 按 `harness.skills` 注入 system_prompt
   - 按 `harness.mcp_servers` 启动 MCP 子进程，把 mcp tools 合入 ToolRegistry
   - 按 `harness.tools + mcp tools` 决定是否走 tool-use 多轮 loop
8. 追踪心跳（`HeartbeatTask`，30s/拍）
9. 失败按 `failure_policy` 处理（`FailureHandler`，§5）
10. 崩溃恢复扫描（`Recovery`，§6）
11. 销毁 Worker（含 close mcp clients）
12. 循环

### 4.1 并发度（v4 §4.1 已实施）

`asyncio.Semaphore(MAX_CONCURRENT_WORKERS=5)`，可配。所有 Worker 拉起/监控/销毁
+ scheduler 主循环都是 asyncio task。**不混 thread**（v4 强调，仍生效）。

---

## 5. DAG 失败模型（v4 §5 沿用，全部实施）

三种 `failure_policy` 表（v4 §5.2，全部 has test）：

| 重试耗尽后 policy | 节点终态 | 任务 | 兄弟节点 |
|---|---|---|---|
| `fail_retry`（默认） | failed | failed | 不影响 |
| `fail_skip` | skipped | 继续 | 不影响 |
| `fail_fast` | failed | failed | 调 `sandbox.cancel`，超时改 `destroy` |

实施在 `orchestrator/failure_handler.py` + `orchestrator/scheduler.py`。

实施细节（v5 补 v4 漏掉的）：

- **重试前清 pending 记忆**（spec v4 §5.2「重试前先按 §6.3 清理」）：
  `FailureHandler.on_node_failed` retry 分支直接调 `memory_store.list_pending_for_node`
  + `delete`，不依赖通用 recovery 扫描（节点状态没切换时 recovery 类 1/3 都不命中）
- **`cancel_siblings` 后节点 mark skipped**：scheduler `cancel` 路径 catch
  `asyncio.CancelledError` 后必须显式 `mark_node_terminal(skipped)`，否则主循环
  `_all_terminal` 永不退出（阶段 4a 实测的 bug，commit `fc633f7`）

---

## 6. Worker 生命周期 + 回写原子性（v4 §6 沿用，全部实施）

`worker/writeback.py` v2 严格三步顺序：

```
1) transcript_store.add_turn      ← 叶子，先写
2) memory_store.add (status=pending)
3) state_store.commit_node_done   ← 唯一提交点（事务）
3.b) memory_store.update_status(pending → active)   ← 失败留给 recovery 类 2
```

`memory_level` 字段（v4 §8.3 写明方向，v5 实施）：
- writeback v2 写 metadata 时从 `dag_nodes.memory_level` 取（默认 node_output）
- DAG JSON 节点可声明 `"memory_level": "task_conclusion"`（一般汇总节点用）
- 语义检索时 task_conclusion 优先（`context_packer._semantic_supplement`）

`orchestrator/recovery.py` 三类扫描（v4 §6.3 全部实施 + 幂等性测试覆盖）：

- 类 1：`status=running` + 心跳过期 → 清 pending 记忆 + 退回 pending + `retry_count+1`
- 类 2：`status=done` + 关联 mem 仍 `pending` → 重新 `update_status(active)`
- 类 3：`status∈{failed, skipped}` + 关联 pending mem → 删

---

## 7. 可插拔沙箱后端

### 7.1 三种后端对比（v4 §7.1，状态更新）

| 后端 | 实施状态 | 隔离 | 启动 | 适用 |
|---|---|---|---|---|
| `LocalBackend` | ✅ 已实施（阶段 1） | 无 | 即时 | 开发 / 单元测试 / CI |
| `E2BBackend` | ✅ 已实施（阶段 4b） | 中（云端容器） | 秒级 | 重要 demo / 强隔离需求 |
| `CubeSandboxBackend` | 未实施（阶段 6 可选） | 强（KVM） | 官称 <60ms | 生产 / 大规模并发 |

### 7.2 沙箱抽象接口（v4 §7.2 沿用）

`worker/sandbox.py`：6 个 async 方法的 `SandboxBackend` ABC。`SandboxHandle`
dataclass 在 v4 §7.2 已定型，v5 不动。

### 7.3 切换机制（**v5 新增**）

`worker/sandbox.py` 提供 `make_sandbox()` 工厂：

```python
def make_sandbox(backend: str | None = None) -> SandboxBackend:
    backend = (backend or os.environ.get("SANDBOX_BACKEND", "local")).strip().lower()
    if backend == "local":  return LocalBackend()
    if backend == "e2b":    return E2BBackend()  # 见 worker/sandbox_e2b.py
    raise ValueError(...)
```

业务代码（scheduler / demo / run-task）一律调 `make_sandbox()`，**永不直接 import
具体 Backend**。换后端只改 `SANDBOX_BACKEND` env，零代码改动。

E2BBackend 实施细节（`worker/sandbox_e2b.py`）：

- 走 e2b 官方 SDK `AsyncSandbox`（2.x API）
- `create`：拿到 sandbox 后把 `context_package` 写到 `/home/user/context.txt`
- `cancel`：简化为 `asyncio.wait_for(sb.kill(), timeout)`（spec §5.3 「取消尽力而为」）
- `read/write_file`：相对路径锚定 `/home/user`，绝对路径透传
- 配置 env：`E2B_API_KEY` / `E2B_TEMPLATE` / `E2B_SANDBOX_TIMEOUT`

---

## 8. 上下文打包（v4 §8 全部实施）

`orchestrator/context_packer.py` 完整版（spec v4 §8.1-§8.3 的四个来源 + token
budget + memory_level 排序全部落地）：

```python
@dataclass
class PackedContext:
    text: str
    handoff_present: bool
    upstream_present: int
    upstream_missing: int
    semantic_added: int
    semantic_dropped_for_budget: int
    query_used: str
    token_count: int
```

主要参数：

- `max_context_tokens=2000`（v4 §8.2 建议 2K）
- `max_query_tokens=200`（query 长度上限，超出按 50/30/20/0 阶梯截上游摘要）
- `semantic_k=3`

token budget 裁剪策略（v4 §8.2 描述，v5 实施）：

1. 必保段：task.title / handoff 原文 / 上游产出 / 子任务说明
2. 可裁段：语义补充记忆（按 distance 从大到小裁）
3. 必保段已超 budget → 硬截 handoff（保留前半部分，附 "（…接力点过长，已截断…）"）
4. 底线：`task.title + sub_task_description` 永不丢

memory_level 排序（v4 §8.3 方向，v5 实施）：

- `task_conclusion` 优先于 `node_output`
- 同级别按 distance 升序

跑 spec §5.4 6 节点 DAG 实测每节点 context 在 137-200 token 之间，远低 2K 上限。

---

## 9. Agent Harness 体系（**v5 完全新增**）

这是 ABC 全段的总纲。从 spec v4 起的根本扩展：**Agent 不是 LLMClient 包一下就够，
而是一个"装备完整"的运行体**，每个节点声明它需要的模型 / 工具 / 技能 / MCP。

### 9.1 LLM Provider 抽象与切换

模块：`worker/llm_clients.py`。`LLMClient` 协议保持单方法 `complete(*, model,
system, messages, max_tokens)`，但工厂支持五家 provider：

```python
PROVIDERS = {
    "anthropic": {"env_key": "ANTHROPIC_API_KEY", "default_model": "claude-sonnet-4-6"},
    "openai":    {"base_url": "https://api.openai.com/v1", ...},
    "deepseek":  {"base_url": "https://api.deepseek.com/v1", "default_model": "deepseek-chat"},
    "openrouter":{"base_url": "https://openrouter.ai/api/v1",
                  "default_model": "anthropic/claude-sonnet-4",
                  "extra_headers": {"HTTP-Referer": "https://github.com/cuiyuntao/multi_agent"}},
    "ollama":    {"base_url": "http://127.0.0.1:11434/v1", "env_key": None, ...},
}
```

- Anthropic 走原生 SDK（避免兼容层 token 损耗）
- 其余四家走 `OpenAICompatibleClient`（httpx 直调 `/v1/chat/completions`）
- `LLM_BASE_URL` env 可覆盖默认 base_url（自部署兼容服务）
- `LLM_PROVIDER` env 决定默认 client；`harness.provider` 节点级覆盖
- scheduler `_client_for_provider` 按 provider 缓存 client，避免重复构造
- `orchestrator.main` 启动时自动加载项目根 `.env`（`python-dotenv`，
  `override=False`：shell 里已 export 的优先级高于 `.env`）

**默认推荐配置**（`.env.example` 已按此预填）：

| 角色 | provider | 理由 |
|---|---|---|
| 主用 | `deepseek` | 便宜（比 Claude 便宜 10-20 倍）、中文好、长上下文稳 |
| 备选 | `openrouter` | 一个 key 通 100+ 模型，需要 Claude / GPT 时无缝切 |
| 高端推理 | `anthropic`（claude-opus-4-7） | 节点级 `harness.provider="anthropic"` 单点用 |
| 本地 | `ollama` | 离线 / 隐私敏感场景 |

**双重默认陷阱**（实施细节，留意）：

- 代码 fallback：`make_llm_client()` 不传参数 + 无 `LLM_PROVIDER` env →
  hardcoded `"anthropic"`（`worker/llm_clients.py:make_llm_client`）
- 新人配置：`.env.example` 给的 `LLM_PROVIDER=deepseek`

复制 `.env.example` → `.env` 后默认 deepseek；直接跑没 `.env` 的环境（如 CI）
会落到代码 anthropic 默认。两者不一致是有意的：CI 跑测试无 LLM 调用都是 mock，
hardcoded anthropic 不会真打外网；新人开发期推 deepseek 走便宜 API。

**禁止做**（详见 plan 讨论）：复用 Claude Code / Codex 的 OAuth 凭证来"省订阅"
是违反 Anthropic / OpenAI ToS 的，账号封禁风险大。便宜的 provider 用 DeepSeek
或 OpenRouter 即可。

### 9.2 Tool-use 多轮循环

模块：`worker/tool_loop.py`。把"多轮思考 + 工具调用"抽象成一个共享数据结构：

```python
@dataclass
class ToolCallRecord:
    tool_name: str
    args: dict
    result: str
    is_error: bool

@dataclass
class ToolLoopResult:
    final_text: str
    turns: int
    tool_calls: list[ToolCallRecord]
    stop_reason: str  # "end_turn" | "max_turns" | "stop" | ...
```

两个 provider 各一个 loop（协议差异较大）：

- `run_anthropic_tool_loop`：`messages.create(tools=[...])` 循环到
  `stop_reason='end_turn'`；每轮处理 `tool_use` 块，把 `tool_result` 块塞回
  下一条 user 消息
- `run_openai_tool_loop`：`chat/completions(tools=..., tool_choice=auto)` 循环到
  `finish_reason != 'tool_calls'`；每轮处理 `tool_calls`，把 `role=tool` 消息塞回

共享：`ToolRegistry` 提供 schema 转换（`to_anthropic_schema` / `to_openai_schema`）
+ 实际工具调用（`call(name, args, sandbox, handle)`）。

`max_turns` 默认 10（保护），到顶后 `stop_reason="max_turns"`，把已知 assistant
文本作为 final 返回（不抛错，让节点至少有产出）。

### 9.3 内置工具

模块：`worker/tools.py`。5 个工具，全部走 `SandboxBackend` 抽象，Local / E2B 零改动：

| 工具 | 输入 | 行为 |
|---|---|---|
| `read_file` | `{path}` | `sandbox.read_file` |
| `write_file` | `{path, content}` | `sandbox.write_file` |
| `exec_command` | `{cmd}` | `sandbox.exec_command` |
| `run_code` | `{code}` | `sandbox.run_code`（Python 3） |
| `web_search` | `{query, max_results}` | 在沙箱里跑 Python 调 DuckDuckGo HTML 端点解析结果 |

`ToolResult` 强制截断到 8K 字符，避免单次工具返回炸 LLM context。

`ToolRegistry.from_specs(harness.tools)` 路由 spec.name → 内置工具实例；未知 tool
**跳过 + warning**（仪表盘仍展示，但 LLM 调不到），不阻塞节点。

`ToolSpec.description` 可覆盖内置 description（节点级定制）。

### 9.4 Skills 加载（Claude Code 风格）

模块：`worker/skills.py`。每个 `SkillSpec` 指向一个 markdown 文件，内容是
"该技能的执行指引"。加载后注入 agent `system_prompt`。

查找顺序：

1. `instructions_path` 是绝对路径 → 直接读
2. 项目根 `skills/<name>.md`（仓库内置，跟代码走）
3. 用户全局 `~/.claude/skills/<name>/SKILL.md`（Claude Code 习惯）
4. 找不到 → 注入占位「skill <name>: instructions 未找到」+ warning

触发规则：

- `invoke_keywords` 非空：当 `sub_task_description` 含任一关键词（大小写不敏感）才注入
- `invoke_keywords` 为空：始终注入（节点级强制技能）

注入格式：

```
<base_system_prompt>

## 技能：structured-output
_<description>_

<markdown 原文>

## 技能：fact-check
...
```

仓库内置两个示例：`skills/structured-output.md`（汇总节点用）+
`skills/fact-check.md`（调研节点用，含来源标注规则）。

### 9.5 MCP Server 集成

模块：`worker/mcp_client.py`。实现 Anthropic Model Context Protocol（MCP）
stdio JSON-RPC 2.0 client：

- `MCPClient.connect(spec)`：启动子进程（`MCPServerSpec.command + args`）
  - handshake: `initialize` → `notifications/initialized` → `tools/list`
  - 异步 `_read_loop` 把 response 路由到 pending future
- `MCPClient.call_tool(name, args)`：`tools/call` JSON-RPC 调用，解析 content list
  拼回 text
- `MCPTool`：把单个 mcp tool 包装为 `worker.tools.Tool`，name 加 prefix
  `mcp_<server>_<tool>` 防与内置工具冲突
- `connect_all(specs)`：批量启动，单个 server 失败容忍（log warning + 跳过）
- 节点结束 `close_all` 终止所有 mcp 子进程

**实施抉择**（spec v4 §7.2 隐含没解决）：MCP server 跑在主机进程（**不在 sandbox
内**）。原因：MCP 协议要求长期 stdio 连接，沙箱内反复 `exec_command` 不可行。
代价：声明的 `command + args` 等同被直接执行，安全责任由 DAG 作者承担。

测试用 `tests/_mcp_fake_server.py`（极简 echo + add server），不依赖 npm / 真实
MCP server。

### 9.6 Harness 应用顺序（scheduler 内）

```
node.harness 反序列化为 AgentHarness
  │
  ├─→ provider → _client_for_provider() （缓存）
  ├─→ skills + sub_task → skill_loader.apply() → final_system_prompt
  ├─→ Agent(model=harness.model, system_prompt=final_system_prompt, client=...)
  │
  ├─→ mcp_servers → connect_all() → mcp_tools
  ├─→ tools + mcp_tools → ToolRegistry
  │
  ├─→ if registry 非空 + client 支持：agent.run_with_tools()
  │      else：agent.respond() （单轮，向后兼容 mock 测试）
  │
  └─→ finally: close_all(mcp_clients) + sandbox.destroy(handle)
```

向后兼容：harness 任意字段为空都退化到合理默认（无 tool→单轮 respond，无 skill→
直接用 harness.system_prompt 或 Agent 默认，无 mcp→只用内置工具）。

### 9.7 Planner Agent

模块：`orchestrator/planner.py`。把「自然语言目标」转为合规 DAG JSON：

```python
@dataclass
class PlannerResult:
    dag_dict: dict      # 通过 parse_dag 校验的 DAG JSON
    dag_def: DagDef     # 反序列化后的对象
    attempts: int       # 尝试次数（含成功那次）
    raw_outputs: list[str]

class Planner:
    def __init__(self, llm_client, *, model=None,
                 max_retries=2, skill_dir=None,
                 default_provider="deepseek",
                 default_model="deepseek-chat"): ...

    async def plan(self, goal: str, *, dag_id_hint: str | None = None
                   ) -> PlannerResult: ...
```

流程：

1. 把 schema 描述 + 5 个可用 provider + 5 个内置 tool + `skills/` 目录扫描结果
   注入 system prompt
2. LLM 输出 → 容忍 markdown 代码块包裹 / 前后闲聊（`_extract_json` 兜底找
   `{...}`）→ `json.loads` → `dag_loader.parse_dag` 严格校验
3. 失败把错误回灌给 LLM，重试至多 `max_retries` 次（默认 2，共 3 次尝试）
4. 全部失败抛 `PlannerError`

**默认推荐**（spec v5 默认全链路一致）：

- 调 Planner 自己用 `deepseek-chat`（DAG 设计不需要 opus）
- 生成的节点 system prompt 推荐**所有节点**用 `deepseek` + `deepseek-chat`
- 只在用户明确要求 Claude / GPT 时才换 provider

CLI：

```bash
python -m orchestrator.main plan-task --goal "调研 3 个国内开源 RAG 框架并选型" \
    --reset
# → Planner LLM 生成 DAG JSON
# → 写到 data/planned_<ts>.json
# → 直接 instantiate + run-task
```

`--mock` 模式：用最小 fixture DAG（2 节点 research → summarize），不调 LLM，
方便离线验证 plan-task 全链路。

工程量实测约 3h（schema 不变，§3.5 AgentHarness 已能容纳生成结果）。
Planner 是建议系统，不动 dag_loader / scheduler 的边界。

### 9.8 节点级接力 HandoffSpec（**v6 新增**）

让下游 agent 看到上游 agent 的**完整多轮 transcript**（含每个 tool_use 调用过程），
而不是只看提炼后一句记忆——这是 v6「真协作」哲学的关键机制。

DAG JSON 节点声明：

```json
{
  "id": "n2", "name": "reviewer", "deps": ["n1"],
  "harness": {
    "model": "deepseek-chat",
    "system_prompt": "你是评审 agent，看上游 agent 的完整思考链路给反馈",
    "handoff": {
      "from_node": "n1",        // 上游节点的 DAG JSON 逻辑 id
      "turn_range": [1, 99]     // 可选；[start, end] 闭区间；缺省取全部 turn
    }
  }
}
```

落地流程：

1. `dag_loader.parse_dag` 解析 `handoff` → `HandoffSpec(from_node="n1", ...)`
2. `dag_loader.instantiate_dag` 实例化时把 `from_node` 从 DAG 逻辑 id 翻译为真实
   `node_id`（要求 `from_node` 出现在依赖图的拓扑前序）
3. 节点 B 调度时，`context_packer._format_node_handoff(task, node)`：
   - 反序列化 `harness.handoff`
   - conv_id 约定 `conv_{task_id}_{from_node_id}`（scheduler 协议）
   - `transcript_store.get_turns_by_range(conv_id, start, end)`，`turn_range` 缺省时
     用 `count_turns` 取全部
   - 按 `turn_kind` 排版（user / tool_call+args / tool_result+text / final）
4. 打包结果落进 `# 上游节点接力（多轮原文）` 段，与 task 级 `# 接力点（原文）` 并存

**vs §8.1 `input_memory_ids`**：

| 维度 | input_memory_ids（v3 起，精确接力指针） | handoff（v6 节点级，多轮接力） |
|---|---|---|
| 取什么 | 上游节点 `output_memory_id` 提炼后**一句话** | 上游节点 conv 的**多轮原文**（含中间 tool_call/result）|
| 长度 | 短（提炼） | 长（按 turn_range 截）|
| 适合 | 单节点高效串接，下游不关心上游怎么思考 | 评审 / 反审 / 复盘类场景，需看上游决策过程 |
| 命中检索 | 同 task 内被语义补充时也能召回 | 不进语义检索，只通过 handoff 字段拉 |

实施改了哪几个文件：

- `worker/harness.py` 加 `HandoffSpec` + `AgentHarness.handoff`
- `orchestrator/dag_loader.py` parse + instantiate 翻译 `from_node` 逻辑 id
- `orchestrator/context_packer.py` 加 `_format_node_handoff` + `# 上游节点接力` 段
- `PackedContext` 加 `node_handoff_present` / `node_handoff_turn_count` 供仪表盘
- 示例 DAG：`dags/coop_demo.json`（2 节点 research + reviewer）

不动模块：scheduler / writeback / state_store（`harness` 字段已在 JSON 列里）。

---

## 10. 技术栈（**v4 §9 更新**）

```toml
# pyproject.toml 实际依赖
dependencies = [
    "chromadb==0.4.15",            # 记忆库；锁定避免 API 漂移
    "sentence-transformers>=2.2.2", # bge embedding
    "anthropic>=0.39.0",            # Anthropic 原生 SDK
    "tiktoken>=0.7.0",              # token 估算（context_packer）
    "fastapi>=0.110",               # 仪表盘 API
    "uvicorn>=0.27",                # ASGI server
    "e2b>=2.0",                     # E2B 云沙箱 SDK
    "python-dotenv>=1.0",           # orchestrator.main 启动时自动加载 .env
    # httpx 通过 fastapi/anthropic 传入，OpenAI 兼容 client 用它直调
]

# dev
"pytest>=7.4", "pytest-asyncio>=0.23", "pytest-timeout>=2.3"
```

embedding 模型：`BAAI/bge-small-zh-v1.5`（512 维，cosine）—— v4 锁定不能换。

LLM provider 灵活：anthropic / openai / deepseek / openrouter / ollama 五家任选
（见 §9.1）。

DAG JSON 不依赖任何 schema 库（手写校验，避免 jsonschema 重型依赖）。

---

## 11. 运行时仪表盘（v4 §10 实施）

`orchestrator/api.py` + `dashboard/index.html`。

### 11.1 后端 API

只读，走 `state_store` 不直连 sqlite3（v4 §10.2 硬约束）：

- `GET /healthz` → `{ok, db}`
- `GET /api/tasks` → `state_store.list_tasks()` 按 `created_at desc`
- `GET /api/dag-status?task_id=...` → `{task: TaskRow, nodes: [DagNode + harness]}`

每个节点返回字段：`id / name / status / deps / failure_policy / retry_count /
max_retries / worker_id / input_memory_ids / output_memory_id /
heartbeat_at / started_at / finished_at / memory_level / model_name / tools /
harness`（含完整子字段）。

旧节点（没 `harness_json`）：自动从 `model_name + tools` 合成最小 harness 返回，
仪表盘不报错。

### 11.2 前端

`dashboard/index.html`，三栏布局：

- 左侧：任务列表（每 5s 刷新）
- 中央：DAG 图，**Cytoscape.js + dagre 自动布局**（节点拓扑随 task 动态变）
  - 节点 1.5s 轮询；拓扑无变化时只更新状态色，拓扑变了整图重建
  - 颜色编码：pending=灰 / running=蓝 / done=绿 / failed=红 / skipped=灰虚线
  - 边：上游节点 `status=done` 时 `edge.sourceDone=true` → 边变绿（`#3fb950`），
    其余维持灰色（`#30363d`）
- 右侧：节点详情卡片（点节点出现）：
  - 基本：name / id / status / policy / retry / memory_level
  - **Harness · 模型**：model chip + provider chip
  - **Harness · 工具**：chip 群，hover 显示 description
  - **Harness · 技能**：chip 群
  - **Harness · MCP servers**：每个 server 一行（含 command + args）
  - **Harness · System Prompt**：折叠 `<details>`
  - 运行时：worker_id / 时间戳
  - input_memory_ids：含 null 占位「上游跳过」红 chip

### 11.3 启动

```bash
python -m orchestrator.main dashboard-serve --port 8000
# http://127.0.0.1:8000
```

CORS 仅本地。

---

## 12. 分阶段开发进度

v4 §11 描述了 6 个阶段；v5 标完成状态 + 对应 commit：

| 阶段 | 内容 | 状态 | 关键 commit |
|---|---|---|---|
| 1 | 单 Agent + 记忆库 | ✅ | `eeb70b0` |
| 2 | 状态库 + 回写原子性 + 崩溃恢复 | ✅ | `5b76e6a` |
| 3 | 双 Agent + 精确接力 | ✅ | `136d749` |
| 4a | DAG 编排 + 失败模型 + 并发 | ✅ | `fc633f7` |
| 4b | E2BBackend 接入 + SANDBOX_BACKEND 开关 | ✅ | `d15e90c` |
| 4c | context_packer 完整版 + token budget | ✅ | `1b6be56` |
| 5 | 运行时仪表盘 + DAG 节点级 model/tools | ✅ | `5c472d5` |
| ABC.A | AgentHarness schema + 5 家 LLM provider | ✅ | `8f262a6` |
| ABC.B | tool-use 多轮循环 + 5 个内置工具 | ✅ | `f18565a` |
| ABC.C | skills 加载 + MCP server 集成 | ✅ | `0adb887` |
| Planner | 自然语言 → DAG（§9.7）+ 4 个真实跑出的 bug 修复 | ✅ | `4058b10` |
| **v6 真协作** | 多轮 transcript（§3.6）+ 节点级接力（§9.8）| ✅ | 本次 commit |
| 6 | CubeSandbox POC + 生产化 | 未实施 | — |
| UI | 蜡笔小新风 + anime.js + draw.io 风编辑器 | 未实施（下一段）| — |

**测试**：268 个 case，约 25 秒全过（含 16 个 v6 协作模型单测，`tests/test_v6_coop.py`）。

---

## 13. 目录结构（v4 §12 实际形态）

```
multi_agent/
├── orchestrator/
│   ├── api.py              # FastAPI 仪表盘 API（阶段 5）
│   ├── context_packer.py   # 上下文打包（§8 完整版）
│   ├── dag_loader.py       # DAG JSON 解析 + 实例化
│   ├── failure_handler.py  # §5 失败矩阵
│   ├── main.py             # CLI（demo-phase*/run-task/plan-task/dashboard-serve/recall-*）
│   ├── planner.py          # Planner Agent：自然语言 → DAG（§9.7）
│   ├── recovery.py         # §6.3 三类崩溃恢复扫描
│   └── scheduler.py        # 并发主循环 + Harness 应用
├── storage/
│   ├── memory_store.py     # Chroma per-user collection
│   ├── state_store.py      # tasks + dag_nodes
│   └── transcript_store.py # 对话原文
├── worker/
│   ├── agent.py            # Agent + LLMClient 协议 + AnthropicClient
│   ├── harness.py          # AgentHarness + Tool/Skill/MCP Spec（§3.5）
│   ├── heartbeat.py        # HeartbeatTask
│   ├── llm_clients.py      # OpenAICompatibleClient + make_llm_client（§9.1）
│   ├── mcp_client.py       # MCP stdio JSON-RPC client（§9.5）
│   ├── sandbox.py          # SandboxBackend ABC + LocalBackend + make_sandbox
│   ├── sandbox_e2b.py      # E2BBackend（§7.1）
│   ├── skills.py           # SkillLoader（§9.4）
│   ├── tool_loop.py        # Anthropic / OpenAI tool-use loop（§9.2）
│   ├── tools.py            # 5 个内置 tool + ToolRegistry（§9.3）
│   └── writeback.py        # §6.2 三步回写
├── dashboard/
│   └── index.html          # Cytoscape.js 自动布局 + harness 卡片（§11）
├── dags/
│   └── research_report.json # spec §5.4 示例 + 完整 harness 写法
├── skills/                 # 项目内置 skills
│   ├── structured-output.md
│   └── fact-check.md
├── tests/                  # 252 个 case
│   ├── _mcp_fake_server.py # 测试用 fake MCP server
│   └── test_*.py
├── data/                   # 运行时数据（.gitignore）
├── multi-agent-architecture-spec-v4.md  # 初心档案，不动
├── multi-agent-architecture-spec-v5.md  # 本文档
├── project-development-plan-v1.md
├── runtime-dashboard-prototype-v2.html  # 原型（已被真实 dashboard 替代）
├── README.md
├── .env.example
└── pyproject.toml          # state_store 内嵌 _SCHEMA 是建表真源；schema.sql 已删
```

---

## 14. v4 → v5 → v6 变更概览

### v5 → v6 增量（本节新加）

| 项 | v5 | v6 |
|---|---|---|
| transcript_turns 字段 | 7 列 | 9 列（+`turn_kind` / `turn_meta`，§3.1 / §3.6）|
| 每节点 transcript 轮数 | 永远 1 | tool-use 节点 2K+2 轮（K=tool_calls 数），单轮节点维持 1 |
| 节点间接力 | 仅 `input_memory_ids`（一句话）| **加 `harness.handoff`（多轮原文）**|
| 下游能看到上游什么 | 一句提炼记忆 | 整个思考过程：搜索词 / 搜到啥 / 推理 / 结论 |
| AgentHarness 新字段 | — | `handoff: HandoffSpec \| None` |
| 新接口 | — | `transcript_store.add_tool_loop_turns` / `count_turns`；`context_packer._format_node_handoff` |
| 测试覆盖 | 252 | 268（+16，`tests/test_v6_coop.py`）|

### v5 之前的变更

### 14.1 数据模型

| 项 | v4 | v5 |
|---|---|---|
| `dag_nodes` 字段 | 14 列 | 18 列（+`memory_level` / `model_name` / `tools` / `harness`） |
| 时间戳精度 | seconds | milliseconds（修并发心跳合并 bug） |
| memory_store 接口 | add/search/get_by_ids/update_status | +delete / +list_pending_for_node / +get_status |
| AgentHarness | 不存在 | §3.5 新增 |

### 14.2 模块

| 模块 | v4 状态 | v5 状态 |
|---|---|---|
| `worker/llm_clients.py` | 不存在 | 5 家 provider 工厂 |
| `worker/harness.py` | 不存在 | AgentHarness + 3 个 Spec |
| `worker/tools.py` | 不存在 | 5 工具 + ToolRegistry |
| `worker/tool_loop.py` | 不存在 | Anthropic + OpenAI 双家 loop |
| `worker/skills.py` | 不存在 | SkillLoader |
| `worker/mcp_client.py` | 不存在 | MCP stdio client |
| `worker/sandbox_e2b.py` | 不存在 | E2BBackend |
| `orchestrator/api.py` | 不存在 | FastAPI 仪表盘 |
| `orchestrator/dag_loader.py` | 不存在 | DAG JSON 加载 + 实例化 |
| `orchestrator/failure_handler.py` | 不存在 | §5 落地 |
| `orchestrator/recovery.py` | 不存在 | §6.3 三类扫描 |
| `orchestrator/planner.py` | 不存在 | Planner Agent（§9.7） |

### 14.3 行为修正

- **并发 Chroma get_or_create**：v4 没考虑；v5 加 `_coll_cache + Lock`（commit `fc633f7`）
- **fail_fast cancel 后节点状态**：v4 说"由 §6.3 清理"但实际 scheduler 必须显式
  `mark_node_terminal(skipped)`，否则主循环 deadlock（v5 §5 注脚）
- **重试前清 pending**：v4 §5.2 提了"先按 §6.3 清理"但 recovery 类 1/3 在重试时
  状态没切换都不命中；v5 在 `FailureHandler` 内手动 `list_pending_for_node + delete`

ABC 段交付后的二次发现（commit `05b5d08` + `ced8880`）：

- **mock 路径漏传 force_default_client**：demo-phase{2,3,4a} 三处 Scheduler 构造
  漏给 mock 模式开 `force_default_client=True`，当 DAG 节点 harness 含非默认
  provider（如 `deepseek`）时会真去要 API key。统一补
- **sandbox handle 早期错误泄漏**：`_execute_node` 的 try-finally 边界曾过窄，
  sandbox.create 之后到 HeartbeatTask 之前的代码任何路径抛错都会跳过 finally；
  扩 try 块覆盖整段，mcp_clients 在 try 前预声明给 finally 用
- **api.py 直连 sqlite3**：违反 §11.1 自身约束。`state_store` 补
  `async list_tasks() -> list[TaskRow]`，`api.py` 改调
- **stale_running 时间戳精度**：cutoff 用 seconds、心跳用 milliseconds 字符串
  比较亚秒抖动；统一改 milliseconds
- **dashboard edge 上游 done 变绿**：原 `edge[?targetDone]` 数据从未赋值，规则
  永不命中；改为 `edge[?sourceDone]`，render + updateStatuses 时按上游 status==done
  设值，§11.2 颜色编码描述对得上
- **`.env` 自动加载**：`orchestrator/main.py` 启动时 `_load_dotenv_if_present`
  读项目根 `.env`（`override=False`，shell export 优先），不再要求用户
  `set -a; source .env`

ABC 后真实跑出的 4 个 bug（commit 见 §12）：

- **示例 DAG 与 README 自相矛盾**：v5 §9 之前 `dags/research_report.json` 6 节点里
  4 个写死 `provider: anthropic`，README 真实模式只指引填 `DEEPSEEK_API_KEY`，
  跑起来 4/6 节点 ANTHROPIC_API_KEY not set。改为默认全 deepseek，多 provider
  演示放节点 system_prompt 注释
- **distill_model 不跟 harness 切换**（P0）：scheduler 实例化 Agent 时只把
  `harness.model` 注入 `chat_model`，`distill_model` 死守默认 `claude-haiku-4-5`；
  非 anthropic provider 跑 distill 时 400 "model not found"。修：scheduler
  把 chat_model 同步注入 distill_model（同 provider 同模型）。
  238 测试全过没挡住——测试全 mock，model name 字符串没人验证；要补端到端
  跑通真 provider 的烟测
- **chromadb telemetry ERROR 噪音**：`Settings(anonymized_telemetry=False)`
  不足以压住——chromadb 0.4.15 + posthog 3.x 的 capture() 签名不兼容仍打
  ERROR。修：`memory_store` 顶部
  `logging.getLogger("chromadb.telemetry.product.posthog").setLevel(CRITICAL)`
- **测试数文档过期**：v5 / README 写 235，实际 ABC + .env 自动加载后已 238，
  本次加 Planner + plan-task + bug fix 后 252。同步

### 14.4 新增哲学

- **节点级 Harness 配置**：每个节点是独立"装备完整"的运行体，不再共享 hardcoded
  `claude-sonnet-4-6` + 默认 system
- **多 provider 自由**：DeepSeek / OpenRouter / Ollama 任选；不要走 Claude Code
  OAuth 复用订阅的歪路（违反 ToS）。**默认推荐 DeepSeek 主用 + OpenRouter 备选**
- **Agent 设计 DAG**：长期方向是 Planner Agent（§9.7）自动从自然语言目标生成
  完整 harness 的 DAG JSON，让"模型/工具/技能"的选择本身也成为 LLM 的决策
- **Sandbox 内 / 外的边界**：5 个内置工具走 sandbox（隔离），MCP server 走主机
  （长连接 + 协议要求）—— 这是 spec v4 §7.2 没明示的取舍

---

## 15. 给后续开发者的实现提示

v4 §13 大部分还有效，v5 补几条：

1. **不要绕过 make_sandbox / make_llm_client 直接 import 具体类**，否则切后端 /
   provider 时业务代码要改。
2. **新加 DAG 节点字段**：先在 `state_store._SCHEMA` 加列 + 在 `_init_schema`
   的兼容性 ALTER TABLE 加一行；`DagNodeRow` + `_row_to_node` + `create_dag_node`
   + `dag_loader` 跟着改。
3. **新加内置工具**：实现 `worker.tools.Tool` 协议 → 加进 `BUILTIN_TOOLS` 表 →
   名字注入 DAG JSON 的 `harness.tools`。schema 字段一次到位的话不用动 DB。
4. **新加 LLM provider**：在 `worker/llm_clients.py` 的 `PROVIDERS` 表加一行就行；
   只要新 provider 是 OpenAI Chat Completions 兼容（绝大多数都是）。
5. **新加 MCP server 类型**：DAG JSON 写 `mcp_servers: [{name, command, args}]`，
   主机要装好对应可执行（如 `npm i -g @modelcontextprotocol/server-github`）。
   不需要写代码。
6. **debugging tool 调用**：scheduler 跑节点时 `loop_res.tool_calls` 含每步
   tool_name + args + result + is_error，需要时把它存进 transcript metadata 或
   新 column（v5 暂未做，仪表盘只展示节点级状态，工具调用轨迹只在日志里）。
7. **token budget 调优**：context_packer 默认 2K，节点产出长 prompt 频繁触发
   裁剪时升到 4K-8K，看 `packed.semantic_dropped_for_budget` 是否还 > 0。
8. **测试两层**：
   - 单元层：每个 store / handler / tool / loop 单独测（不打外网）
   - 集成层：scheduler E2E 测，用 httpx MockTransport 拦 LLM，LocalBackend 跑沙箱

---

## 16. 已知风险登记

| ID | 风险 | 影响 | 缓解 |
|---|---|---|---|
| R-5.1 | MCP server 跑在主机 → 安全责任在 DAG 作者 | 阶段 C+ | 文档明示；将来 §7 CubeSandbox 可考虑把 MCP 也放进去 |
| R-5.2 | OpenAI 兼容 client 不支持 streaming | 长产出体验差 | 单轮 / 多轮都用 max_tokens 控；后续可加 streaming |
| R-5.3 | chromadb 0.4.15 锁定 | 不能用新版功能 | 换 0.5.x 是迁移工程；现状够用 |
| R-5.4 | E2B 按用量计费 | 失控烧钱 | 默认 local；重要 demo 才切 e2b；监控 dashboard.usage |
| R-5.5 | tool_loop max_turns=10 是经验值 | 复杂任务可能不够 | 节点级 harness 留口扩展（暂未暴露到 schema） |
| R-5.6 | `LLM_PROVIDER` 代码 fallback 与 `.env.example` 默认不一致（前者 anthropic，后者 deepseek） | CI / 无 `.env` 环境跑到 anthropic 分支 | 故意保留：CI 全 mock 不真打外网无影响；开发期复制 `.env.example` 后落到 deepseek |
| R-5.7 | ~~DAG JSON 手写门槛~~ | ~~新场景需要人工设计~~ | **已修**：§9.7 Planner Agent + `plan-task` CLI 落地 |
| R-5.8 | Planner LLM 可能稳定性差（输出非 JSON / schema 漂） | plan-task 真实模式偶发失败 | max_retries=2 自动回灌错误重试；耗尽抛 PlannerError；CLI 用 deepseek-chat 实测稳定 |
| R-5.9 | mock 测试无法覆盖 distill_model 字段是否对得上真 provider | 类型问题不会被 pytest 抓出 | 端到端真跑加进 release checklist；spec v5 §14.3 注明 |
| R-6.1 | 节点级 handoff 把上游全部 turn 拼进下游 prompt，token 可能爆 | tool-use loop 后 transcript 1-2K token，多节点链式接力可能撑爆 2K context budget | `turn_range` 截短；token budget 沿用 §8（先丢语义补充，再硬截接力原文）|
| R-6.2 | `handoff.from_node` 在 DAG JSON 写错只在 instantiate 阶段抛错 | 调试体验差 | `parse_dag` 已校验类型；`instantiate_dag` 翻译失败给清晰错误 |

---

## 17. 与 v4 的关系

v4 是**初心档案**：写下"想做什么 + 为什么这样设计"。读 v4 是了解哲学。

v5 是 **ABC 段实施手册**：5 家 LLM provider + tool-use + skills + MCP + Planner，
但每节点 transcript 单轮，下游看不到上游思考过程。

v6 是**当前代码 268 测试全过的形态**：加了「多轮 transcript + 节点级接力」让真协作
有物理基础。读 v6 是知道"现在能做什么 / 怎么用 / 往里加东西从哪改起"。

三份都不要丢；新需求来了写在 v6 注脚，等积累多了再 v7。
