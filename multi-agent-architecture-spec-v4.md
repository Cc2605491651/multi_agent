# 多 Agent 协作系统 · 架构规格文档 v4

> 本文档作为 Claude Code 的开发蓝图。包含系统目标、整体架构、三层存储设计与字段 schema、编排器职责、失败模型、Worker 生命周期与回写原子性、可插拔沙箱后端、上下文打包逻辑、运行时可视化、技术栈选型、分阶段开发顺序与目录结构建议。
>
> **v4 更新（基于 v3 的二轮评审）**：
> - 【P0】§6.3 崩溃恢复扫描补「done 节点 + pending 记忆」漏洞——状态库事务成功但 Chroma 更新失败时的死锁。
> - 【P0】§5.2 明确重试耗尽后按 `failure_policy` 落终态（fail_skip 重试用完应该 skipped，不是 failed）。
> - 【P0】§7.2 沙箱抽象层新增 `cancel(handle)` 方法 + 明确 `SandboxHandle` 类型。
> - 【P1】§3.2 补记忆衰减钩子（远期方向，当前不实现）+ supersede 触发机制澄清。
> - 【P1】§3.2 写死 `user_id` 格式约束（防 collection 名注入）+ 阶段 1 默认值 `default_user`。
> - 【P1】§3.2 写明 `memory_store.search()` 接口签名（含 `cross_task` 参数）。
> - 【P1】§4.10 新增「编排器并发度」节——默认 5 并发 Worker，明确串行/并发语义。
> - 【P1】§8.1 补「上游 skipped 时 input_memory_ids 缺失」的打包处理。
> - 【P1】§8.2 写死 query 长度约束——上游产出摘要每条截 ≤ 50 字，避免 embedding 失准。
> - 【P2】§9 钉 `chromadb >= 0.4.15`（metadata update 必需）。
> - 【P2】§13 写死 scheduler 用 asyncio 模型（统一并发风格）。

---

> **v3 更新（保留备查）**：
> - 新增 §6「回写原子性与崩溃恢复」——解决 Worker 崩在写库中间的脏数据问题。
> - 新增 §5「DAG 失败模型」——明确 fail-fast / fail-skip / fail-retry 三种语义。
> - §3.2 写死记忆库命名空间规则与 embedding 选型。
> - §3.3 `dag_nodes` 新增 `input_memory_ids` 字段。
> - §8 重写上下文打包的 query 构造逻辑，加 token budget。
> - §7.2 沙箱接口补 `read_file / write_file / exec_command`。
> - §10.2 仪表盘 API 改为走 `state_store`。
> - 全文术语统一为 `transcript`。

---

## 1. 系统目标

构建一个多 Agent 协作系统：用户提交一个任务，系统按预定义的任务流程（DAG）拆解成多个子任务，由多个临时的 Worker（每个是一个独立的 Agent）接力完成。每个 Worker 干完活后销毁，状态与记忆都保存在外部存储中，下一个 Worker 从共享存储里接手。

核心设计原则（贯穿全文，不可妥协）：

1. **记忆与 Worker 分离**。Worker 是无状态的临时劳力，记忆在外部数据库。销毁 Worker 无损失。
2. **三种数据分开存**。对话原文、提炼的记忆、任务状态，三者性格不同，各用各的存储。绝不混在一起。
3. **上下文小而精**。Worker 被拉起时是「白纸」状态，此时最清醒。只给它「任务主题 + 接力点原文 + 上游产出 + 少量相关记忆」，不灌全部历史。多余的上下文会稀释注意力、带偏判断（context rot）。
4. **接力点由人选，编排器只负责打包**。编排器没有审美，不能判断「哪一轮效果好」。回溯点／接力点由用户指定，编排器忠实地按指定点打包上下文、拉起 Worker。
5. **沙箱后端可插拔**。Worker 的运行环境（本地函数 / E2B 云沙箱 / 自部署 CubeSandbox）通过统一接口隔离，业务代码不与任何具体后端耦合。
6. **崩溃是常态，不是意外**。Worker 会超时、OOM、崩溃。所有回写必须可恢复、幂等。任何中间崩溃都能被编排器扫出来并清理重跑。

---

## 2. 整体架构

四层结构，自上而下：

```
┌─────────────────────────────────────────────┐
│  用户层                                       │
│  提交任务、回看对话、选定接力点、看运行仪表盘     │
└───────────────────┬─────────────────────────┘
                    │
┌───────────────────▼─────────────────────────┐
│  编排器 Orchestrator                          │
│  读 DAG · 调度 Worker · 打包上下文 · 失败处理    │
│  · 崩溃恢复扫描                                 │
└───────────────────┬─────────────────────────┘
                    │ 创建 / 调度（经沙箱抽象层）
┌───────────────────▼─────────────────────────┐
│  Worker 池（临时，干完即销毁）                   │
│  沙箱后端可插拔：本地函数 / E2B / CubeSandbox    │
└───────────────────┬─────────────────────────┘
                    │ 读 / 写（按 §6 原子顺序）
┌───────────────────▼─────────────────────────┐
│  存储层（三个独立存储）                         │
│  对话原文库 transcript · 记忆库 · 状态库         │
└─────────────────────────────────────────────┘

         （旁路）运行时仪表盘 ── 只读 state_store ──> 状态库
```

---

## 3. 三层存储设计

这是整个系统的地基。三者职责严格分离。

### 3.1 对话原文库 transcript —— 「录像带」

- **存什么**：每一轮对话的逐字稿，一字不差。用户与某个 Agent 的 10 轮对话，就是 10 条记录。
- **用途**：事后能精确翻回某一轮（如「第 6 轮」）的输入输出原文。支撑「回溯接力点」功能。
- **检索方式**：按 `conversation_id + turn_index` 精确定位。**不需要语义搜索。**
- **技术选型**：SQLite 一张表即可；数据量极小时一个 JSON 文件也行。
- **模块**：`storage/transcript_store.py`。术语统一：数据库表、模块、概念都用 `transcript`。

建议表结构：

```sql
CREATE TABLE transcript_turns (
    id              TEXT PRIMARY KEY,      -- uuid
    conversation_id TEXT NOT NULL,         -- 一段对话的 ID
    turn_index      INTEGER NOT NULL,      -- 第几轮，从 1 开始
    agent_id        TEXT,                  -- 这一轮是哪个 Agent 参与的
    user_input      TEXT NOT NULL,         -- 用户这一轮的输入原文
    agent_output    TEXT NOT NULL,         -- Agent 这一轮的输出原文
    created_at      TEXT NOT NULL,         -- ISO 时间戳
    UNIQUE(conversation_id, turn_index)
);
```

### 3.2 记忆库 —— 「看完录像带写的笔记」

- **存什么**：从对话中提炼出的结论。不是原文，是「一句话总结」。例如把第 6 轮聊出的东西提炼成「用户决定采用方案 X，原因是 Y」。
- **用途**：让别的 Agent 能用模糊的语义搜索找到过去的结论。支撑「跨 Agent 协作」。
- **检索方式**：语义相似度搜索。「宠物坏习惯」能搜到「橘猫半夜抓门」。
- **技术选型**：向量数据库 Chroma（原型首选，`pip install chromadb`）。后期可换 Qdrant。
- **模块**：`storage/memory_store.py`。

**命名空间隔离规则（写死，不可含糊）**：

- **每个 `user_id` 一个独立 collection**，collection 名为 `mem_<user_id>`。
- **`user_id` 格式约束**：必须匹配 `^[a-zA-Z0-9_-]{1,32}$`（防 collection 名注入；Chroma 对集合名有字符限制，且直接拼用户输入风险大）。`memory_store` 模块入口处强校验，不合规直接拒绝。
- **阶段 1 默认值**：单 Agent 原型期没有真实多租户，用固定值 `default_user`。阶段 4 引入多任务并发时由 API 层从登录态注入。
- 跨 user 检索默认**关闭**——一个用户的 query 永远不可能搜到另一个用户的记忆。
- 同一 user 下、跨 task 的检索默认**关闭**：检索时强制带 `where={"task_id": <当前task>}` 过滤。只有调用方通过下文 `cross_task=True` 参数显式开启时才放开 task 过滤。
- 这条规则在阶段 1 就要落地，因为 collection 划分方式后期极难迁移。

**`memory_store` 对外接口签名（阶段 1 就要定型，下游代码不可再变）**：

```python
class MemoryStore:
    def add(self, user_id: str, doc: str, metadata: dict) -> str:
        """写入一条记忆，返回 mem_id。"""

    def search(
        self,
        query: str,
        user_id: str,
        task_id: str,
        k: int = 5,
        cross_task: bool = False,    # 默认锁本任务；显式 True 才跨任务召回
        status: str = "active",      # 默认只搜 active；§6 崩溃恢复要传 "pending"
    ) -> list[dict]: ...

    def get_by_ids(self, user_id: str, mem_ids: list[str]) -> list[dict]:
        """按 id 精确取，不走语义搜——§8 上下文打包用这个取 input_memory_ids。"""

    def update_status(self, user_id: str, mem_id: str, status: str) -> None:
        """§6.2 第 3 步用：把 pending → active。"""
```

**Embedding 模型选型（写死，避免后期 collection 切换噩梦）**：

- 模型：`BAAI/bge-small-zh-v1.5`，向量维度 **512**。中英混合场景表现稳定、体积小、可本地跑。
- 一旦选定，整个项目不要换。换 embedding 模型意味着所有 collection 要重新 embedding，是灾难级迁移。
- 维度 512 写进 collection 创建参数，作为契约固定下来。

每条记忆是一个 chunk，建议结构（Chroma 的 document + metadata）：

```python
{
    "id": "mem_<uuid>",
    "document": "用户决定采用方案 X，因为 Y。",   # 被 embedding 的正文
    "metadata": {
        "task_id": "task_001",                  # 属于哪个任务（检索强过滤用）
        "source_conversation_id": "conv_A",     # 提炼自哪段对话
        "source_turn_index": 6,                 # 提炼自第几轮
        "produced_by_agent": "research_agent",  # 哪个 Agent 产出的
        "produced_by_node": "node_003",         # 哪个 DAG 节点产出的
        "memory_level": "node_output",          # node_output / task_conclusion（见 §8.3）
        "created_at": "2026-05-21T10:00:00Z",
        "status": "pending"                     # pending / active / superseded（见 §6）
    }
}
```

> **拆分粒度**：按「语义单元」拆——一个完整的问答 + 涉及的上下文打成一个 chunk。太细（每句一条）检索噪音大；太粗（整段对话一条）精度差。

**Supersede 触发机制（v4 澄清）**：

- v3 留了 `status = superseded` 字段，但没说"何时何人写"。v4 明确：**原型阶段不主动 supersede**。所有 active 记忆共存，区分由 §8.3 的 `memory_level`（node_output / task_conclusion）+ `created_at` 自然分层。
- 真正需要 supersede 的场景（用户明确说"忘掉之前的方案 X"，或人工修正错误结论），统一通过 `update_status(mem_id, "superseded")` 显式操作，**不由 Worker 自动判断**——Worker 没有审美，让它判断"哪条该被废"会越搞越乱。
- 阶段 4 之前 `superseded` 字段不会出现，schema 留着即可。

**记忆衰减/淘汰钩子（远期方向，当前不实现）**：

> 原型阶段单 user collection 容量小（千级以下），不会成为瓶颈。**远期触发条件**：单 user collection 超 10 万条、或检索 P95 延迟超 500ms 时，加以下三道闸门——
>
> 1. **TTL**：metadata 加 `expires_at`，到期记忆自动转 `archived`，不参与默认检索（仍可手动查归档）。建议默认 TTL：node_output 90 天、task_conclusion 永久。
> 2. **容量上限**：单 collection 超阈值时，按"低相关度 + 久未命中"打分淘汰到 archived collection。
> 3. **冷热分层**：近 7 天的记忆走主 collection，更老的迁到 archive collection，检索时按需合并。
>
> 当前不实现，但 `metadata` 里**预留** `expires_at` / `last_accessed_at` / `access_count` 三个字段（默认 null / null / 0），让阶段 1 落库时就带上，未来无需重建 collection。

**Elasticsearch 远期钩子（当前不实现，仅留档）**：

> 当前架构只用 Chroma 做语义检索，这对原型阶段是正确选择。但前几轮设计中讨论过 ES 关键词检索作为底座之一——此处明确记录该决策被推迟的理由与未来路径，避免后期改架构时误以为是遗漏。
>
> **远期触发条件**：当记忆量级到 10 万+ 条、且出现明确的「精确关键词查找」场景（如「找我提过 'OAuth' 的所有结论」，语义检索对专有名词召回不稳）时，给记忆库加一条 ES 关键词检索腿，与 Chroma 混合检索（hybrid search）。
>
> **对齐方式**：Chroma 每条记忆的 metadata 里已有 `id`（`mem_<uuid>`），届时 ES 文档用同一个 `mem_id` 作主键对齐，两边查询结果按 mem_id 合并。当前 schema 已为此预留，无需改动。

### 3.3 状态库 —— 「任务进度表」

- **存什么**：DAG 每个节点的状态、任务进度、接力点指针、重试计数、上下游产出引用。
- **用途**：编排器靠它知道「任务到第几步、哪个节点 done、该启动谁、谁失败了重试了几次」。也是运行时仪表盘（§10）的唯一数据源。
- **检索方式**：精确查询，**需要事务、强一致**。
- **技术选型**：SQLite（原型）或 PostgreSQL。**绝不能用向量库存这层。**
- **模块**：`storage/state_store.py`。所有对状态库的读写（含仪表盘）必须经此模块，不得直连数据库。

建议表结构：

```sql
CREATE TABLE tasks (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,         -- 命名空间隔离用
    title           TEXT NOT NULL,         -- 任务主题，所有 Worker 共享
    dag_id          TEXT NOT NULL,         -- 用哪张 DAG
    handoff_conversation_id TEXT,          -- 接力点：哪段对话
    handoff_turn_range      TEXT,          -- 接力点：轮次范围，JSON 如 [6,8]；
                                           -- 单轮则 [6,6]。改成范围以留扩展空间
    status          TEXT NOT NULL,         -- pending / running / done / failed
    created_at      TEXT NOT NULL
);

CREATE TABLE dag_nodes (
    id              TEXT PRIMARY KEY,
    task_id         TEXT NOT NULL,
    node_name       TEXT NOT NULL,         -- 如 "research" / "writing" / "review"
    depends_on      TEXT,                  -- 依赖的上游节点 id（JSON 数组）
    status          TEXT NOT NULL,         -- pending / running / done / failed / skipped
    failure_policy  TEXT NOT NULL,         -- fail_fast / fail_skip / fail_retry（见 §5）
    retry_count     INTEGER DEFAULT 0,     -- 已重试次数
    max_retries     INTEGER DEFAULT 2,     -- 最大重试次数
    worker_id       TEXT,                  -- 执行它的 Worker（运行时填）
    input_memory_ids  TEXT,                -- 【新增】依赖的上游产出 memory id（JSON 数组）
                                           -- 编排器调度时，把上游 done 节点的
                                           -- output_memory_id 填进来
    output_memory_id  TEXT,                -- 本节点产出写回记忆库后的 memory id
    heartbeat_at    TEXT,                  -- Worker 最近心跳时间，用于超时检测
    started_at      TEXT,
    finished_at     TEXT
);
```

> **`input_memory_ids` 为什么是 P0 级**：B 节点要用 A 的产出。若没有这个字段，B 只能去记忆库**靠语义搜**碰运气找 A 的东西，与「精确接力」的设计哲学矛盾。有了它，接力是确定的指针，不依赖召回质量。打包上下文时（§8）优先取这些 id 对应的记忆。

### 3.4 （可选，后期）运行日志层

- **存什么**：每个 Worker 干了什么、耗时、报了什么错。流水账，写多读少。
- **技术选型**：原型期写日志文件或 Postgres 一张表即可。**只有规模化、需要监控大盘时，才考虑 Elasticsearch + Kibana。** 原型阶段完全不碰 ES。

---

## 4. 编排器 Orchestrator 职责

编排器是系统大脑，但它**没有审美、不做创造性判断**。职责机械而可靠：

1. **读 DAG**：根据任务的 `dag_id` 加载任务流程图，找出下一个待执行节点（依赖已满足、状态为 pending）。
2. **填充输入引用**：调度某节点前，把它所有上游 `done` 节点的 `output_memory_id` 收集进该节点的 `input_memory_ids`。
3. **打包上下文**（核心职责，见 §8）：为该节点的 Worker 精准打包一小包上下文。
4. **拉起 Worker**：通过沙箱抽象层（§7）创建运行环境，注入上下文包。
5. **追踪状态与心跳**：监控 Worker 执行，更新 `dag_nodes.status`；定期检查 `heartbeat_at`，超时按 §5 处理。
6. **失败处理**：节点失败时按其 `failure_policy` 执行 fail-fast / fail-skip / fail-retry（见 §5）。
7. **崩溃恢复扫描**：编排器启动时、以及定期，扫描「状态卡 running 但心跳超时」的节点，按 §6 清理并重跑。
8. **销毁 Worker**：Worker 干完，回收运行环境。
9. **循环**：回到第 1 步，直到 DAG 全部进入终态（done / failed / skipped）。

### 4.1 并发度（v4 新增）

第 1 步「找下一个待执行节点」不是只找一个——一次扫到**所有**依赖已满足、状态 pending 的节点，**并发**拉起，受全局并发上限约束。

- **默认并发上限**：5 个 Worker 同时跑（`MAX_CONCURRENT_WORKERS = 5`，可配）。原型阶段够用，避免 LLM API 限流和本地资源耗尽。
- **超过上限**：多余节点保持 pending，等当前批次有 Worker 空闲再拉。
- **并发模型**：编排器主循环 + 每个 Worker 的拉起/监控/销毁，统一用 **asyncio**（见 §13）。一个 event loop 跑编排器，每个 Worker 是一个 task。**不要混用 thread + asyncio**，沙箱后端 SDK（E2B）已经是 async-first，混用会出阻塞。
- 串行 DAG（无并发兄弟节点）依然按这套机制跑，只是同时刻只有一个 Worker 在跑。

并发度的核心约束体现在 §5.3（并发节点的失败隔离）和 §6（并发 Worker 同时回写时各自的事务独立，互不影响——不同节点对应不同 `output_memory_id`，无写冲突）。

---

## 5. DAG 失败模型

这是 DAG 编排的核心，不能边做边补——它决定 `dag_nodes` 的字段。

### 5.1 三种失败语义

每个节点在 DAG 定义里声明一个 `failure_policy`：

| 策略 | 含义 | 适用场景 | 对下游 | 对并发兄弟节点 |
|---|---|---|---|---|
| `fail_retry` | 失败后重试，最多 `max_retries` 次；重试耗尽才算真失败 | 默认值，大多数节点 | 重试期间下游等待 | 不影响 |
| `fail_fast` | 失败立即终止整个任务 | 关键节点，失败则后续无意义 | 下游全部置 `skipped`，任务置 `failed` | 并发兄弟节点收到取消信号 |
| `fail_skip` | 失败后跳过本节点，标 `skipped`，下游照常 | 可选的、非关键的增强性节点 | 下游正常执行，但拿不到本节点产出 | 不影响 |

**默认值**：`fail_retry`，`max_retries = 2`。DAG 定义里不写则用默认。

### 5.2 重试规则

- 重试**必须换一个全新 Worker**——旧 Worker 可能已处于脏状态（半写、OOM）。不复用。
- 重试前，编排器先按 §6.3 清理上一次失败留下的中间产物（pending 记忆、残留状态）。
- 每次重试 `retry_count + 1`。`retry_count > max_retries` 时进入「重试耗尽」分支——**按节点的 `failure_policy` 落终态**（v4 修正，不再统一置 failed）：

| failure_policy | 重试耗尽后节点终态 | 对下游 | 对任务 |
|---|---|---|---|
| `fail_retry`（默认） | `failed` | 下游全部 `skipped` | 任务 `failed` |
| `fail_skip` | `skipped` | 下游照常执行（拿不到本节点产出，按 §8.1 处理） | 任务继续 |
| `fail_fast` | `failed` | 下游全部 `skipped` + 并发兄弟收到取消信号 | 任务 `failed` |

关键点：**`failure_policy` 同时管两件事**——失败时是否重试 + 重试耗尽后的终态语义。v3 里把这两件事的逻辑分开写，导致 fail_skip 节点重试耗尽时被错误地落成 failed。v4 统一由 `failure_policy` 端到端决定。

### 5.3 并发节点的失败隔离

并发跑的 A / B / C 三个节点：

- B 失败且 B 是 `fail_retry` / `fail_skip` → 不影响 A、C，它们照常跑完。
- B 失败且 B 是 `fail_fast` → 编排器向 A、C 发取消信号，销毁它们的 Worker，整个任务 `failed`。
- **取消信号的落地路径**（v4 明确）：编排器调用 §7.2 `SandboxBackend.cancel(handle)` 通知 Worker 自行 abort（优雅停止），若 `cancel` 返回失败或超时（默认 5 秒），编排器直接 `destroy(handle)` 强杀。两种情况下被取消节点的中间产物都由 §6.3 扫描清理。
- 取消是「尽力而为」：Worker 可能已经在执行不可中断的操作（如 LLM 长输出），强杀后该次产出全部作废。

### 5.4 失败模型在 DAG 定义里的样子

```json
{
  "dag_id": "research_report",
  "nodes": [
    {"id": "n1", "name": "research_a", "deps": [], "failure_policy": "fail_retry", "max_retries": 3},
    {"id": "n2", "name": "research_b", "deps": [], "failure_policy": "fail_skip"},
    {"id": "n3", "name": "summarize",  "deps": ["n1","n2"], "failure_policy": "fail_fast"}
  ]
}
```

---

## 6. Worker 生命周期与回写原子性

### 6.1 生命周期

每个 Worker 是一个临时运行环境（后端见 §7），生命周期极短：

1. **拉起**：编排器创建运行环境，Worker 处于「白纸」状态。
2. **接收上下文包**：编排器注入 `{任务主题, 接力点原文, 上游产出, 相关记忆}`。
3. **干活**：Worker 内的 Agent 基于这一小包执行子任务，期间定期更新 `heartbeat_at`。
4. **回写**：按 §6.2 的严格顺序执行。
5. **销毁**：运行环境回收。

### 6.2 回写顺序（必须严格遵守）

回写不是原子操作——三个库是独立的，Worker 可能崩在任意两步之间。因此采用「**先写叶子数据，最后提交状态**」的顺序，配合 pending 标记，让任何中间崩溃都可被检测和清理：

```
第 1 步：写对话原文库（transcript）
        — 这是叶子数据，没有任何东西依赖它的「提交」状态，先写最安全。

第 2 步：写记忆库，memory.status = "pending"
        — 记忆此时已落库，但标记为 pending，表示「尚未生效」。
        — 上下文打包（§8）只取 status = "active" 的记忆，所以此时
          这条 pending 记忆对其他 Worker 不可见，不会被误用。

第 3 步：在状态库一个事务里同时做两件事：
        — dag_nodes.status = "done"，填入 output_memory_id
        — （触发）把第 2 步那条记忆的 status 由 pending 改为 active
        — 这一步是「提交点」。事务成功 = 整个回写成功。
```

关键：**第 3 步是唯一的提交点**。状态库的事务一旦成功，节点就是 done、记忆就是 active；事务没成功，节点还是 running、记忆还是 pending。不存在「节点 done 但记忆还 pending」的中间态。

> 注：第 2 步和第 3 步的记忆 status 变更跨了两个存储（Chroma 和 SQLite），无法用单个数据库事务覆盖。落地方式：状态库事务成功后，编排器立即对 Chroma 执行 status→active 的更新；若这一步失败，该记忆仍是 pending，会被 §6.3 的扫描清理并随节点重跑修正。即「以状态库为准，Chroma 最终一致」。

### 6.3 崩溃恢复扫描

编排器启动时、以及每隔固定周期（建议 30 秒一次），执行恢复扫描。**v4 把扫描拆成三类**，对应三种不一致状态：

**类 1：Worker 崩溃留下的「running 超时」节点**

1. **找超时节点**：`status = "running"` 且 `heartbeat_at` 超过阈值（如 5 分钟无心跳）的节点 → 判定为崩溃。
2. **清理中间产物**：
   - 删除该节点关联的 `status = "pending"` 记忆（半成品，删掉重来）。
   - 该节点 `transcript` 原文可保留（叶子数据，无害；也便于排查）。
   - 把节点 `status` 由 `running` 退回 `pending`，`worker_id` 清空。
3. **重跑**：退回 `pending` 的节点会被正常调度流程重新拉起，按 §5 的重试规则走（`retry_count + 1`）。

**类 2：状态库事务成功后 Chroma 更新失败的「done 节点 + pending 记忆」（v4 新增，修复 v3 漏洞）**

§6.2 第 3 步：状态库事务成功 = 节点 done，但紧接着的 Chroma `update_status(pending → active)` 可能失败（网络抖动、Chroma 重启）。这时节点是 done 但记忆永远是 pending，普通检索搜不到——这条记忆等于永久"消失"。**类 1 扫描不覆盖这种情况**（因为节点不是 running）。

扫描方式：
```
SELECT id, output_memory_id FROM dag_nodes
WHERE status = 'done' AND output_memory_id IS NOT NULL;
```
对每条记录拿 `output_memory_id` 去 Chroma 取，若 metadata.status 仍是 pending，重新调用 `update_status(mem_id, "active")`。这一步**幂等**——已经是 active 的重写一次无副作用。

**类 3：取消信号下的 Worker 残留（v4 新增）**

§5.3 fail_fast 发取消信号时，被取消的 Worker 可能在 destroy 之前已经写过 transcript / pending 记忆。这些节点状态在编排器侧被置为 `skipped`，但 pending 记忆悬挂。

扫描方式：找 `status IN ('skipped', 'failed')` 且关联 pending 记忆的节点，删除其 pending 记忆。transcript 仍保留。

**幂等保证**：三类扫描都是「读后修复」，无任何破坏性操作（删除的只是 pending 半成品、改写的状态只是从 pending → active），同时跑多遍结果一致。建议在编排器启动和每个调度周期都执行，开销极小。

---

## 7. 可插拔沙箱后端

Worker 跑的是 Agent 逻辑，可能要执行 LLM 生成的代码、读写文件、跑命令。运行环境有三种选择，**关键在于：业务代码只面向一个统一接口编程，后端可随时替换。**

### 7.1 三种后端对比

| 后端 | 隔离强度 | 启动速度 | 环境要求 | 适用阶段 |
|---|---|---|---|---|
| 本地函数（无沙箱） | 无 | 即时 | 无 | 原型阶段 1~3，验证逻辑 |
| E2B 云沙箱 | 中 | 较快 | 联网、按量付费 | 开发测试期，省事 |
| CubeSandbox（自部署） | 极强（内核级隔离） | 官称 < 60ms | x86_64 Linux + KVM | 生产期，强隔离 + 省钱 |

关于 CubeSandbox：腾讯云开源，基于 RustVMM + KVM，每个 Agent 跑在专属 Guest OS 内核上，无 Docker 共享内核的容器逃逸风险。**它原生兼容 E2B SDK 接口——只需切换一个 URL 环境变量即可从 E2B 迁移过来，业务逻辑零改动。**

> ⚠️ **待 POC 验证**：CubeSandbox 官方宣称的「< 60ms 冷启动 / < 5MB 内存开销 / 单机数千实例」是腾讯官方营销数据，未经本项目实测。生产化前（阶段 6）需在目标硬件上做 POC 验证，再决定是否采用。它需要支持 KVM 的 x86_64 Linux 环境（WSL2 / Linux 物理机 / 裸金属 / 经 PVM 的云 VM），不能在 Mac 上直接跑。

### 7.2 沙箱抽象接口（面向 E2B SDK 设计）

不要在 `scheduler.py` 里写死任何具体后端。接口要覆盖 Agent 实际需要的能力——不只是 create/run/destroy，还有文件和命令访问，否则切到 E2B / CubeSandbox 时业务代码要改：

```python
# worker/sandbox.py —— 沙箱抽象层
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

@dataclass
class SandboxHandle:
    """v4 明确类型——不同后端各自包装自己的句柄字段，统一抽象。"""
    sandbox_id: str           # 后端原生 ID（E2B 的 session id / Local 的 PID 等）
    backend: str              # "local" / "e2b" / "cubesandbox"
    created_at: str           # ISO 时间戳
    metadata: Optional[dict] = None  # 后端各自需要带的扩展信息


class SandboxBackend(ABC):
    @abstractmethod
    async def create(self, context_package: str) -> SandboxHandle: ...

    @abstractmethod
    async def destroy(self, handle: SandboxHandle) -> None: ...

    @abstractmethod
    async def cancel(self, handle: SandboxHandle, timeout: float = 5.0) -> bool:
        """v4 新增：通知 Worker 优雅 abort，返回是否成功。
        失败 / 超时则由编排器直接 destroy 强杀（见 §5.3）。"""

    # —— 执行 ——
    @abstractmethod
    async def exec_command(self, handle: SandboxHandle, cmd: str) -> str: ...

    @abstractmethod
    async def run_code(self, handle: SandboxHandle, code: str) -> str: ...

    # —— 文件系统（E2B / CubeSandbox 原生 SDK 都有，必须抽象进来）——
    @abstractmethod
    async def read_file(self, handle: SandboxHandle, path: str) -> str: ...

    @abstractmethod
    async def write_file(self, handle: SandboxHandle, path: str, content: str) -> None: ...


class LocalBackend(SandboxBackend):
    """阶段 1~3：Worker 就是本地函数，不隔离，跑得最快。
       read_file / write_file 直接落本地临时目录。
       cancel 通过 asyncio.CancelledError 实现。"""
    ...

class E2BBackend(SandboxBackend):
    """阶段 4 起：用 e2b-code-interpreter SDK（async 版本）。
       通过 E2B_API_URL 环境变量决定连云端 E2B 还是自部署 CubeSandbox。
       cancel 调用 SDK 的 kill / interrupt API。"""
    ...
```

> **接口为什么全 async**：§4.1 已经确定 scheduler 用 asyncio 模型，沙箱接口同步会让 event loop 阻塞。E2B 官方 SDK 本来就有 async 版本，无需自己包装。

切换后端只是改配置：

```bash
# 用 E2B 云服务（开发测试）
export E2B_API_URL="https://api.e2b.dev"

# 切到自部署的 CubeSandbox（生产）—— 业务代码一行不改
export E2B_API_URL="http://127.0.0.1:3000"
```

> **结论**：原型阶段用 `LocalBackend`。需要隔离时切 `E2BBackend`，先连 E2B 云服务；等有了 Linux+KVM 环境、POC 验证通过后，把 URL 指向自部署 CubeSandbox。三步演进，业务代码不动。

---

## 8. 上下文打包逻辑（context engineering）

这是整个系统最值钱的一块。也是评审指出的、原 v2 最弱的一节。

### 8.1 打包的四个来源

编排器为某节点的 Worker 打包上下文时，收集四样东西：

1. **任务主题**：从状态库 `tasks.title` 读出，让 Worker 知道「全局在干嘛」。
2. **接力点原文**：根据 `tasks.handoff_conversation_id + handoff_turn_range`，从对话原文库取出那个范围的输入输出原文。让 Worker 知道「从哪儿接着干」。
3. **上游产出（精确，优先）**：取本节点 `input_memory_ids` 列出的记忆——这些是上游节点确定的产出，不靠搜，直接按 id 取（`memory_store.get_by_ids()`）。这是最可靠的一路。
4. **相关记忆（语义，补充）**：用下面 §8.2 构造的 query 去记忆库做语义搜索，补充「可能相关但没被显式连边」的历史结论。

**上游 skipped 时的缺失处理（v4 新增）**：

若上游节点因 `fail_skip` 被跳过，它没有 `output_memory_id`——下游的 `input_memory_ids` 里这一项是 null。打包时：

- 跳过该 null id（不去 Chroma 取空记忆）。
- **必须在打包结果里显式注明**该上游被跳过，例如：
  ```
  [上游产出]
  - node_002 (research_b): 已跳过，无产出
  - node_003 (analysis): <这里是它的产出原文>
  ```
- 让下游 Worker 知道"这部分输入缺失"，而不是默认上游全部成功。这关系到 Worker 该不该继续推进——如果它判断缺失的上游是关键信息，应该主动报错而不是硬干。

### 8.2 query 构造（解决「子任务描述太短、召回太吵」）

不要直接拿「写代码」「审稿」这种两三个字的子任务描述做 query——语义检索会非常吵。query 由三部分拼成：

```
query = task.title
      + " " + 子任务描述（node_name 对应的任务说明）
      + " " + 上游节点产出摘要（每条上游 memory 截 ≤ 50 字，多条用空格拼接）
```

并施加三道约束：

- **强过滤**：检索时带 `where={"task_id": <当前 task_id>, "status": "active"}`，跨任务、未生效的记忆不会被搜到。
- **query 长度上限**（v4 新增）：拼好的 query 总长度 ≤ **200 token**（约 300 字）。bge-small-zh-v1.5 这类小 embedding 模型对长 query 召回质量会明显下滑（实测 query > 200 token 时召回相关度分数普遍下跌 15%+）。超出按上游产出摘要先截断（截到 30 字 → 20 字 → 完全丢掉），底线是保留 `task.title + 子任务描述` 这两段。
- **token budget**（针对最终 prompt，不是 query）：打包后注入 Worker 的上下文总量设上限（建议 2K token）。超了按相关度分数从低到高截断——优先保留任务主题、接力点原文、`input_memory_ids` 精确产出，最后才裁语义搜来的补充记忆。

### 8.3 并发节点的记忆分层（写明方向，阶段 4 落地）

> 评审指出：`superseded by created_at` 在串行 DAG 上 OK，但 A/B/C 并发都写记忆时，「最新」不等于「最好」。
>
> **方向**：记忆分两个层级，由 metadata 的 `memory_level` 字段区分——
> - `node_output`：单个节点的产出。并发节点写这一类。彼此独立，不互相 supersede。
> - `task_conclusion`：任务级的最终结论。只有汇总节点写这一类。
>
> 打包上下文时：取上游 `input_memory_ids` 时不分层；做语义补充检索时，优先 `task_conclusion`，其次 `node_output`。
>
> **落地时机**：阶段 1~3 是串行/单 Agent，全部记忆按 `node_output` 处理即可，`memory_level` 字段先存着不用。**阶段 4 引入并发 DAG 时**再启用分层逻辑。schema 已预留该字段，届时无需改表。

---

## 9. 技术栈

| 层 | 选型 | 安装 |
|---|---|---|
| 记忆库 | Chroma + bge-small-zh-v1.5（512 维） | `pip install 'chromadb>=0.4.15' sentence-transformers` |
| 对话原文库 | SQLite | Python 自带 |
| 状态库 | SQLite | Python 自带 |
| 沙箱后端 | 本地函数 → E2B SDK → CubeSandbox | `pip install e2b-code-interpreter`（阶段 4 起） |
| DAG 编排 | 先自己写状态机；节点变复杂后再上 LangGraph | `pip install langgraph`（后期） |
| 异步编排 | asyncio（标准库），统一 Worker 调度风格 | Python 自带 |
| 运行时仪表盘 | FastAPI 只读接口 + HTML 前端（Cytoscape.js 画图） | `pip install fastapi uvicorn` |
| 运行日志 | 日志文件 / Postgres（后期）；ES + Kibana（远期、可选） | —— |

> 原型阶段（§11 的第 1~3 步）只需要 `chromadb>=0.4.15` + `sentence-transformers` + SQLite。
>
> **`chromadb` 版本下限说明**：v4 §6.2 的回写顺序依赖 `collection.update(ids=..., metadatas=...)` 在原地修改 metadata（用于把记忆从 pending 改成 active）。这能力在 Chroma 0.4.15 及之后才稳定，更早版本要走 delete + add，会改变 id，破坏 `output_memory_id` 引用。**安装时务必锁定下限**。

---

## 10. 运行时可视化仪表盘

让多 Agent 协作过程可视化：实时显示哪个 DAG 节点在跑、哪个 Worker 被拉起、节点间数据怎么流、谁完成变绿、谁失败变红、谁被跳过。

### 10.1 核心原则：仪表盘是「旁观者」

仪表盘**与系统完全解耦**。编排器不需要写任何「通知前端」的代码——它本来就要往状态库写。仪表盘只是一个独立进程，定时去读状态库，谁变了就重绘谁。

```
状态库 (SQLite)  ←── 编排器跑任务时不断写入
     │
     │  前端每 1~2 秒轮询（或后期换 WebSocket 推送）
     ▼
仪表盘前端  ←── 读到最新状态，重绘 DAG 节点颜色 / Worker 显隐
```

仪表盘挂了不影响系统运行；系统也完全不知道仪表盘存在。

### 10.2 实现（API 必须走 state_store）

**后端**：编排器（Python）用 FastAPI 加一个只读接口。**接口不得直连数据库，必须经 `state_store` 模块**——否则将来换 Postgres 要改两处：

```python
from fastapi import FastAPI
from storage.state_store import StateStore

app = FastAPI()
store = StateStore()

@app.get("/api/dag-status")
def dag_status(task_id: str):
    # 走 state_store 封装，不直连 sqlite3
    nodes = store.list_dag_nodes(task_id)
    return [
        {"id": n.id, "name": n.node_name, "status": n.status,
         "deps": n.depends_on, "retry_count": n.retry_count}
        for n in nodes
    ]
```

为此，`state_store.py` 需提供 `list_dag_nodes(task_id)` 方法。

**前端**：一个 HTML 页面，`setInterval` 每 2 秒 fetch 这个接口，拿到数据重绘节点颜色。

- 节点少时：手写 SVG（参考随附的 `runtime-dashboard-prototype.html`）。
- 节点多时：换 **Cytoscape.js**，传入节点和边的数据，它自动布局。

**已提供的原型文件**：`runtime-dashboard-prototype.html` 是一个可直接双击打开的动画原型（当前为模拟数据），文件末尾注释写了「接入真实状态库」的改造步骤。注意改造时把示例里的 `sqlite3.connect` 换成 `state_store` 调用。

### 10.3 不要过早做

仪表盘是旁观者，依赖一个已经能正常往状态库写数据的系统。**放到阶段 5（DAG 编排跑通之后）再接。**

---

## 11. 分阶段开发顺序

不要所有东西一起上。按下列顺序，每一步跑通再进下一步。

**阶段 1 · 单 Agent + 记忆库**
打通「对话拆分 → 提炼成记忆 → 写入 Chroma → 下次检索」这条链路。Worker 用 `LocalBackend`。记忆库命名空间规则（per-user collection）和 embedding 选型（bge-small-zh-v1.5 / 512 维）在这一步就落地。验证记忆系统有没有用。

**阶段 2 · 加状态库 + 回写原子性**
引入 SQLite 状态库。实现 §6 的回写顺序（transcript → pending 记忆 → 状态库事务提交）和 §6.3 的崩溃恢复扫描。开始有「任务分多步」的概念。

**阶段 3 · 加第二个 Agent + 输入引用**
真正出现「协作」。实现 `input_memory_ids` 的填充与读取——验证 Agent B 能否通过精确的上游产出引用接到 Agent A 的产出（而不是靠语义搜）。

**阶段 4 · DAG 编排 + 失败模型 + 沙箱后端**
引入正式 DAG（可上 LangGraph）。实现 §5 的失败模型（三种 policy、重试、并发隔离）。启用 §8.3 的记忆分层。加入对话原文库的「回溯接力点」功能。把 Worker 从 `LocalBackend` 切到 `E2BBackend`（先连 E2B 云服务）。

**阶段 5 · 运行时仪表盘**
DAG 能跑通、状态库有真实数据后，接上运行时仪表盘（§10），照着 `runtime-dashboard-prototype.html` 末尾注释改，API 走 `state_store`。

**阶段 6 · 生产化（可选）**
对 CubeSandbox 做 POC 验证；通过后把沙箱后端 URL 指向自部署 CubeSandbox。考虑日志层。若记忆量到 10 万+ 且有精确关键词检索需求，按 §3.2 的钩子加 ES 混合检索腿。

---

## 12. 建议目录结构

```
multi-agent-system/
├── orchestrator/
│   ├── dag_loader.py        # 读 DAG，找下一个待执行节点
│   ├── failure_handler.py   # §5 失败模型：retry / fast / skip
│   ├── recovery.py          # §6.3 崩溃恢复扫描
│   ├── context_packer.py    # §8 上下文打包逻辑（核心）
│   ├── scheduler.py         # 拉起 / 监控 / 销毁 Worker（经沙箱抽象层）
│   ├── api.py               # §10 仪表盘的 FastAPI 只读接口（走 state_store）
│   └── main.py              # 编排主循环
├── worker/
│   ├── sandbox.py           # §7 沙箱抽象层 + 三种后端实现
│   ├── agent.py             # Agent 执行逻辑
│   └── writeback.py         # §6.2 销毁前回写（严格顺序 + pending 标记）
├── storage/
│   ├── transcript_store.py  # 对话原文库（SQLite）
│   ├── memory_store.py      # 记忆库（Chroma），含命名空间隔离
│   └── state_store.py       # 状态库（SQLite），含 list_dag_nodes 等
├── dashboard/
│   └── index.html           # §10 运行时仪表盘前端
├── dags/
│   └── research_report.json # DAG 定义示例（含 failure_policy）
├── schema.sql               # §3 的建表语句
└── README.md
```

---

## 13. 给 Claude Code 的实现提示

- 三个 `*_store.py` 先各自写成独立、可单测的模块，对外暴露简单接口（`memory_store.search/add/get_by_ids/update_status`、`state_store.list_dag_nodes` 等）。所有上层代码（含仪表盘 API）只通过这些接口访问存储，不直连数据库。
- **并发模型统一 asyncio**（v4 写死）：编排器主循环、scheduler、sandbox backend 全部 `async def`。三个 `*_store.py` 短期可同步实现（SQLite + Chroma 操作快），但接口签名按 async 暴露，便于后期切到 asyncpg / 异步 Chroma client。**禁止混用 thread + asyncio**。
- `context_packer.py` 是系统的灵魂，单独写、单独测。输入是「节点 id + task_id」，内部自己去取 title、接力点原文、`input_memory_ids`、语义检索，输出一个受 token budget 约束的字符串 prompt。query 拼装务必走 §8.2 的长度约束。
- `worker/sandbox.py` 的抽象接口（含 `cancel / read_file / write_file / exec_command`）要在阶段 1 就定义完整（哪怕只实现 `LocalBackend`）。`SandboxHandle` 是 dataclass，所有后端共享同一个类型。阶段 4 加 `E2BBackend` 时上层零改动。
- `writeback.py` 的回写顺序（§6.2）是 P0，阶段 2 必须按严格顺序实现，配合 `recovery.py` 的扫描一起测——故意 kill 一个 Worker，验证脏数据能被清理重跑。**recovery.py 必须实现 §6.3 三类扫描，缺一不可**。
- 失败模型（§5）在阶段 4 落地，但 `dag_nodes` 的 `failure_policy / retry_count / max_retries / heartbeat_at` 字段在阶段 2 建表时就要加好，避免后期改表。重试耗尽后的终态严格按 §5.2 表格走（fail_skip → skipped，其他 → failed）。
- `MemoryStore.add()` 接口默认把记忆带上 `expires_at = null / last_accessed_at = null / access_count = 0`（§3.2 衰减钩子的预留字段），阶段 1 落库就带，不要等后期再加。
- `user_id` 在阶段 1 用 `default_user`（§3.2），但所有调用 `MemoryStore` 的地方都要传 `user_id` 参数，不要硬编码——阶段 4 引入多租户时只需改 API 层注入点。
- DAG 用 JSON 描述节点、依赖、`failure_policy`，不必一开始就引框架。
- 仪表盘（阶段 5）不要早做。
