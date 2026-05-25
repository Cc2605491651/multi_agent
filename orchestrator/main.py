"""CLI 入口（spec v5 全阶段）。

子命令：

- ``demo-phase1`` 5 轮对话 → 提炼 → 检索（阶段 1）
- ``demo-phase2`` 串行 2 节点 DAG（阶段 2）
- ``demo-phase3`` 双 Agent + 精确接力（阶段 3）
- ``demo-phase4a`` spec §5.4 完整 DAG + 三种 failure_policy（阶段 4a）
- ``run-task --dag --title [--handoff-conv --handoff-range ...]`` 通用 DAG 入口（阶段 4c）
- ``dashboard-serve --port 8000`` 启动 FastAPI 仪表盘（阶段 5）
- ``recall-baseline`` / ``recall-baseline-v2`` / ``recall-drift`` 召回质量评估
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv_if_present() -> None:
    """启动时自动加载项目根 ``.env``（如果存在）。

    - 已存在的环境变量**不覆盖**（用户在 shell 里 ``export`` 的优先级最高）
    - ``.env`` 不存在则静默；``python-dotenv`` 没装也静默
    - 缺这一步时仍可走老路 ``set -a; source .env; set +a``
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)


_load_dotenv_if_present()


from orchestrator.context_packer import ContextPacker
from orchestrator.dag_loader import instantiate_dag, load_dag
from orchestrator.failure_handler import FailureHandler
from orchestrator.planner import Planner, PlannerError
from orchestrator.recovery import Recovery
from orchestrator.scheduler import Scheduler
from storage.memory_store import MemoryStore
from storage.state_store import StateStore
from storage.transcript_store import TranscriptStore
from worker.agent import Agent, default_client
from worker.sandbox import make_sandbox

def _resolve_data_dir() -> Path:
    """数据目录解析优先级（从高到低）：
    1. MA_DATA_DIR 环境变量
    2. 当前工作目录下已存在的 ./data/（不破坏开发期习惯）
    3. ~/.multi_agent_tool/（pipx 全局安装默认）
    """
    import os as _os

    env_dir = _os.environ.get("MA_DATA_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    cwd_data = Path.cwd() / "data"
    if cwd_data.is_dir():
        return cwd_data.resolve()
    return Path.home() / ".multi_agent_tool"


DATA_DIR = _resolve_data_dir()
TRANSCRIPT_DB = DATA_DIR / "transcript.db"
CHROMA_DIR = DATA_DIR / "chroma"
STATE_DB = DATA_DIR / "state.db"

DEMO_TURNS: list[tuple[str, str]] = [
    # (user_input, mock_agent_output) —— mock 模式下用第二项；真实模式只用第一项
    ("我有一只橘猫，叫米饭，今年 3 岁", "好的，已记住——米饭，橘猫，3 岁。"),
    (
        "米饭最近半夜总抓门，把我吵醒，怎么办？",
        "可以试试增加白天的运动量，比如逗猫棒；夜里把卧室门关好。",
    ),
    (
        "对了，我自己只喝美式咖啡，从来不加糖",
        "好的，已记住你的咖啡偏好：美式、无糖。",
    ),
    (
        "我打算下个月带米饭去做绝育，你知道术后要注意什么吗？",
        "术后 24h 留意精神和食欲，伊丽莎白圈戴满 10 天防舔伤口，运动量减半两周。",
    ),
    (
        "顺便记一下，我和女朋友打算明年春天结婚",
        "恭喜！我会把这件事记下来。",
    ),
]

DEMO_RECALL_QUERY = "用户的宠物"


@dataclass
class _MockClient:
    """``--mock`` 模式：用预设输出 + 简单规则提炼。"""

    chat_outputs: list[str]
    _idx: dict[str, int] = field(default_factory=lambda: {"sonnet": 0})

    async def complete(self, *, model: str, system: str, messages, max_tokens=1024):
        if "提炼员" in system:
            # 简单规则提炼：取最后一条 user message，截短作为「结论」
            user_msg = ""
            for m in messages:
                if m["role"] == "user":
                    user_msg = m["content"]
            # 从 prompt 里挖 user_input 段
            tag = "【用户输入】"
            if tag in user_msg:
                user_msg = user_msg.split(tag, 1)[1]
                user_msg = user_msg.split("【", 1)[0]
            user_msg = user_msg.strip()
            if not user_msg or len(user_msg) < 3:
                return ""
            return f"用户提到：{user_msg[:50]}"
        idx = self._idx["sonnet"]
        if idx < len(self.chat_outputs):
            out = self.chat_outputs[idx]
            self._idx["sonnet"] = idx + 1
            return out
        return "好的，已收到。"


async def run_demo_phase1(*, mock: bool, reset: bool) -> int:
    if reset:
        for p in (TRANSCRIPT_DB, CHROMA_DIR):
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                import shutil

                shutil.rmtree(p)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    transcript_store = TranscriptStore(TRANSCRIPT_DB)
    memory_store = MemoryStore(CHROMA_DIR)

    if mock:
        client = _MockClient(chat_outputs=[o for _, o in DEMO_TURNS])
        print("[demo] running with --mock (no API calls)")
    else:
        try:
            client = default_client()
        except RuntimeError as e:
            print(f"[demo] {e}", file=sys.stderr)
            print("[demo] 建议先 export ANTHROPIC_API_KEY=... 或用 --mock", file=sys.stderr)
            return 2
        print("[demo] running with real Claude API")

    agent = Agent(agent_id="demo_agent", client=client)
    conversation_id = f"conv_{uuid.uuid4().hex[:8]}"
    user_id = "default_user"
    task_id = f"task_demo_{uuid.uuid4().hex[:6]}"
    print(f"[demo] user_id={user_id}  task_id={task_id}  conv_id={conversation_id}")

    history: list[dict] = []
    for i, (user_input, _) in enumerate(DEMO_TURNS, start=1):
        agent_output = await agent.respond(history, user_input)
        print(f"\n[round {i}] user: {user_input}")
        print(f"[round {i}] agent: {agent_output}")
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": agent_output})

        # 阶段 1 简化：transcript + memory(active) 直写，不走 §6.2 三步顺序
        await transcript_store.add_turn(
            conversation_id=conversation_id,
            turn_index=i,
            user_input=user_input,
            agent_output=agent_output,
            agent_id=agent.agent_id,
        )
        doc = await agent.distill(user_input, agent_output)
        if doc:
            mem_id = await memory_store.add(
                user_id,
                doc,
                {
                    "task_id": task_id,
                    "source_conversation_id": conversation_id,
                    "source_turn_index": i,
                    "produced_by_agent": agent.agent_id,
                    "produced_by_node": "",
                    "memory_level": "node_output",
                },
            )
            print(f"[memory extracted] {doc!r}  (mem_id={mem_id})")
        else:
            print("[memory] (skipped — 提炼空)")

    print(f"\n[query] {DEMO_RECALL_QUERY!r}")
    hits = await memory_store.search(DEMO_RECALL_QUERY, user_id, task_id, k=3)
    if not hits:
        print("[recall] (no hits)")
        return 1
    for h in hits:
        sim = 1.0 - h["distance"]
        print(f"  - sim={sim:.3f}  {h['document']}")
    return 0


# ---- 1.11 召回质量摸底 ----

_RECALL_DATASET: list[tuple[str, list[str]]] = [
    # (待写入的记忆 doc, 该 doc 应该被以下哪些 query 召回)
    ("用户有一只橘猫，叫米饭，3 岁", ["用户的宠物", "用户的猫", "米饭", "橘猫的年龄"]),
    ("用户养了一只柯基犬，叫豆豆", ["用户的狗", "豆豆", "柯基"]),
    ("用户喜欢喝美式咖啡，从不加糖", ["用户的咖啡偏好", "用户喝什么饮料"]),
    ("用户对花生过敏，吃花生会咳嗽", ["用户的过敏", "用户能吃花生吗", "花生"]),
    ("用户在上海浦东工作，住静安区", ["用户在哪工作", "用户住在哪", "上海"]),
    ("用户打算明年春天结婚", ["用户的婚姻状况", "用户的人生大事"]),
    ("用户用 MacBook Pro M3，16G 内存", ["用户的电脑", "用户的设备"]),
    ("用户最近在学 Rust 编程", ["用户在学什么", "用户的编程语言"]),
    ("用户喜欢周末爬山", ["用户的爱好", "用户周末做什么"]),
    ("用户的妈妈生日是 12 月 5 日", ["妈妈的生日", "家人生日"]),
    ("用户曾在阿里巴巴工作 5 年", ["用户的工作经历", "用户以前在哪上班"]),
    ("用户的本科是计算机专业，毕业于清华", ["用户的学历", "用户毕业院校"]),
    ("用户的米饭猫绝育术后恢复良好", ["米饭术后", "猫的术后情况"]),
    ("用户预算每月养宠 800 元", ["养宠预算", "用户养猫花多少钱"]),
    ("用户开蓝色 Model 3", ["用户的车", "用户开什么车"]),
    ("用户的家庭医生是张医生，每周三出诊", ["用户的医生", "看病时间"]),
    ("用户最爱看刘慈欣的科幻小说", ["用户喜欢的书", "用户看什么小说"]),
    ("用户上次旅行去了云南大理", ["用户去过哪", "用户的旅行"]),
    ("用户每天晚上 11 点睡觉", ["用户的作息", "用户几点睡"]),
    ("用户最近三个月在减肥，目标 10kg", ["用户的健康目标", "用户的减肥计划"]),
]


async def run_recall_baseline(*, k: int = 5) -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    persist = DATA_DIR / "chroma_recall_baseline"
    if persist.exists():
        import shutil

        shutil.rmtree(persist)
    memory_store = MemoryStore(persist)

    user_id = "recall_test"
    task_id = "task_recall"
    doc_to_mem: dict[str, str] = {}
    for doc, _ in _RECALL_DATASET:
        mid = await memory_store.add(user_id, doc, {"task_id": task_id})
        doc_to_mem[doc] = mid

    queries: list[tuple[str, str]] = []
    for doc, q_list in _RECALL_DATASET:
        for q in q_list:
            queries.append((q, doc_to_mem[doc]))

    hits_at_5 = 0
    mrr_sum = 0.0
    details: list[dict] = []
    for q, expected_mid in queries:
        results = await memory_store.search(q, user_id, task_id, k=k)
        ids = [r["id"] for r in results]
        hit = expected_mid in ids
        rank = ids.index(expected_mid) + 1 if hit else 0
        if hit:
            hits_at_5 += 1
            mrr_sum += 1.0 / rank
        details.append(
            {
                "query": q,
                "expected_doc": next(d for d, m in doc_to_mem.items() if m == expected_mid),
                "rank": rank,
                "top1": results[0]["document"] if results else None,
            }
        )

    p_at_5 = hits_at_5 / len(queries)
    mrr = mrr_sum / len(queries)
    print(f"\n=== 召回质量基线（k={k}） ===")
    print(f"样本：{len(_RECALL_DATASET)} 条记忆，{len(queries)} 条 query")
    print(f"P@{k} = {p_at_5:.3f}   MRR = {mrr:.3f}")
    print(f"未命中样例：")
    miss_count = 0
    for d in details:
        if d["rank"] == 0:
            miss_count += 1
            if miss_count <= 5:
                print(f"  - q={d['query']!r}  expected={d['expected_doc']!r}")
                print(f"    top1={d['top1']!r}")
    if miss_count == 0:
        print("  (全部命中)")

    import json

    out_path = DATA_DIR / "recall_baseline.json"
    out_path.write_text(
        json.dumps(
            {"p_at_k": p_at_5, "mrr": mrr, "k": k, "samples": len(queries), "details": details},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n结果已写入 {out_path}")
    return 0


# ============ 阶段 2 demo：串行 2 节点 DAG ============

_PHASE2_TASK_TITLE = "调研并撰写：橘猫米饭的居家护理 3 条要点"

_PHASE2_MOCK_OUTPUTS = {
    "research": (
        "1) 橘猫消化敏感，固定时间投喂、品牌不轻易切换；"
        "2) 室内放置抓板，减少夜间抓门频率；"
        "3) 每年一次体检，重点查泌尿系统。"
    ),
    "writing": (
        "建议：固定喂食习惯（同品牌定时定量）、配置抓板转移夜间精力、"
        "每年体检关注泌尿——这三件事覆盖了橘猫米饭最常见的居家风险点。"
    ),
}


@dataclass
class _Phase2MockClient:
    """按 user_input 里 [node:xxx] 标识返回不同 mock 输出。"""

    chat_outputs: dict[str, str]

    async def complete(self, *, model: str, system: str, messages, max_tokens=1024):
        if "提炼员" in system:
            content = messages[-1]["content"]
            tag = "【Agent 输出】"
            if tag in content:
                tail = content.split(tag, 1)[1]
                # 截掉 distill prompt 末尾固定的「请按要求...」尾巴
                for stop in ("\n\n请按", "\n请按", "【"):
                    if stop in tail:
                        tail = tail.split(stop, 1)[0]
                        break
                content = tail.strip()
            if not content:
                return ""
            return content[:160]
        # chat 调用：从 user message 里挖出 [node:xxx]
        text = messages[-1]["content"]
        for key, out in self.chat_outputs.items():
            if f"[node:{key}]" in text:
                return out
        return "（mock 默认回复）"


def _phase2_sub_task(node, ctx) -> str:
    if node.node_name == "research":
        return "[node:research] 围绕任务给出 3 条最关键的护理事实，每条 ≤ 20 字"
    if node.node_name == "writing":
        return "[node:writing] 基于上游产出，综合写一段 100 字内的护理建议"
    return f"[node:{node.node_name}] 完成"


async def run_demo_phase2(*, mock: bool, reset: bool) -> int:
    if reset:
        import shutil

        for p in (TRANSCRIPT_DB, CHROMA_DIR, STATE_DB):
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    transcript_store = TranscriptStore(TRANSCRIPT_DB)
    memory_store = MemoryStore(CHROMA_DIR)
    state_store = StateStore(STATE_DB)
    sandbox = make_sandbox()
    recovery = Recovery(state_store, memory_store, stale_seconds=300)

    if mock:
        client = _Phase2MockClient(chat_outputs=_PHASE2_MOCK_OUTPUTS)
        print("[demo2] running with --mock")
    else:
        try:
            client = default_client()
        except RuntimeError as e:
            print(f"[demo2] {e}", file=sys.stderr)
            return 2
        print("[demo2] running with real Claude API")

    user_id = "default_user"
    task_id = await state_store.create_task(
        user_id=user_id, title=_PHASE2_TASK_TITLE, dag_id="phase2_simple"
    )
    n_research = await state_store.create_dag_node(
        task_id=task_id, node_name="research"
    )
    n_writing = await state_store.create_dag_node(
        task_id=task_id, node_name="writing", depends_on=[n_research]
    )
    print(f"[demo2] task={task_id} | nodes: research={n_research}  writing={n_writing}")

    packer = ContextPacker(
        state_store=state_store,
        transcript_store=transcript_store,
        memory_store=memory_store,
    )
    scheduler = Scheduler(
        state_store=state_store,
        transcript_store=transcript_store,
        memory_store=memory_store,
        sandbox=sandbox,
        llm_client=client,
        recovery=recovery,
        context_packer=packer,
        failure_handler=FailureHandler(state_store, memory_store),
        sub_task_builder=_phase2_sub_task,
        heartbeat_interval=2.0,
        force_default_client=mock,
    )
    final = await scheduler.run_task(task_id)
    print(f"\n[demo2] task final status: {final}")

    nodes = await state_store.list_dag_nodes(task_id)
    for n in nodes:
        print(
            f"  - {n.node_name}: status={n.status}  retry={n.retry_count}  "
            f"mem={n.output_memory_id}"
        )

    print("\n[demo2] 记忆库内容（active）：")
    hits = await memory_store.search("护理建议", user_id, task_id, k=5)
    for h in hits:
        sim = 1.0 - h["distance"]
        print(f"  sim={sim:.3f}  {h['document']}")
    return 0 if final == "done" else 1


# ============ 阶段 3 任务 3.6：id 取 vs 语义召回对比 ============


_DRIFT_DOCS = [
    # (doc, 是否「最终决策」) —— 都和"决策/结论/选型/A 工具"沾边，刁难召回
    (
        "选型决策：最终选 A 工具，理由是开源 + 本地部署，年成本 30 万落在 50 万预算内。",
        True,
    ),
    ("评估结论：B 工具是商业版，本地部署，年成本 40 万，超出预算被否决。", False),
    ("评估结论：C 工具是云端 SaaS，年成本 20 万，但不支持本地部署被否决。", False),
    ("调研结论：可选方案为 A 工具、B 工具、C 工具三款。", False),
    ("需求结论：硬约束是必须本地部署 + 年预算 ≤ 50 万 + 团队规模 100 人。", False),
    ("中间结论：A 工具的开源协议是 Apache 2.0，可商用。", False),
    ("阶段决策：先短列 3 款再做对比评估，最后选 1 款。", False),
    ("过往决策：去年我们选过类似的 D 工具，但 6 个月后弃用了。", False),
]

_DRIFT_QUERIES = [
    "最终选型决策",
    "我们选了哪款",
    "决策结论",
    "工具选了什么",
    "选型最终落到哪个",
    "结论",
    "我们的决策",
]


async def run_recall_drift(*, k: int = 3) -> int:
    """对比 spec §3.3 「P0 级」判断：input_memory_ids 精确接力 vs 纯语义召回。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    persist = DATA_DIR / "chroma_recall_drift"
    if persist.exists():
        import shutil

        shutil.rmtree(persist)
    memory_store = MemoryStore(persist)
    user_id = "drift_test"
    task_id = "task_drift"

    correct_mid: str | None = None
    for doc, is_correct in _DRIFT_DOCS:
        mid = await memory_store.add(user_id, doc, {"task_id": task_id})
        if is_correct:
            correct_mid = mid
    assert correct_mid is not None

    print("=" * 70)
    print("阶段 3 任务 3.6：input_memory_ids 精确取 vs 语义召回")
    print("=" * 70)
    print(f"样本：{len(_DRIFT_DOCS)} 条记忆（含 1 条「正确答案」），{len(_DRIFT_QUERIES)} 条 query")
    print(f"\nA. 用 input_memory_ids = [{correct_mid}] 精确取：")
    by_id = await memory_store.get_by_ids(user_id, [correct_mid])
    print(f"   命中 1/1（100%）；原文：{by_id[0]['document']}")

    print(f"\nB. 用 query 做语义召回（top-{k}）：")
    top1_hits = 0
    topk_hits = 0
    misses: list[tuple[str, str]] = []
    for q in _DRIFT_QUERIES:
        hits = await memory_store.search(q, user_id, task_id, k=k)
        ids = [h["id"] for h in hits]
        if ids and ids[0] == correct_mid:
            top1_hits += 1
        if correct_mid in ids:
            topk_hits += 1
        else:
            misses.append((q, hits[0]["document"] if hits else "(empty)"))

    n = len(_DRIFT_QUERIES)
    print(f"   top-1 命中率: {top1_hits}/{n} ({top1_hits/n:.0%})")
    print(f"   top-{k} 命中率: {topk_hits}/{n} ({topk_hits/n:.0%})")
    if misses:
        print(f"   top-{k} 仍未命中（拿到错的）样例：")
        for q, top in misses:
            print(f"     - q={q!r}")
            print(f"       top1={top}")

    print("\n" + "=" * 70)
    if top1_hits < n:
        print(
            "结论：语义召回 top-1 在 query 措辞上有飘移；按 input_memory_ids 精确取 100% 可靠。"
        )
        print("→ 验证 spec §3.3 P0 判断成立：精确接力 ≠ 召回，二者不可替代。")
    else:
        print(
            "本批样本下 top-1 仍 100% 命中，但 query 一旦再「松」一些会立刻飘移。"
        )
    print("=" * 70)

    import json

    out_path = DATA_DIR / "recall_drift.json"
    out_path.write_text(
        json.dumps(
            {
                "samples": len(_DRIFT_DOCS),
                "queries": _DRIFT_QUERIES,
                "id_path_hit_rate": 1.0,
                "search_top1_hit_rate": top1_hits / n,
                "search_topk_hit_rate": topk_hits / n,
                "k": k,
                "misses": [{"q": q, "top1": top} for q, top in misses],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n结果已写入 {out_path}")
    return 0


# ============ 阶段 3 demo：双 Agent + 精确接力 ============

_PHASE3_TASK_TITLE = (
    "为「100 人研发团队 + 50 万年预算 + 必须本地部署」选型 AI 工作流工具"
)

# 模拟用户和早期 agent 聊过的几轮——任务在这之上接力
_PHASE3_HISTORY = [
    ("我想给团队选一款 AI 工作流工具", "好的，目标团队规模是多少？"),
    ("100 人左右的研发团队", "预算和必须满足的硬性条件呢？"),
    ("年预算 50 万以内，必须能本地部署", "明白，我可以帮你产出一份选型决策"),
]

_PHASE3_MOCK_OUTPUTS = {
    "research": (
        "调研结果——3 款候选：A 工具（开源、本地、年成本 30 万）；"
        "B 工具（商业、本地、年成本 40 万）；C 工具（云端 SaaS、年成本 20 万）。"
    ),
    "writing": (
        "决策：选 A 工具。理由——开源 + 本地部署同时满足两条硬约束；"
        "年成本 30 万落在 50 万预算内；B 价格偏高，C 不支持本地部署故剔除。"
    ),
}


def _phase3_sub_task(node, ctx) -> str:
    if node.node_name == "research":
        return (
            "[node:research] 调研可选方案：列出 3 个候选 + 关键参数"
            "（部署模式、年成本）"
        )
    if node.node_name == "writing":
        return (
            "[node:writing] 基于上游产出与接力点中的硬约束，"
            "给出最终选型决策与理由"
        )
    return f"[node:{node.node_name}] 完成"


async def run_demo_phase3(*, mock: bool, reset: bool) -> int:
    if reset:
        import shutil

        for p in (TRANSCRIPT_DB, CHROMA_DIR, STATE_DB):
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    transcript_store = TranscriptStore(TRANSCRIPT_DB)
    memory_store = MemoryStore(CHROMA_DIR)
    state_store = StateStore(STATE_DB)
    sandbox = make_sandbox()
    recovery = Recovery(state_store, memory_store, stale_seconds=300)
    packer = ContextPacker(
        state_store=state_store,
        transcript_store=transcript_store,
        memory_store=memory_store,
    )

    if mock:
        client = _Phase2MockClient(chat_outputs=_PHASE3_MOCK_OUTPUTS)
        print("[demo3] running with --mock")
    else:
        try:
            client = default_client()
        except RuntimeError as e:
            print(f"[demo3] {e}", file=sys.stderr)
            return 2
        print("[demo3] running with real Claude API")

    user_id = "default_user"

    # 1. 写「历史对话」作为接力点原文
    conv_id = "conv_phase3_history"
    for i, (u, a) in enumerate(_PHASE3_HISTORY, start=1):
        await transcript_store.add_turn(
            conversation_id=conv_id,
            turn_index=i,
            user_input=u,
            agent_output=a,
            agent_id="planner",
        )
    print(f"[demo3] 历史对话已写入 {conv_id}（共 {len(_PHASE3_HISTORY)} 轮）")

    # 2. 建 task + DAG，task 引用接力点
    task_id = await state_store.create_task(
        user_id=user_id,
        title=_PHASE3_TASK_TITLE,
        dag_id="phase3_serial",
        handoff_conversation_id=conv_id,
        handoff_turn_range=[1, len(_PHASE3_HISTORY)],
    )
    n_research = await state_store.create_dag_node(
        task_id=task_id, node_name="research"
    )
    n_writing = await state_store.create_dag_node(
        task_id=task_id, node_name="writing", depends_on=[n_research]
    )
    print(f"[demo3] task={task_id}")
    print(f"[demo3] DAG: research={n_research} → writing={n_writing}")

    # 3. 跑 scheduler
    scheduler = Scheduler(
        state_store=state_store,
        transcript_store=transcript_store,
        memory_store=memory_store,
        sandbox=sandbox,
        llm_client=client,
        recovery=recovery,
        context_packer=packer,
        failure_handler=FailureHandler(state_store, memory_store),
        sub_task_builder=_phase3_sub_task,
        heartbeat_interval=2.0,
        force_default_client=mock,
    )
    final = await scheduler.run_task(task_id)
    print(f"\n[demo3] task 最终状态: {final}")

    # 4. 重新拉一次 writing 节点的 packed context 验证接力 + 上游精确产出
    packed = await packer.pack(
        task_id=task_id,
        node_id=n_writing,
        sub_task_description=_phase3_sub_task(
            await state_store.get_dag_node(n_writing),
            type("Ctx", (), {"task_id": task_id, "title": _PHASE3_TASK_TITLE, "user_id": user_id})(),
        ),
    )
    print(
        "\n==== 阶段 3 验收：writing 节点收到的上下文 ===="
    )
    print(packed.text)
    print(
        f"\n==== 上游产出 present={packed.upstream_present} "
        f"missing={packed.upstream_missing} | handoff={packed.handoff_present} ===="
    )

    # 5. 节点状态摘要
    for n in await state_store.list_dag_nodes(task_id):
        print(
            f"  - {n.node_name}: status={n.status}  "
            f"input_mids={n.input_memory_ids}  output_mid={n.output_memory_id}"
        )
    return 0 if final == "done" else 1


# ============ 阶段 4a demo：spec §5.4 完整 DAG ============

_PHASE4A_TASK_TITLE = "多源调研 + 双写作 + 汇总 demo（spec §5.4）"

_PHASE4A_MOCK_OUTPUTS = {
    "research_a": "调研 A：A 工具开源、本地部署、年成本 30 万。",
    "research_b": "（research_b 通常应失败 → fail_skip → 下游 writing_b 拿不到产出）",
    "research_c": "调研 C：C 工具云端 SaaS、年 20 万、不支持本地。",
    "writing_a": "基于 research_a 写出：A 工具是首选，开源且符合预算。",
    "writing_b": "基于 research_b 写出：调研 B 缺失，本节段建议另起调研。",
    "summarize": "汇总：综合 research_c + writing_a + writing_b 给出最终决策。选 A 工具。",
}


def _phase4a_sub_task(node, ctx) -> str:
    return f"[node:{node.node_name}] 完成 {ctx.title} 的「{node.node_name}」子任务"


async def run_demo_phase4a(*, mock: bool, reset: bool, fail_b: bool) -> int:
    if reset:
        import shutil

        for p in (TRANSCRIPT_DB, CHROMA_DIR, STATE_DB):
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    transcript_store = TranscriptStore(TRANSCRIPT_DB)
    memory_store = MemoryStore(CHROMA_DIR)
    state_store = StateStore(STATE_DB)
    sandbox = make_sandbox()
    recovery = Recovery(state_store, memory_store, stale_seconds=300)
    packer = ContextPacker(
        state_store=state_store,
        transcript_store=transcript_store,
        memory_store=memory_store,
    )

    if mock:
        # 用一个增强 mock：可对 research_b 抛错以演示 fail_skip
        @dataclass
        class _Phase4aClient:
            chat_map: dict[str, str]
            fail_nodes: set[str]

            async def complete(self, *, model: str, system: str, messages, max_tokens=1024):
                text = messages[-1]["content"]
                if "提炼员" in system:
                    tag = "【Agent 输出】"
                    if tag in text:
                        tail = text.split(tag, 1)[1]
                        for stop in ("\n\n请按", "\n请按", "【"):
                            if stop in tail:
                                tail = tail.split(stop, 1)[0]
                                break
                        return tail.strip()[:160]
                    return text.strip()[:160]
                for nname in self.fail_nodes:
                    if f"[node:{nname}]" in text:
                        raise RuntimeError(f"scripted failure for {nname}")
                for nname, out in self.chat_map.items():
                    if f"[node:{nname}]" in text:
                        return out
                return "default"

        fail_set = {"research_b"} if fail_b else set()
        client = _Phase4aClient(
            chat_map=_PHASE4A_MOCK_OUTPUTS, fail_nodes=fail_set
        )
        print(f"[demo4a] running with --mock  fail_b={fail_b}")
    else:
        try:
            client = default_client()
        except RuntimeError as e:
            print(f"[demo4a] {e}", file=sys.stderr)
            return 2
        print("[demo4a] running with real Claude API")

    user_id = "default_user"
    dag_path = Path(__file__).resolve().parent.parent / "dags" / "research_report.json"
    dag = load_dag(dag_path)
    print(f"[demo4a] loaded DAG: {dag.dag_id} ({len(dag.nodes)} nodes)")

    task_id, mapping = await instantiate_dag(
        state_store, dag, user_id=user_id, title=_PHASE4A_TASK_TITLE
    )
    print(f"[demo4a] task={task_id}")
    for logical_id, node_id in mapping.items():
        print(f"   {logical_id} → {node_id}")

    scheduler = Scheduler(
        state_store=state_store,
        transcript_store=transcript_store,
        memory_store=memory_store,
        sandbox=sandbox,
        llm_client=client,
        recovery=recovery,
        context_packer=packer,
        failure_handler=FailureHandler(state_store, memory_store),
        sub_task_builder=_phase4a_sub_task,
        max_concurrent_workers=3,
        heartbeat_interval=2.0,
        cancel_timeout=2.0,
        force_default_client=mock,
    )
    final = await scheduler.run_task(task_id)
    print(f"\n[demo4a] task 最终状态: {final}\n")

    print("节点最终状态：")
    for n in await state_store.list_dag_nodes(task_id):
        print(
            f"  - {n.node_name:12s}  status={n.status:8s}  retry={n.retry_count}  "
            f"policy={n.failure_policy:11s}  mem={n.output_memory_id}"
        )
    return 0 if final == "done" else 1


# ============ 通用 run-task（接力点参数化，spec §11 阶段 4c 任务 4c.3）============


def _generic_sub_task(node, ctx) -> str:
    return f"[node:{node.node_name}] 完成 「{ctx.title}」 的 「{node.node_name}」 子任务"


async def run_task_cli(
    *,
    dag_path: str,
    title: str,
    handoff_conv: str | None,
    handoff_range: tuple[int, int] | None,
    user_id: str,
    mock: bool,
    reset: bool,
    max_concurrent: int,
    workdir: str | None = None,
) -> int:
    if reset:
        import shutil

        for p in (TRANSCRIPT_DB, CHROMA_DIR, STATE_DB):
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    transcript_store = TranscriptStore(TRANSCRIPT_DB)
    memory_store = MemoryStore(CHROMA_DIR)
    state_store = StateStore(STATE_DB)
    sandbox = make_sandbox(workdir=workdir)
    recovery = Recovery(state_store, memory_store, stale_seconds=300)
    packer = ContextPacker(
        state_store=state_store,
        transcript_store=transcript_store,
        memory_store=memory_store,
    )

    if mock:
        @dataclass
        class _GenericMock:
            async def complete(self, *, model, system, messages, max_tokens=1024):
                text = messages[-1]["content"]
                if "提炼员" in system:
                    tag = "【Agent 输出】"
                    if tag in text:
                        tail = text.split(tag, 1)[1]
                        for stop in ("\n\n请按", "\n请按", "【"):
                            if stop in tail:
                                tail = tail.split(stop, 1)[0]
                                break
                        return tail.strip()[:160]
                    return text.strip()[:160]
                # 回执：根据 [node:xxx] 标识返回
                import re as _re

                m = _re.search(r"\[node:(\w+)\]", text)
                name = m.group(1) if m else "node"
                return f"[mock 输出] 节点 {name} 完成"

        client = _GenericMock()
        print("[run-task] mock 模式")
    else:
        try:
            client = default_client()
        except RuntimeError as e:
            print(f"[run-task] {e}", file=sys.stderr)
            return 2
        print("[run-task] 真实 Claude API 模式")

    dag = load_dag(dag_path)
    print(f"[run-task] DAG: {dag.dag_id} ({len(dag.nodes)} nodes)")

    task_id, mapping = await instantiate_dag(
        state_store,
        dag,
        user_id=user_id,
        title=title,
        handoff_conversation_id=handoff_conv,
        handoff_turn_range=list(handoff_range) if handoff_range else None,
    )
    print(f"[run-task] task={task_id}")
    if handoff_conv:
        print(f"[run-task] handoff = conv={handoff_conv} range={handoff_range}")
    for logical_id, node_id in mapping.items():
        print(f"   {logical_id} → {node_id}")

    scheduler = Scheduler(
        state_store=state_store,
        transcript_store=transcript_store,
        memory_store=memory_store,
        sandbox=sandbox,
        llm_client=client,
        recovery=recovery,
        context_packer=packer,
        failure_handler=FailureHandler(state_store, memory_store),
        sub_task_builder=_generic_sub_task,
        max_concurrent_workers=max_concurrent,
        heartbeat_interval=2.0,
        cancel_timeout=2.0,
        force_default_client=mock,  # mock 时忽略 harness.provider 真实切换
    )
    final = await scheduler.run_task(task_id)
    print(f"\n[run-task] task 最终状态: {final}\n")

    # 阶段 4c 验收：打印每个节点的 packed context token 数（不超 2K）
    nodes = await state_store.list_dag_nodes(task_id)
    print("每个节点的 context token 实测：")
    for n in nodes:
        try:
            packed = await packer.pack(
                task_id=task_id, node_id=n.id,
                sub_task_description=_generic_sub_task(
                    n, type("Ctx", (), {"task_id": task_id, "title": title, "user_id": user_id})(),
                ),
            )
            print(
                f"  - {n.node_name:14s}  status={n.status:8s}  "
                f"tokens={packed.token_count:4d}  "
                f"sem_added={packed.semantic_added}  sem_dropped={packed.semantic_dropped_for_budget}"
            )
        except Exception as e:
            print(f"  - {n.node_name}: pack failed ({e})")
    return 0 if final == "done" else 1


# ============ 4c.5 召回质量 v2（context_packer 路径）============


async def run_recall_baseline_v2(*, k: int = 5) -> int:
    """与 1.11 同 20 条 doc + 45 query，但走 context_packer 内部的 query 构造路径
    （title + sub_task + 上游摘要 + memory_level 排序），对比基线。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    persist = DATA_DIR / "chroma_recall_v2"
    if persist.exists():
        import shutil

        shutil.rmtree(persist)
    state_db = DATA_DIR / "state_recall_v2.db"
    if state_db.exists():
        state_db.unlink()
    transcript_db = DATA_DIR / "transcript_recall_v2.db"
    if transcript_db.exists():
        transcript_db.unlink()

    memory_store = MemoryStore(persist)
    state_store = StateStore(state_db)
    transcript_store = TranscriptStore(transcript_db)
    packer = ContextPacker(
        state_store=state_store,
        transcript_store=transcript_store,
        memory_store=memory_store,
    )

    user_id = "recall_test_v2"
    task_id = await state_store.create_task(
        user_id=user_id, title="召回基线 v2", dag_id="recall_v2"
    )

    # 复用 1.11 的 20 条样本
    doc_to_mem: dict[str, str] = {}
    for doc, _ in _RECALL_DATASET:
        mid = await memory_store.add(
            user_id,
            doc,
            {
                "task_id": task_id,
                "produced_by_node": "n_seed",
                "produced_by_agent": "seed",
                "memory_level": "node_output",
                "status": "active",
            },
        )
        doc_to_mem[doc] = mid

    queries: list[tuple[str, str]] = []
    for doc, q_list in _RECALL_DATASET:
        for q in q_list:
            queries.append((q, doc_to_mem[doc]))

    # 用一个独立节点做 pack，每次只让它的 sub_task 等同于 query
    probe_node_id = await state_store.create_dag_node(
        task_id=task_id, node_name="probe"
    )
    await state_store.set_node_input_memory_ids(probe_node_id, [])

    hits_at_k = 0
    mrr_sum = 0.0
    details: list[dict] = []
    for q, expected_mid in queries:
        packed = await packer.pack(
            task_id=task_id, node_id=probe_node_id,
            sub_task_description=q,
        )
        # 用 packer 内部的查询路径
        results = await memory_store.search(
            packed.query_used, user_id, task_id, k=k,
            cross_task=False, status="active",
        )
        ids = [r["id"] for r in results]
        hit = expected_mid in ids
        rank = ids.index(expected_mid) + 1 if hit else 0
        if hit:
            hits_at_k += 1
            mrr_sum += 1.0 / rank
        details.append(
            {
                "query": q, "rank": rank,
                "query_used": packed.query_used,
                "top1": results[0]["document"] if results else None,
            }
        )

    n = len(queries)
    p_at_k = hits_at_k / n
    mrr = mrr_sum / n
    print(f"\n=== 召回质量基线 v2（k={k}）context_packer 路径 ===")
    print(f"样本：{len(_RECALL_DATASET)} 条记忆，{n} 条 query")
    print(f"P@{k} = {p_at_k:.3f}   MRR = {mrr:.3f}")
    miss = [d for d in details if d["rank"] == 0]
    if miss:
        print(f"未命中 {len(miss)} 条：")
        for d in miss[:5]:
            print(f"  - q={d['query']!r}  top1={d['top1']!r}")
    else:
        print("（全部命中）")

    import json

    out_path = DATA_DIR / "recall_baseline_v2.json"
    out_path.write_text(
        json.dumps(
            {"p_at_k": p_at_k, "mrr": mrr, "k": k, "samples": n, "details": details},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n结果已写入 {out_path}")
    return 0


# ============ Planner Agent：plan-task ============

# mock 模式用的最小 fixture DAG（自然语言 → DAG 完全离线演示）
_PLAN_TASK_FIXTURE = {
    "dag_id": "planned_demo",
    "description": "Planner mock 输出（无 LLM 调用）",
    "nodes": [
        {
            "id": "n1", "name": "research", "deps": [],
            "failure_policy": "fail_retry",
            "harness": {
                "model": "deepseek-chat", "provider": "deepseek",
                "system_prompt": "你是「调研」Agent，把用户目标拆成 3 条关键事实。",
                "tools": ["web_search"],
            },
        },
        {
            "id": "n2", "name": "summarize", "deps": ["n1"],
            "failure_policy": "fail_fast",
            "memory_level": "task_conclusion",
            "harness": {
                "model": "deepseek-chat", "provider": "deepseek",
                "system_prompt": "你是「汇总」Agent，输出层级 Markdown 报告。",
                "tools": ["write_file"],
                "skills": [{"name": "structured-output"}],
            },
        },
    ],
}


async def run_plan_task(
    *,
    goal: str,
    out_path: str | None,
    mock: bool,
    reset: bool,
    max_concurrent: int,
    user_id: str,
    workdir: str | None = None,
) -> int:
    """plan-task：用 Planner 把目标转成 DAG JSON，落盘后直接 run-task。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if mock:
        dag_dict = _PLAN_TASK_FIXTURE
        print("[plan-task] mock 模式：使用 fixture DAG（不调 LLM）")
    else:
        try:
            client = default_client()
        except RuntimeError as e:
            print(f"[plan-task] {e}", file=sys.stderr)
            return 2
        planner = Planner(client)
        print(f"[plan-task] 调 LLM 设计 DAG，目标：{goal}")
        try:
            result = await planner.plan(goal)
        except PlannerError as e:
            print(f"[plan-task] Planner 失败：{e}", file=sys.stderr)
            return 3
        dag_dict = result.dag_dict
        print(
            f"[plan-task] Planner OK（{result.attempts} 次尝试）："
            f"{dag_dict.get('dag_id')} | {len(dag_dict.get('nodes', []))} 节点"
        )

    if not out_path:
        import time

        out_path = str(DATA_DIR / f"planned_{int(time.time())}.json")
    Path(out_path).write_text(
        json.dumps(dag_dict, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[plan-task] DAG 已写入 {out_path}")

    # 复用 run_task_cli 跑出来
    return await run_task_cli(
        dag_path=out_path, title=goal,
        handoff_conv=None, handoff_range=None,
        user_id=user_id, mock=mock, reset=reset,
        max_concurrent=max_concurrent,
        workdir=workdir,
    )


# ============ 阶段 5 dashboard-serve ============


def run_dashboard_serve(*, host: str, port: int) -> int:
    import uvicorn

    from orchestrator.api import create_app

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    app = create_app(STATE_DB)
    print(
        f"[dashboard] serving on http://{host}:{port}  (state_db={STATE_DB})\n"
        f"[dashboard] 任务列表 + DAG 状态实时可视化（spec §10）"
    )
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0


# ============ CLI 入口 ============


def main() -> int:
    # 不带子命令 / 只有 multi-agent → 进交互向导
    if len(sys.argv) == 1:
        from orchestrator.wizard import run_wizard

        return run_wizard()

    parser = argparse.ArgumentParser(prog="multi-agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_demo1 = sub.add_parser("demo-phase1", help="阶段 1 端到端 demo")
    p_demo1.add_argument("--mock", action="store_true")
    p_demo1.add_argument("--reset", action="store_true")

    p_demo2 = sub.add_parser("demo-phase2", help="阶段 2 串行 2 节点 DAG")
    p_demo2.add_argument("--mock", action="store_true")
    p_demo2.add_argument("--reset", action="store_true")

    p_demo3 = sub.add_parser("demo-phase3", help="阶段 3 双 Agent + 精确接力")
    p_demo3.add_argument("--mock", action="store_true")
    p_demo3.add_argument("--reset", action="store_true")

    p_demo4a = sub.add_parser(
        "demo-phase4a", help="阶段 4a 完整 DAG（spec §5.4）+ 失败模型 + 并发"
    )
    p_demo4a.add_argument("--mock", action="store_true")
    p_demo4a.add_argument("--reset", action="store_true")
    p_demo4a.add_argument(
        "--fail-b", action="store_true",
        help="模拟 research_b 一直失败，演示 fail_skip 跳过 + 下游照常跑",
    )

    p_run = sub.add_parser(
        "run-task", help="跑任意 DAG JSON + 可选接力点（阶段 4c 任务 4c.3）"
    )
    p_run.add_argument("--dag", required=True, help="DAG JSON 路径")
    p_run.add_argument("--title", required=True, help="任务主题")
    p_run.add_argument("--user-id", default="default_user")
    p_run.add_argument(
        "--handoff-conv", default=None,
        help="接力点 conversation_id（须事先存在于 transcript_store）",
    )
    p_run.add_argument(
        "--handoff-range", default=None,
        help="接力点 turn 范围，格式 start,end，如 1,5",
    )
    p_run.add_argument("--mock", action="store_true")
    p_run.add_argument("--reset", action="store_true")
    p_run.add_argument("--max-concurrent", type=int, default=3)
    p_run.add_argument(
        "--workdir", default=None,
        help="让 agent 在指定目录工作（直接读写你的项目文件，请勿对未版本控制目录使用）",
    )

    p_recall = sub.add_parser("recall-baseline", help="阶段 1 任务 1.11 召回基线")
    p_recall.add_argument("-k", type=int, default=5)

    p_recall_v2 = sub.add_parser(
        "recall-baseline-v2",
        help="阶段 4c 任务 4c.5：context_packer 路径召回质量",
    )
    p_recall_v2.add_argument("-k", type=int, default=5)

    p_dash = sub.add_parser(
        "dashboard-serve", help="阶段 5：启动运行时仪表盘（spec §10）"
    )
    p_dash.add_argument("--host", default="127.0.0.1")
    p_dash.add_argument("--port", type=int, default=8000)

    p_plan = sub.add_parser(
        "plan-task",
        help="Planner Agent：自然语言目标 → DAG JSON → 直接 run-task（spec v5 §9.7）",
    )
    p_plan.add_argument("--goal", required=True, help="任务的自然语言描述")
    p_plan.add_argument("--out", default=None, help="DAG 落盘路径；默认 data/planned_<ts>.json")
    p_plan.add_argument("--user-id", default="default_user")
    p_plan.add_argument("--mock", action="store_true", help="不调 LLM，用 fixture DAG")
    p_plan.add_argument("--reset", action="store_true")
    p_plan.add_argument("--max-concurrent", type=int, default=3)
    p_plan.add_argument(
        "--workdir", default=None,
        help="让 agent 在指定目录工作（直接读写你的项目文件，请勿对未版本控制目录使用）",
    )

    p_drift = sub.add_parser(
        "recall-drift", help="阶段 3 任务 3.6 对比实验：id 取 vs 语义召回飘移"
    )
    p_drift.add_argument("-k", type=int, default=3)

    args = parser.parse_args()
    if args.cmd == "demo-phase1":
        return asyncio.run(run_demo_phase1(mock=args.mock, reset=args.reset))
    if args.cmd == "demo-phase2":
        return asyncio.run(run_demo_phase2(mock=args.mock, reset=args.reset))
    if args.cmd == "demo-phase3":
        return asyncio.run(run_demo_phase3(mock=args.mock, reset=args.reset))
    if args.cmd == "demo-phase4a":
        return asyncio.run(
            run_demo_phase4a(
                mock=args.mock, reset=args.reset, fail_b=args.fail_b
            )
        )
    if args.cmd == "run-task":
        handoff_range = None
        if args.handoff_range:
            parts = args.handoff_range.split(",")
            if len(parts) != 2:
                print(
                    "[run-task] --handoff-range 格式应为 start,end（如 1,5）",
                    file=sys.stderr,
                )
                return 2
            handoff_range = (int(parts[0]), int(parts[1]))
        return asyncio.run(
            run_task_cli(
                dag_path=args.dag,
                title=args.title,
                handoff_conv=args.handoff_conv,
                handoff_range=handoff_range,
                user_id=args.user_id,
                mock=args.mock,
                reset=args.reset,
                max_concurrent=args.max_concurrent,
                workdir=args.workdir,
            )
        )
    if args.cmd == "recall-baseline":
        return asyncio.run(run_recall_baseline(k=args.k))
    if args.cmd == "recall-baseline-v2":
        return asyncio.run(run_recall_baseline_v2(k=args.k))
    if args.cmd == "recall-drift":
        return asyncio.run(run_recall_drift(k=args.k))
    if args.cmd == "dashboard-serve":
        return run_dashboard_serve(host=args.host, port=args.port)
    if args.cmd == "plan-task":
        return asyncio.run(run_plan_task(
            goal=args.goal, out_path=args.out, mock=args.mock,
            reset=args.reset, max_concurrent=args.max_concurrent,
            user_id=args.user_id, workdir=args.workdir,
        ))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
