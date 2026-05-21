"""阶段 1 CLI 入口。

子命令：

- ``demo-phase1``：端到端 demo，跑 5 轮对话 → 提炼 → 检索（spec §11 阶段 1 验收）。
  ``--mock`` 用预设 stub 不打 API；不加默认走真实 Claude（需 ``ANTHROPIC_API_KEY``）。
- ``recall-baseline``：阶段 1 任务 1.11 召回质量摸底，输出 P@5。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from storage.memory_store import MemoryStore
from storage.transcript_store import TranscriptStore
from worker.agent import Agent, default_client
from worker.writeback import writeback_turn

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRANSCRIPT_DB = DATA_DIR / "transcript.db"
CHROMA_DIR = DATA_DIR / "chroma"

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

        result = await writeback_turn(
            transcript_store=transcript_store,
            memory_store=memory_store,
            agent=agent,
            user_id=user_id,
            task_id=task_id,
            conversation_id=conversation_id,
            turn_index=i,
            user_input=user_input,
            agent_output=agent_output,
        )
        if result.memory_id:
            print(f"[memory extracted] {result.memory_doc!r}  (mem_id={result.memory_id})")
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


def main() -> int:
    parser = argparse.ArgumentParser(prog="orchestrator")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_demo = sub.add_parser("demo-phase1", help="阶段 1 端到端 demo（spec §11）")
    p_demo.add_argument("--mock", action="store_true", help="用 stub LLM 不打 API")
    p_demo.add_argument("--reset", action="store_true", help="跑前清空 data/transcript.db & chroma")

    p_recall = sub.add_parser("recall-baseline", help="阶段 1 任务 1.11 召回质量摸底")
    p_recall.add_argument("-k", type=int, default=5)

    args = parser.parse_args()
    if args.cmd == "demo-phase1":
        return asyncio.run(run_demo_phase1(mock=args.mock, reset=args.reset))
    if args.cmd == "recall-baseline":
        return asyncio.run(run_recall_baseline(k=args.k))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
