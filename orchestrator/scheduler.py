"""阶段 2 最小调度器：串行拓扑 + recovery + 单节点 writeback v2。

阶段 4a 起会扩展为 ``asyncio`` 并发 + 完整失败模型；本模块只解决
spec §11 阶段 2 验收 demo 所需的「串行 2 节点 DAG」。
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Awaitable, Callable

from orchestrator.recovery import Recovery
from storage.memory_store import MemoryStore
from storage.state_store import DagNodeRow, StateStore
from storage.transcript_store import TranscriptStore
from worker.agent import Agent, LLMClient
from worker.heartbeat import HeartbeatTask
from worker.sandbox import SandboxBackend
from worker.writeback import writeback_turn

_log = logging.getLogger(__name__)


@dataclass
class NodeRunOutcome:
    node_id: str
    final_status: str
    output_memory_id: str | None
    agent_output: str | None


PromptBuilder = Callable[[DagNodeRow, list[dict], "TaskContext"], str]


@dataclass
class TaskContext:
    task_id: str
    user_id: str
    title: str


class Scheduler:
    """串行执行 DAG。阶段 4a 升级为并发，但接口预留。"""

    def __init__(
        self,
        *,
        state_store: StateStore,
        transcript_store: TranscriptStore,
        memory_store: MemoryStore,
        sandbox: SandboxBackend,
        llm_client: LLMClient,
        recovery: Recovery,
        prompt_builder: PromptBuilder,
        heartbeat_interval: float = 30.0,
    ) -> None:
        self._state_store = state_store
        self._transcript_store = transcript_store
        self._memory_store = memory_store
        self._sandbox = sandbox
        self._llm = llm_client
        self._recovery = recovery
        self._prompt_builder = prompt_builder
        self._heartbeat_interval = heartbeat_interval

    async def run_task(self, task_id: str) -> str:
        await self._recovery.sweep_all()

        task = await self._state_store.get_task(task_id)
        if task is None:
            raise ValueError(f"task not found: {task_id}")
        await self._state_store.update_task_status(task_id, "running")
        ctx = TaskContext(task_id=task_id, user_id=task.user_id, title=task.title)

        while True:
            nodes = await self._state_store.list_dag_nodes(task_id)
            if self._all_terminal(nodes):
                break
            ready = self._find_ready(nodes)
            if not ready:
                # 阶段 2 不处理「卡死」场景；阶段 4a 补
                pending_left = [n for n in nodes if n.status == "pending"]
                raise RuntimeError(
                    f"deadlock: {len(pending_left)} pending node(s) have unmet deps"
                )
            for node in ready:  # 串行
                await self._execute_node(node, nodes, ctx)

        final = self._task_final_status(
            await self._state_store.list_dag_nodes(task_id)
        )
        await self._state_store.update_task_status(task_id, final)
        return final

    # ---- node execution ----

    async def _execute_node(
        self, node: DagNodeRow, all_nodes: list[DagNodeRow], ctx: TaskContext
    ) -> NodeRunOutcome:
        # 阶段 3 接力点 + input_memory_ids：本阶段简单收集上游 done 节点的 output_memory_id
        upstream_mids = [
            n.output_memory_id
            for n in all_nodes
            if n.id in node.depends_on
            and n.status == "done"
            and n.output_memory_id is not None
        ]
        await self._state_store.set_node_input_memory_ids(node.id, upstream_mids)

        worker_id = f"w_{uuid.uuid4().hex[:8]}"
        if not await self._state_store.claim_node_running(node.id, worker_id):
            # 别处已经接管
            current = await self._state_store.get_dag_node(node.id)
            return NodeRunOutcome(
                node_id=node.id,
                final_status=current.status if current else "unknown",
                output_memory_id=current.output_memory_id if current else None,
                agent_output=None,
            )

        input_mems = (
            await self._memory_store.get_by_ids(ctx.user_id, upstream_mids)
            if upstream_mids
            else []
        )

        # 重新读 node 以拿到最新 input_memory_ids（也可直接用 upstream_mids，但保持一致）
        node = await self._state_store.get_dag_node(node.id) or node

        user_input = self._prompt_builder(node, input_mems, ctx)
        context_package = self._format_context_package(node, input_mems, ctx, user_input)
        handle = await self._sandbox.create(context_package=context_package)

        agent = Agent(agent_id=node.node_name, client=self._llm)
        try:
            async with HeartbeatTask(
                self._state_store, node.id, interval=self._heartbeat_interval
            ):
                agent_output = await agent.respond([], user_input)
                conv_id = f"conv_{ctx.task_id}_{node.id}"
                result = await writeback_turn(
                    transcript_store=self._transcript_store,
                    memory_store=self._memory_store,
                    state_store=self._state_store,
                    agent=agent,
                    user_id=ctx.user_id,
                    task_id=ctx.task_id,
                    node_id=node.id,
                    conversation_id=conv_id,
                    turn_index=1,
                    user_input=user_input,
                    agent_output=agent_output,
                )
                if not result.committed:
                    _log.warning("commit aborted for node=%s", node.id)
                    return NodeRunOutcome(
                        node_id=node.id,
                        final_status="aborted",
                        output_memory_id=None,
                        agent_output=agent_output,
                    )
                return NodeRunOutcome(
                    node_id=node.id,
                    final_status="done",
                    output_memory_id=result.memory_id,
                    agent_output=agent_output,
                )
        except Exception as e:  # noqa: BLE001
            # 阶段 2 简化：失败直接 mark failed；阶段 4a 替换为完整失败模型
            _log.exception("node %s raised: %s", node.id, e)
            await self._state_store.mark_node_terminal(node.id, "failed")
            return NodeRunOutcome(
                node_id=node.id,
                final_status="failed",
                output_memory_id=None,
                agent_output=None,
            )
        finally:
            await self._sandbox.destroy(handle)

    # ---- DAG helpers ----

    @staticmethod
    def _all_terminal(nodes: list[DagNodeRow]) -> bool:
        terminal = {"done", "failed", "skipped"}
        return all(n.status in terminal for n in nodes)

    @staticmethod
    def _find_ready(nodes: list[DagNodeRow]) -> list[DagNodeRow]:
        done_or_skipped = {
            n.id for n in nodes if n.status in {"done", "skipped"}
        }
        ready: list[DagNodeRow] = []
        for n in nodes:
            if n.status != "pending":
                continue
            if all(dep in done_or_skipped for dep in n.depends_on):
                ready.append(n)
        return ready

    @staticmethod
    def _task_final_status(nodes: list[DagNodeRow]) -> str:
        # 任一失败 → 任务 failed；否则 done（skipped 不算失败）
        for n in nodes:
            if n.status == "failed":
                return "failed"
        return "done"

    @staticmethod
    def _format_context_package(
        node: DagNodeRow,
        input_mems: list[dict],
        ctx: TaskContext,
        user_input: str,
    ) -> str:
        lines = [
            f"# 任务主题",
            f"{ctx.title}",
            "",
            f"# 本节点",
            f"id={node.id}  name={node.node_name}",
            "",
            "# 上游产出",
        ]
        if not input_mems:
            lines.append("（无）")
        else:
            for m in input_mems:
                lines.append(f"- {m['document']}")
        lines.extend(["", "# 用户输入", user_input])
        return "\n".join(lines)
