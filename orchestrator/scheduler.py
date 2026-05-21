"""阶段 2/3 调度器：串行拓扑 + recovery + context_packer + writeback v2。

阶段 4a 起会扩展为 ``asyncio`` 并发 + 完整失败模型；本模块只覆盖
spec §11 阶段 2/3 验收所需的「串行 DAG + 精确接力」。
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Callable

from orchestrator.context_packer import ContextPacker
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
    packed_context: str | None = None


@dataclass
class TaskContext:
    task_id: str
    user_id: str
    title: str


SubTaskBuilder = Callable[[DagNodeRow, TaskContext], str]
"""返回该节点的「子任务说明」，将作为 context_packer 打包结果的 ``# 子任务`` 段。"""


class Scheduler:
    def __init__(
        self,
        *,
        state_store: StateStore,
        transcript_store: TranscriptStore,
        memory_store: MemoryStore,
        sandbox: SandboxBackend,
        llm_client: LLMClient,
        recovery: Recovery,
        context_packer: ContextPacker,
        sub_task_builder: SubTaskBuilder,
        heartbeat_interval: float = 30.0,
    ) -> None:
        self._state_store = state_store
        self._transcript_store = transcript_store
        self._memory_store = memory_store
        self._sandbox = sandbox
        self._llm = llm_client
        self._recovery = recovery
        self._packer = context_packer
        self._sub_task_builder = sub_task_builder
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
                pending_left = [n for n in nodes if n.status == "pending"]
                raise RuntimeError(
                    f"deadlock: {len(pending_left)} pending node(s) have unmet deps"
                )
            for node in ready:  # 阶段 2/3 串行；阶段 4a 改并发
                await self._execute_node(node, nodes, ctx)

        final = self._task_final_status(
            await self._state_store.list_dag_nodes(task_id)
        )
        await self._state_store.update_task_status(task_id, final)
        return final

    async def _execute_node(
        self, node: DagNodeRow, all_nodes: list[DagNodeRow], ctx: TaskContext
    ) -> NodeRunOutcome:
        # spec §3.3 / §8.1：按 depends_on 顺序填 input_memory_ids（含 None 占位）
        node_map = {n.id: n for n in all_nodes}
        upstream_mids: list[str | None] = []
        for dep_id in node.depends_on:
            dep = node_map.get(dep_id)
            if (
                dep is not None
                and dep.status == "done"
                and dep.output_memory_id is not None
            ):
                upstream_mids.append(dep.output_memory_id)
            else:
                upstream_mids.append(None)
        await self._state_store.set_node_input_memory_ids(node.id, upstream_mids)

        worker_id = f"w_{uuid.uuid4().hex[:8]}"
        if not await self._state_store.claim_node_running(node.id, worker_id):
            current = await self._state_store.get_dag_node(node.id)
            return NodeRunOutcome(
                node_id=node.id,
                final_status=current.status if current else "unknown",
                output_memory_id=current.output_memory_id if current else None,
                agent_output=None,
            )

        # 重新读 node 以拿到刚写入的 input_memory_ids
        node = await self._state_store.get_dag_node(node.id) or node

        sub_task = self._sub_task_builder(node, ctx)
        packed = await self._packer.pack(
            task_id=ctx.task_id,
            node_id=node.id,
            sub_task_description=sub_task,
        )
        handle = await self._sandbox.create(context_package=packed.text)

        agent = Agent(agent_id=node.node_name, client=self._llm)
        try:
            async with HeartbeatTask(
                self._state_store, node.id, interval=self._heartbeat_interval
            ):
                agent_output = await agent.respond([], packed.text)
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
                    user_input=packed.text,
                    agent_output=agent_output,
                )
                if not result.committed:
                    _log.warning("commit aborted for node=%s", node.id)
                    return NodeRunOutcome(
                        node_id=node.id,
                        final_status="aborted",
                        output_memory_id=None,
                        agent_output=agent_output,
                        packed_context=packed.text,
                    )
                return NodeRunOutcome(
                    node_id=node.id,
                    final_status="done",
                    output_memory_id=result.memory_id,
                    agent_output=agent_output,
                    packed_context=packed.text,
                )
        except Exception as e:  # noqa: BLE001
            _log.exception("node %s raised: %s", node.id, e)
            await self._state_store.mark_node_terminal(node.id, "failed")
            return NodeRunOutcome(
                node_id=node.id,
                final_status="failed",
                output_memory_id=None,
                agent_output=None,
                packed_context=packed.text,
            )
        finally:
            await self._sandbox.destroy(handle)

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
        for n in nodes:
            if n.status == "failed":
                return "failed"
        return "done"
