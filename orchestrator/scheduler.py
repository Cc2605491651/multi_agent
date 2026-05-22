"""阶段 4a 并发调度器（spec v4 §4.1 / §5 / §6 / §8）。

主循环：

- 启动前跑一次 ``recovery.sweep_all``
- 每轮：扫所有 ready 节点 → 用 ``asyncio.Semaphore(MAX_CONCURRENT_WORKERS)``
  并发拉起，每节点一个 ``asyncio.Task``
- 节点失败 → ``FailureHandler.on_node_failed`` 决定重试 / skip / fail
- ``fail_fast`` 重试耗尽 → 对所有 active 兄弟 ``sandbox.cancel``，5s 超时改 ``destroy``
- 任务级失败 → 进入「收尾期」：等所有 active 收口，新一轮不再拉起，下游 pending 节点
  按 spec §5.2 表全部标 ``skipped``

阶段 4c 升级 context_packer 完整版 + token budget；阶段 4b 接 E2BBackend。
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass
from typing import Callable

from orchestrator.context_packer import ContextPacker
from orchestrator.failure_handler import FailureHandler
from orchestrator.recovery import Recovery
from storage.memory_store import MemoryStore
from storage.state_store import DagNodeRow, StateStore
from storage.transcript_store import TranscriptStore
from worker.agent import Agent, LLMClient
from worker.harness import AgentHarness
from worker.heartbeat import HeartbeatTask
from worker.llm_clients import make_llm_client
from worker.mcp_client import close_all as mcp_close_all
from worker.mcp_client import connect_all as mcp_connect_all
from worker.sandbox import SandboxBackend, SandboxHandle
from worker.skills import SkillLoader
from worker.tools import ToolRegistry
from worker.writeback import writeback_turn

_log = logging.getLogger(__name__)

DEFAULT_MAX_CONCURRENT = 5  # spec §4.1
CANCEL_TIMEOUT_SECONDS = 5.0  # spec §5.3


@dataclass
class NodeRunOutcome:
    node_id: str
    final_status: str  # done/failed/skipped/cancelled
    output_memory_id: str | None
    agent_output: str | None
    packed_context: str | None = None
    error: str | None = None


@dataclass
class TaskContext:
    task_id: str
    user_id: str
    title: str


SubTaskBuilder = Callable[[DagNodeRow, TaskContext], str]


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
        failure_handler: FailureHandler,
        sub_task_builder: SubTaskBuilder,
        max_concurrent_workers: int = DEFAULT_MAX_CONCURRENT,
        heartbeat_interval: float = 30.0,
        cancel_timeout: float = CANCEL_TIMEOUT_SECONDS,
        default_provider: str | None = None,
        force_default_client: bool = False,
        skill_loader: SkillLoader | None = None,
    ) -> None:
        self._state_store = state_store
        self._transcript_store = transcript_store
        self._memory_store = memory_store
        self._sandbox = sandbox
        self._llm = llm_client
        self._recovery = recovery
        self._packer = context_packer
        self._failure = failure_handler
        self._sub_task_builder = sub_task_builder
        self._semaphore = asyncio.Semaphore(max_concurrent_workers)
        self._heartbeat_interval = heartbeat_interval
        self._cancel_timeout = cancel_timeout
        # scheduler 默认 provider 标签：节点 harness 的 provider 匹配时直接复用
        # ``self._llm``（避免在 mock 模式下被 make_llm_client 重新构造打到真实 API）
        self._default_provider = (
            default_provider or os.environ.get("LLM_PROVIDER", "anthropic")
        ).strip().lower()
        # mock / 集成测试场景：忽略 harness.provider，永远用 self._llm
        self._force_default = force_default_client
        # 节点级 provider 切换时的 client 缓存
        self._provider_client_cache: dict[str, LLMClient] = {}
        # 阶段 C：skill 加载器（指令包注入）
        self._skill_loader = skill_loader or SkillLoader()

    async def run_task(self, task_id: str) -> str:
        await self._recovery.sweep_all()

        task = await self._state_store.get_task(task_id)
        if task is None:
            raise ValueError(f"task not found: {task_id}")
        await self._state_store.update_task_status(task_id, "running")
        ctx = TaskContext(task_id=task_id, user_id=task.user_id, title=task.title)

        # 持有 active node 的 asyncio.Task + sandbox handle，便于 fail_fast 取消
        active_tasks: dict[str, asyncio.Task] = {}
        active_handles: dict[str, SandboxHandle] = {}
        cancelled = False
        outcomes: dict[str, NodeRunOutcome] = {}

        while True:
            if cancelled:
                # 任务级失败：把所有 pending 下游 cascade 到 skipped
                await self._cascade_skip_downstream(task_id)

            nodes = await self._state_store.list_dag_nodes(task_id)
            if self._all_terminal(nodes) and not active_tasks:
                break

            if not cancelled:
                ready = self._find_ready(nodes)
                for node in ready:
                    if node.id in active_tasks:
                        continue
                    t = asyncio.create_task(
                        self._run_one_node(node, ctx, active_handles),
                        name=f"node:{node.id}",
                    )
                    active_tasks[node.id] = t

                if not active_tasks:
                    pending_left = [n for n in nodes if n.status == "pending"]
                    if pending_left:
                        raise RuntimeError(
                            f"deadlock: {len(pending_left)} pending node(s) have unmet deps"
                        )
                    break

            if not active_tasks:
                # cancelled 模式 + 没活的 task → cascade 已跑完，下一轮顶部 break
                continue

            done, _pending = await asyncio.wait(
                active_tasks.values(), return_when=asyncio.FIRST_COMPLETED
            )

            for finished_task in done:
                node_id = self._task_to_node_id(active_tasks, finished_task)
                if node_id is None:
                    continue
                active_tasks.pop(node_id, None)
                active_handles.pop(node_id, None)

                try:
                    outcome = finished_task.result()
                except asyncio.CancelledError:
                    _log.info("node task %s was cancelled", node_id)
                    # cancel 路径下 _execute_node 没机会落终态，DB 里仍 running
                    db_node = await self._state_store.get_dag_node(node_id)
                    if db_node and db_node.status == "running":
                        await self._state_store.mark_node_terminal(
                            node_id, "skipped"
                        )
                    outcomes[node_id] = NodeRunOutcome(
                        node_id=node_id,
                        final_status="cancelled",
                        output_memory_id=None,
                        agent_output=None,
                    )
                    continue
                except Exception as e:  # noqa: BLE001
                    _log.exception("node task %s crashed: %s", node_id, e)
                    outcome = NodeRunOutcome(
                        node_id=node_id,
                        final_status="failed",
                        output_memory_id=None,
                        agent_output=None,
                        error=str(e),
                    )

                outcomes[node_id] = outcome

                if outcome.final_status == "failed":
                    failed_node = await self._state_store.get_dag_node(node_id)
                    if failed_node is None:
                        continue
                    action = await self._failure.on_node_failed(failed_node)

                    if action.cancel_siblings and not cancelled:
                        cancelled = True
                        _log.info(
                            "fail_fast cancellation triggered by node=%s; "
                            "cancelling %d siblings",
                            node_id,
                            len(active_tasks),
                        )
                        await self._cancel_active(active_tasks, active_handles)

                    if action.task_should_fail and not cancelled:
                        # 标 task failed 但不取消兄弟（fail_retry 重试耗尽路径）
                        # 兄弟跑完即停（下一轮 cancelled=False 但 ready=空 + 下游全部 skipped）
                        cancelled = True
                        _log.info(
                            "task-level failure from node=%s; entering drain mode",
                            node_id,
                        )

                # done / skipped 路径正常走，下一轮调度

        final = self._task_final_status(
            await self._state_store.list_dag_nodes(task_id)
        )
        await self._state_store.update_task_status(task_id, final)
        return final

    @staticmethod
    def _task_to_node_id(
        active_tasks: dict[str, asyncio.Task], finished: asyncio.Task
    ) -> str | None:
        for nid, t in active_tasks.items():
            if t is finished:
                return nid
        return None

    # ---- 单节点执行 ----

    async def _run_one_node(
        self,
        node: DagNodeRow,
        ctx: TaskContext,
        active_handles: dict[str, SandboxHandle],
    ) -> NodeRunOutcome:
        async with self._semaphore:
            return await self._execute_node(node, ctx, active_handles)

    async def _execute_node(
        self,
        node: DagNodeRow,
        ctx: TaskContext,
        active_handles: dict[str, SandboxHandle],
    ) -> NodeRunOutcome:
        # 按 depends_on 顺序填 input_memory_ids（含 None 占位，spec §8.1）
        all_nodes = await self._state_store.list_dag_nodes(ctx.task_id)
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

        node = await self._state_store.get_dag_node(node.id) or node
        sub_task = self._sub_task_builder(node, ctx)
        packed = await self._packer.pack(
            task_id=ctx.task_id,
            node_id=node.id,
            sub_task_description=sub_task,
        )
        handle = await self._sandbox.create(context_package=packed.text)
        active_handles[node.id] = handle
        # finally 块需要 mcp_clients 可见，先预声明（即使后续 harness 解析抛错也能 close）
        mcp_clients: list = []

        try:
            # 解析节点 harness：覆盖 provider / model / system_prompt + skills + mcp
            harness = (
                AgentHarness.from_dict(node.harness_json)
                if node.harness_json else AgentHarness()
            )
            client = self._client_for_provider(harness.provider)
            chat_model = harness.model or node.model_name

            # 阶段 C：skill 注入 system_prompt
            final_system, _ = self._skill_loader.apply(
                harness.skills, sub_task, harness.system_prompt
            )

            agent_kwargs: dict = {"agent_id": node.node_name, "client": client}
            if chat_model:
                agent_kwargs["chat_model"] = chat_model
            if final_system:
                agent_kwargs["system_prompt"] = final_system
            agent = Agent(**agent_kwargs)

            # 阶段 B+C：合并 builtin tools + MCP tools 进 ToolRegistry
            mcp_tools_list: list = []
            if harness.mcp_servers:
                mcp_clients, mcp_tools_list = await mcp_connect_all(harness.mcp_servers)

            registry: ToolRegistry | None = None
            if harness.tools or mcp_tools_list:
                registry = ToolRegistry.from_specs(harness.tools)
                for t in mcp_tools_list:
                    registry.tools[t.name] = t

            async with HeartbeatTask(
                self._state_store, node.id, interval=self._heartbeat_interval
            ):
                if registry and registry.names() and self._can_use_tools(agent):
                    loop_res = await agent.run_with_tools(
                        packed.text,
                        registry=registry,
                        sandbox=self._sandbox,
                        handle=handle,
                    )
                    agent_output = loop_res.final_text
                else:
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
        except asyncio.CancelledError:
            _log.info("node %s execution cancelled mid-flight", node.id)
            # 让 finally 跑 destroy
            raise
        except Exception as e:  # noqa: BLE001
            _log.exception("node %s raised: %s", node.id, e)
            return NodeRunOutcome(
                node_id=node.id,
                final_status="failed",
                output_memory_id=None,
                agent_output=None,
                packed_context=packed.text,
                error=str(e),
            )
        finally:
            if mcp_clients:
                await mcp_close_all(mcp_clients)
            try:
                await self._sandbox.destroy(handle)
            except Exception as e:  # noqa: BLE001
                _log.warning("destroy handle for node=%s failed: %s", node.id, e)

    # ---- 取消并发兄弟 ----

    async def _cancel_active(
        self,
        active_tasks: dict[str, asyncio.Task],
        active_handles: dict[str, SandboxHandle],
    ) -> None:
        """spec §5.3：先调 sandbox.cancel，超时改 destroy 强杀。"""
        for nid, handle in list(active_handles.items()):
            try:
                ok = await self._sandbox.cancel(handle, timeout=self._cancel_timeout)
                if not ok:
                    _log.warning(
                        "cancel timed out for node=%s; forcing destroy", nid
                    )
                    await self._sandbox.destroy(handle)
            except Exception as e:  # noqa: BLE001
                _log.warning("cancel/destroy node=%s failed: %s", nid, e)
        # 同时 cancel 兄弟 asyncio.Task（让 await 点抛出 CancelledError）
        for t in active_tasks.values():
            t.cancel()

    # ---- cascade skip ----

    async def _cascade_skip_downstream(self, task_id: str) -> None:
        """任务级失败后：所有 pending 节点链式标 skipped。"""
        # 多轮直到稳定（一个节点 skip 后可能解锁下游可 skip）
        while True:
            nodes = await self._state_store.list_dag_nodes(task_id)
            changed = False
            for n in nodes:
                if n.status != "pending":
                    continue
                await self._failure.on_dependency_terminal_failed(n)
                changed = True
            if not changed:
                break

    # ---- DAG helpers ----

    @staticmethod
    def _can_use_tools(agent: Agent) -> bool:
        """判定 agent.client 是否支持 tool-use loop。"""
        from worker.agent import AnthropicClient
        from worker.llm_clients import OpenAICompatibleClient

        return isinstance(agent.client, (AnthropicClient, OpenAICompatibleClient))

    def _client_for_provider(self, provider: str | None) -> LLMClient:
        if self._force_default or not provider:
            return self._llm
        if provider.strip().lower() == self._default_provider:
            return self._llm
        if provider in self._provider_client_cache:
            return self._provider_client_cache[provider]
        client = make_llm_client(provider)
        self._provider_client_cache[provider] = client
        return client

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
