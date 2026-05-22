"""Agent Harness 数据模型（阶段 ABC.A.2）。

「Harness」= 包裹 LLM 让它真正能干活的所有运行时配置：

- ``model`` / ``provider``：用哪家哪款模型
- ``system_prompt``：节点级系统提示词覆盖
- ``tools``：可调用的工具列表（B 段真实化）
- ``skills``：技能包（C 段真实化），markdown 风格指令包
- ``mcp_servers``：要在 sandbox 里启动的 MCP server 进程（C 段真实化）

阶段 A 只把 schema 一次到位；B/C 才把每一项「真跑起来」。schema 不再改。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    """工具声明。阶段 A 只是结构；阶段 B 内置工具按 ``name`` 路由实现。"""

    name: str
    description: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "params": dict(self.params)}

    @classmethod
    def from_obj(cls, obj: Any) -> "ToolSpec":
        if isinstance(obj, str):
            return cls(name=obj)
        if isinstance(obj, dict):
            return cls(
                name=str(obj["name"]),
                description=str(obj.get("description", "")),
                params=dict(obj.get("params", {})),
            )
        raise TypeError(f"tool spec must be str or dict, got {type(obj).__name__}")


@dataclass(frozen=True)
class SkillSpec:
    """Claude Code 风格的技能包声明。阶段 C 落地加载逻辑。"""

    name: str
    description: str = ""
    instructions_path: str | None = None
    invoke_keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "instructions_path": self.instructions_path,
            "invoke_keywords": list(self.invoke_keywords),
        }

    @classmethod
    def from_obj(cls, obj: Any) -> "SkillSpec":
        if isinstance(obj, str):
            return cls(name=obj)
        if isinstance(obj, dict):
            return cls(
                name=str(obj["name"]),
                description=str(obj.get("description", "")),
                instructions_path=obj.get("instructions_path"),
                invoke_keywords=list(obj.get("invoke_keywords") or []),
            )
        raise TypeError(f"skill spec must be str or dict, got {type(obj).__name__}")


@dataclass(frozen=True)
class HandoffSpec:
    """节点级接力点（spec v6 §9.8）：从指定上游节点的 transcript 拿原始多轮对话。

    - ``from_node``：DAG JSON 里写逻辑 id（如 ``"n1"``）；
      ``dag_loader.instantiate_dag`` 实例化时翻译成真实 ``node_id``
    - ``turn_range``：``[start, end]``（含两端）；``None`` 表示取该 conv 全部 turn
    """

    from_node: str
    turn_range: list[int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_node": self.from_node,
            "turn_range": list(self.turn_range) if self.turn_range else None,
        }

    @classmethod
    def from_obj(cls, obj: Any) -> "HandoffSpec":
        if not isinstance(obj, dict):
            raise TypeError(f"handoff must be a dict, got {type(obj).__name__}")
        from_node = obj.get("from_node")
        if not isinstance(from_node, str) or not from_node:
            raise ValueError("handoff.from_node is required and must be a non-empty string")
        tr = obj.get("turn_range")
        if tr is not None:
            if not isinstance(tr, list) or len(tr) != 2:
                raise ValueError("handoff.turn_range must be a [start, end] list")
            if not all(isinstance(x, int) and x > 0 for x in tr):
                raise ValueError("handoff.turn_range elements must be positive ints")
            if tr[0] > tr[1]:
                raise ValueError(f"handoff.turn_range start({tr[0]}) > end({tr[1]})")
        return cls(from_node=from_node, turn_range=tr)


@dataclass(frozen=True)
class MCPServerSpec:
    """sandbox 内启动的 MCP server。阶段 C 落地握手 + tool 路由。"""

    name: str
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "command": self.command,
            "args": list(self.args),
            "env": dict(self.env),
        }

    @classmethod
    def from_obj(cls, obj: Any) -> "MCPServerSpec":
        if isinstance(obj, str):
            return cls(name=obj)
        if isinstance(obj, dict):
            return cls(
                name=str(obj["name"]),
                command=str(obj.get("command", "")),
                args=list(obj.get("args") or []),
                env=dict(obj.get("env") or {}),
            )
        raise TypeError(f"mcp server spec must be str or dict, got {type(obj).__name__}")


@dataclass(frozen=True)
class AgentHarness:
    model: str | None = None
    provider: str | None = None
    system_prompt: str | None = None
    tools: list[ToolSpec] = field(default_factory=list)
    skills: list[SkillSpec] = field(default_factory=list)
    mcp_servers: list[MCPServerSpec] = field(default_factory=list)
    handoff: HandoffSpec | None = None

    def is_empty(self) -> bool:
        return (
            self.model is None
            and self.provider is None
            and self.system_prompt is None
            and not self.tools
            and not self.skills
            and not self.mcp_servers
            and self.handoff is None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "system_prompt": self.system_prompt,
            "tools": [t.to_dict() for t in self.tools],
            "skills": [s.to_dict() for s in self.skills],
            "mcp_servers": [m.to_dict() for m in self.mcp_servers],
            "handoff": self.handoff.to_dict() if self.handoff else None,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Any) -> "AgentHarness":
        if data is None:
            return cls()
        if isinstance(data, str):
            data = json.loads(data) if data else {}
        if not isinstance(data, dict):
            raise TypeError(f"harness must be dict, got {type(data).__name__}")
        handoff = HandoffSpec.from_obj(data["handoff"]) if data.get("handoff") else None
        return cls(
            model=data.get("model"),
            provider=data.get("provider"),
            system_prompt=data.get("system_prompt"),
            tools=[ToolSpec.from_obj(t) for t in (data.get("tools") or [])],
            skills=[SkillSpec.from_obj(s) for s in (data.get("skills") or [])],
            mcp_servers=[
                MCPServerSpec.from_obj(m) for m in (data.get("mcp_servers") or [])
            ],
            handoff=handoff,
        )

    @classmethod
    def from_legacy(
        cls,
        *,
        model_name: str | None = None,
        tools: list[str] | None = None,
    ) -> "AgentHarness":
        """老式平铺字段（``model_name`` + ``tools: list[str]``）兼容构造。"""
        return cls(
            model=model_name,
            tools=[ToolSpec(name=t) for t in (tools or [])],
        )
