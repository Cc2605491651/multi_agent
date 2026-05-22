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

    def is_empty(self) -> bool:
        return (
            self.model is None
            and self.provider is None
            and self.system_prompt is None
            and not self.tools
            and not self.skills
            and not self.mcp_servers
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "system_prompt": self.system_prompt,
            "tools": [t.to_dict() for t in self.tools],
            "skills": [s.to_dict() for s in self.skills],
            "mcp_servers": [m.to_dict() for m in self.mcp_servers],
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
        return cls(
            model=data.get("model"),
            provider=data.get("provider"),
            system_prompt=data.get("system_prompt"),
            tools=[ToolSpec.from_obj(t) for t in (data.get("tools") or [])],
            skills=[SkillSpec.from_obj(s) for s in (data.get("skills") or [])],
            mcp_servers=[
                MCPServerSpec.from_obj(m) for m in (data.get("mcp_servers") or [])
            ],
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
