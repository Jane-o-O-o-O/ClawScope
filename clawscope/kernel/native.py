"""Native ClawScope kernel implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from clawscope.agent import ReActAgent
from clawscope.config import AgentConfig
from clawscope.memory import InMemoryMemory, MemoryBase
from clawscope.model import ModelRegistry
from clawscope.tool import ToolRegistry

from clawscope.kernel.base import AgentKernel


class NativeKernel(AgentKernel):
    """Kernel that uses ClawScope's built-in ReAct agent."""

    def __init__(
        self,
        agent_config: AgentConfig,
        tool_registry: ToolRegistry,
        model_registry: ModelRegistry,
        workspace: Path,
    ) -> None:
        super().__init__(
            agent_config=agent_config,
            tool_registry=tool_registry,
            workspace=workspace,
        )
        self.model_registry = model_registry

    def create_agent(
        self,
        *,
        name: str | None = None,
        sys_prompt: str | None = None,
        memory: MemoryBase | None = None,
        max_iterations: int | None = None,
        **kwargs: Any,
    ) -> ReActAgent:
        """Create a native ClawScope ReAct agent."""
        model = kwargs.pop("model", None) or self.model_registry.get_model()

        return ReActAgent(
            name=name or self.agent_config.name,
            sys_prompt=self.build_sys_prompt(sys_prompt),
            model=model,
            memory=memory or InMemoryMemory(),
            tools=self.tool_registry,
            max_iterations=max_iterations or self.agent_config.max_iterations,
            **kwargs,
        )


__all__ = ["NativeKernel"]
