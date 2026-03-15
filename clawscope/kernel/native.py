"""Native ClawScope kernel implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from clawscope.agent import ReActAgent
from clawscope.config import AgentConfig
from clawscope.kernel.loop import LoopConfig, NativeAgentLoop
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

    def create_loop(
        self,
        *,
        max_iterations: int | None = None,
        **kwargs: Any,
    ) -> NativeAgentLoop:
        """Create the kernel-owned native reasoning loop."""
        max_tokens = int(kwargs.pop("max_tokens", self.agent_config.max_tokens))
        return NativeAgentLoop(
            LoopConfig(
                max_iterations=max_iterations or self.agent_config.max_iterations,
                max_tokens=max_tokens,
            )
        )

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
        loop = kwargs.pop("loop", None) or self.create_loop(
            max_iterations=max_iterations,
            **kwargs,
        )
        max_tokens = int(kwargs.pop("max_tokens", self.agent_config.max_tokens))

        return ReActAgent(
            name=name or self.agent_config.name,
            sys_prompt=self.build_sys_prompt(sys_prompt),
            model=model,
            memory=memory or InMemoryMemory(),
            tools=self.tool_registry,
            max_iterations=max_iterations or self.agent_config.max_iterations,
            max_tokens=max_tokens,
            loop=loop,
            **kwargs,
        )


__all__ = ["NativeKernel"]
