"""Base abstractions for pluggable agent kernels."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from clawscope.agent import AgentBase
from clawscope.config import AgentConfig
from clawscope.memory import MemoryBase
from clawscope.tool import ToolRegistry
from clawscope.workspace_prompt import WorkspacePromptBuilder


class AgentKernel(ABC):
    """Factory interface for creating agents backed by a concrete kernel."""

    def __init__(
        self,
        agent_config: AgentConfig,
        tool_registry: ToolRegistry,
        workspace: Path,
    ) -> None:
        self.agent_config = agent_config
        self.tool_registry = tool_registry
        self.workspace = workspace
        self.prompt_builder = WorkspacePromptBuilder(workspace)

    def build_sys_prompt(self, base_prompt: str | None = None) -> str:
        """Build the kernel's final system prompt."""
        return self.prompt_builder.build(base_prompt or self.agent_config.sys_prompt)

    @abstractmethod
    def create_agent(
        self,
        *,
        name: str | None = None,
        sys_prompt: str | None = None,
        memory: MemoryBase | None = None,
        max_iterations: int | None = None,
        **kwargs: Any,
    ) -> AgentBase:
        """Create an agent instance for this kernel."""


__all__ = ["AgentKernel"]
