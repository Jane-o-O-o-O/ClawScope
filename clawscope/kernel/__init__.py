"""Kernel integration helpers for ClawScope."""

from pathlib import Path

from clawscope.config import AgentConfig, ModelConfig
from clawscope.model import ModelRegistry
from clawscope.tool import ToolRegistry

from clawscope.kernel.base import AgentKernel
from clawscope.kernel.agentscope import (
    AgentScopeKernel,
    AgentScopeKernelError,
    AgentScopeReActAgent,
    create_agentscope_react_agent,
)
from clawscope.kernel.native import NativeKernel


def build_kernel(
    *,
    agent_config: AgentConfig,
    model_config: ModelConfig,
    model_registry: ModelRegistry,
    tool_registry: ToolRegistry,
    workspace: Path,
) -> AgentKernel:
    """Build the configured kernel implementation."""
    if agent_config.kernel == "agentscope":
        return AgentScopeKernel(
            agent_config=agent_config,
            model_config=model_config,
            tool_registry=tool_registry,
            workspace=workspace,
        )

    return NativeKernel(
        agent_config=agent_config,
        tool_registry=tool_registry,
        model_registry=model_registry,
        workspace=workspace,
    )

__all__ = [
    "AgentKernel",
    "AgentScopeKernel",
    "AgentScopeKernelError",
    "AgentScopeReActAgent",
    "NativeKernel",
    "build_kernel",
    "create_agentscope_react_agent",
]
