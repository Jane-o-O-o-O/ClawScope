"""Kernel integration helpers for ClawScope."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

from clawscope.config import AgentConfig, ModelConfig
from clawscope.model import ModelRegistry
from clawscope.tool import ToolRegistry

from clawscope.kernel.base import AgentKernel
from clawscope.kernel.loop import AgentLoop, LoopConfig, NativeAgentLoop

__all__ = [
    "AgentKernel",
    "AgentLoop",
    "AgentScopeAgentLoop",
    "AgentScopeKernel",
    "AgentScopeKernelError",
    "AgentScopeReActAgent",
    "KernelAgent",
    "LoopConfig",
    "NativeAgentLoop",
    "NativeKernel",
    "build_kernel",
    "create_agentscope_react_agent",
]


_LAZY_IMPORTS = {
    "KernelAgent": ("clawscope.kernel.agent", "KernelAgent"),
    "AgentScopeAgentLoop": ("clawscope.kernel.agentscope", "AgentScopeAgentLoop"),
    "AgentScopeKernel": ("clawscope.kernel.agentscope", "AgentScopeKernel"),
    "AgentScopeKernelError": ("clawscope.kernel.agentscope", "AgentScopeKernelError"),
    "AgentScopeReActAgent": ("clawscope.kernel.agentscope", "AgentScopeReActAgent"),
    "create_agentscope_react_agent": (
        "clawscope.kernel.agentscope",
        "create_agentscope_react_agent",
    ),
    "NativeKernel": ("clawscope.kernel.native", "NativeKernel"),
}


def __getattr__(name: str) -> Any:
    """Lazily import kernel implementations to avoid circular imports."""
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module 'clawscope.kernel' has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


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
        AgentScopeKernel = __getattr__("AgentScopeKernel")
        return AgentScopeKernel(
            agent_config=agent_config,
            model_config=model_config,
            tool_registry=tool_registry,
            workspace=workspace,
        )

    NativeKernel = __getattr__("NativeKernel")
    return NativeKernel(
        agent_config=agent_config,
        tool_registry=tool_registry,
        model_registry=model_registry,
        workspace=workspace,
    )
