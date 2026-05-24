"""ClawScope agent system."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from clawscope.agent.base import AgentBase

__all__ = [
    "AgentBase",
    "ReActAgent",
    "OrchestratorAgent",
    "create_orchestrator",
    "UserAgent",
    "ChannelUserAgent",
    "RealtimeAgent",
    "A2AAgent",
    "A2AMessage",
    "A2AMessageType",
    "A2ARouter",
    "AgentCard",
    "AgentCapability",
    "get_router",
    "RealtimeConnection",
    "OpenAIRealtimeConnection",
    "AudioProvider",
    "MicrophoneProvider",
    "AudioConfig",
]


_LAZY_IMPORTS = {
    "ReActAgent": ("clawscope.agent.react", "ReActAgent"),
    "OrchestratorAgent": ("clawscope.agent.orchestrator", "OrchestratorAgent"),
    "create_orchestrator": ("clawscope.agent.orchestrator", "create_orchestrator"),
    "UserAgent": ("clawscope.agent.user", "UserAgent"),
    "ChannelUserAgent": ("clawscope.agent.user", "ChannelUserAgent"),
    "RealtimeAgent": ("clawscope.agent.realtime", "RealtimeAgent"),
    "A2AAgent": ("clawscope.agent.a2a", "A2AAgent"),
    "A2AMessage": ("clawscope.agent.a2a", "A2AMessage"),
    "A2AMessageType": ("clawscope.agent.a2a", "A2AMessageType"),
    "A2ARouter": ("clawscope.agent.a2a", "A2ARouter"),
    "AgentCard": ("clawscope.agent.a2a", "AgentCard"),
    "AgentCapability": ("clawscope.agent.a2a", "AgentCapability"),
    "get_router": ("clawscope.agent.a2a", "get_router"),
    "RealtimeConnection": ("clawscope.agent.realtime", "RealtimeConnection"),
    "OpenAIRealtimeConnection": ("clawscope.agent.realtime", "OpenAIRealtimeConnection"),
    "AudioProvider": ("clawscope.agent.realtime", "AudioProvider"),
    "MicrophoneProvider": ("clawscope.agent.realtime", "MicrophoneProvider"),
    "AudioConfig": ("clawscope.agent.realtime", "AudioConfig"),
}


def __getattr__(name: str) -> Any:
    """Lazily import agent implementations to avoid circular imports."""
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module 'clawscope.agent' has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value