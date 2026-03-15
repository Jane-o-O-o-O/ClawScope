"""
ClawScope - Unified AI Agent Platform

Combining AgentScope and Nanobot capabilities for enterprise-grade
multi-channel AI agent applications.

Quick Start:
    >>> from clawscope import ClawScope, quick_chat
    >>>
    >>> # Simple one-off chat
    >>> response = await quick_chat("Hello!")
    >>>
    >>> # Full platform
    >>> app = ClawScope.create(model_provider="openai")
    >>> await app.start()
    >>> response = await app.chat("Hello!")
"""

from importlib import import_module
from typing import Any

from clawscope._version import __version__
from clawscope.config import Config

__all__ = [
    # Version
    "__version__",
    # Main app
    "ClawScope",
    "Config",
    # Quick start
    "quick_chat",
    "create_agent",
    # Core components
    "Msg",
    "ReActAgent",
    "AgentBase",
    "InMemoryMemory",
]


_LAZY_IMPORTS = {
    "ClawScope": ("clawscope.app", "ClawScope"),
    "quick_chat": ("clawscope.app", "quick_chat"),
    "create_agent": ("clawscope.app", "create_agent"),
    "Msg": ("clawscope.message", "Msg"),
    "ReActAgent": ("clawscope.agent", "ReActAgent"),
    "AgentBase": ("clawscope.agent", "AgentBase"),
    "InMemoryMemory": ("clawscope.memory", "InMemoryMemory"),
}


def __getattr__(name: str) -> Any:
    """Lazily import heavyweight exports on first access."""
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module 'clawscope' has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
