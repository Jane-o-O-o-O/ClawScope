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

from clawscope._version import __version__
from clawscope.app import ClawScope, quick_chat, create_agent
from clawscope.config import Config

# Re-export commonly used components
from clawscope.message import Msg
from clawscope.agent import ReActAgent, AgentBase
from clawscope.memory import InMemoryMemory

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
