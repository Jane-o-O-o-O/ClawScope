"""
ClawScope - Unified AI Agent Platform

Combining AgentScope and Nanobot capabilities for enterprise-grade
multi-channel AI agent applications.
"""

from clawscope._version import __version__
from clawscope.app import ClawScope
from clawscope.config import Config

__all__ = [
    "__version__",
    "ClawScope",
    "Config",
]
