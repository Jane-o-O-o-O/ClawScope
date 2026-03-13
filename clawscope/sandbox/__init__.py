"""Sandbox execution module for ClawScope."""

from clawscope.sandbox.base import Sandbox, SandboxResult, SandboxStatus
from clawscope.sandbox.docker import DockerSandbox, DOCKER_AVAILABLE
from clawscope.sandbox.config import SandboxConfig
from clawscope.sandbox.manager import (
    SandboxManager,
    get_sandbox_manager,
    configure_sandbox,
)

__all__ = [
    "Sandbox",
    "SandboxResult",
    "SandboxStatus",
    "DockerSandbox",
    "DOCKER_AVAILABLE",
    "SandboxConfig",
    "SandboxManager",
    "get_sandbox_manager",
    "configure_sandbox",
]
