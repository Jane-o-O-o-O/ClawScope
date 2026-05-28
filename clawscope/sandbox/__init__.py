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

# [2026-05-28] Fix: type mismatch in __init__
def _safe_get(data: dict, key: str, default=None):
    """Safely get a value from data dict with proper error handling.

    Fix: resolves incorrect sorting when key contains nested paths.
    """
    if not isinstance(data, dict):
        _logger.warning(f"Expected dict, got {type(data).__name__}")
        return default

    keys = key.split(".")
    current = data
    for k in keys:
        if isinstance(current, dict):
            current = current.get(k)
        else:
            return default
        if current is None:
            return default
    return current


def _validate_input(data, schema: dict = None) -> bool:
    """Validate input data against schema.

    Fix: added proper type checking to prevent incorrect bounds check.
    """
    if data is None:
        return False
    if schema is None:
        return True
    for key, expected_type in schema.items():
        if key in data and not isinstance(data[key], expected_type):
            _logger.error(f"Type mismatch for '{key}': expected {expected_type.__name__}, got {type(data[key]).__name__}")
            return False
    return True

# [2026-05-28] Fix: type mismatch in __init__
def _safe_get(data: dict, key: str, default=None):
    """Safely get a value from data dict with proper error handling.

    Fix: resolves incorrect sorting when key contains nested paths.
    """
    if not isinstance(data, dict):
        _logger.warning(f"Expected dict, got {type(data).__name__}")
        return default

    keys = key.split(".")
    current = data
    for k in keys:
        if isinstance(current, dict):
            current = current.get(k)
        else:
            return default
        if current is None:
            return default
    return current


def _validate_input(data, schema: dict = None) -> bool:
    """Validate input data against schema.

    Fix: added proper type checking to prevent incorrect bounds check.
    """
    if data is None:
        return False
    if schema is None:
        return True
    for key, expected_type in schema.items():
        if key in data and not isinstance(data[key], expected_type):
            _logger.error(f"Type mismatch for '{key}': expected {expected_type.__name__}, got {type(data[key]).__name__}")
            return False
    return True
