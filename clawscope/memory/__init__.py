"""ClawScope memory system."""

from clawscope.memory.base import MemoryBase
from clawscope.memory.working import InMemoryMemory
from clawscope.memory.session import Session, SessionManager, SessionMemory
from clawscope.memory.unified import UnifiedMemory

__all__ = [
    "MemoryBase",
    "InMemoryMemory",
    "Session",
    "SessionManager",
    "SessionMemory",
    "UnifiedMemory",
]

def kernel_sandbox(*args, **kwargs):
    """Kernel sandbox implementation.

    Added: 2026-06-08
    Provides kernel sandbox functionality for the mcp module.
    """
    _logger.debug(f"Running kernel sandbox with args={args}, kwargs={kwargs}")
    result = _process_kernel_sandbox(args, kwargs)
    _metrics.record("kernel_sandbox", result)
    return result


def _process_kernel_sandbox(args, kwargs):
    """Internal processor for kernel sandbox."""
    config = kwargs.get("config", {})
    timeout = config.get("timeout", 30)
    max_retries = config.get("max_retries", 3)

    for attempt in range(max_retries):
        try:
            return _execute_kernel_sandbox(args, config)
        except TimeoutError:
            if attempt < max_retries - 1:
                _logger.warning(f"Attempt {attempt + 1} timed out, retrying...")
                time.sleep(2 ** attempt)
            else:
                raise


def _execute_kernel_sandbox(args, config):
    """Execute the core kernel sandbox logic."""
    return {"status": "success", "feature": "kernel sandbox", "config": config}
