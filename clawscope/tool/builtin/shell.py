"""Shell execution tool for ClawScope."""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from clawscope.config import ToolsConfig
    from clawscope.sandbox import SandboxConfig


# Dangerous patterns to block (even in sandbox for extra safety)
DANGEROUS_PATTERNS = [
    r"rm\s+-rf\s+/",
    r"rm\s+-rf\s+\*",
    r"mkfs\.",
    r"dd\s+if=",
    r":\(\)\{",  # Fork bomb
    r">\s*/dev/sd",
    r"chmod\s+-R\s+777\s+/",
]


async def execute_shell(
    command: str,
    timeout: int = 60,
    config: "ToolsConfig | None" = None,
    session_id: str = "default",
    use_sandbox: bool | None = None,
) -> str:
    """
    Execute a shell command, optionally in a Docker sandbox.

    Args:
        command: Shell command to execute
        timeout: Timeout in seconds
        config: Tool configuration
        session_id: Session ID for sandbox isolation
        use_sandbox: Force sandbox usage (None = auto-detect)

    Returns:
        Command output
    """
    # Check for dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return f"Error: Command blocked for safety: matches pattern '{pattern}'"

    # Get timeout from config
    if config:
        timeout = min(timeout, config.shell_timeout)

    # Determine sandbox usage
    sandbox_enabled = use_sandbox
    if sandbox_enabled is None:
        sandbox_enabled = config.sandbox_enabled if config else False

    if sandbox_enabled:
        return await _execute_in_sandbox(command, timeout, session_id)
    else:
        return await _execute_direct(command, timeout, config)


async def _execute_in_sandbox(
    command: str,
    timeout: int,
    session_id: str,
) -> str:
    """Execute command in Docker sandbox."""
    try:
        from clawscope.sandbox import get_sandbox_manager

        manager = get_sandbox_manager()

        if not manager.is_available:
            logger.warning("Sandbox not available, falling back to direct execution")
            return await _execute_direct(command, timeout, None)

        result = await manager.execute(
            command=command,
            session_id=session_id,
            timeout=timeout,
        )

        return result.output

    except ImportError:
        logger.warning("Sandbox module not available, using direct execution")
        return await _execute_direct(command, timeout, None)
    except Exception as e:
        logger.error(f"Sandbox execution failed: {e}")
        return f"Error: Sandbox execution failed: {str(e)}"


async def _execute_direct(
    command: str,
    timeout: int,
    config: "ToolsConfig | None",
) -> str:
    """Execute command directly on host (legacy mode)."""
    try:
        # Create subprocess
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Wait with timeout
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            process.kill()
            return f"Error: Command timed out after {timeout} seconds"

        # Decode output
        output = stdout.decode("utf-8", errors="replace")
        error = stderr.decode("utf-8", errors="replace")

        # Build result
        result = []
        if output:
            result.append(f"STDOUT:\n{output}")
        if error:
            result.append(f"STDERR:\n{error}")
        if process.returncode != 0:
            result.append(f"Exit code: {process.returncode}")

        if not result:
            return "Command completed with no output"

        full_result = "\n".join(result)

        # Truncate if too long
        max_length = config.max_output_length if config else 16000
        if len(full_result) > max_length:
            full_result = full_result[:max_length] + "\n... [truncated]"

        return full_result

    except Exception as e:
        return f"Error executing command: {str(e)}"


async def execute_shell_sandboxed(
    command: str,
    timeout: int = 60,
    session_id: str = "default",
    env: dict[str, str] | None = None,
    cwd: str | None = None,
) -> str:
    """
    Execute a shell command in Docker sandbox (explicit sandbox mode).

    Args:
        command: Shell command to execute
        timeout: Timeout in seconds
        session_id: Session ID for sandbox isolation
        env: Environment variables
        cwd: Working directory

    Returns:
        Command output
    """
    # Check for dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return f"Error: Command blocked for safety: matches pattern '{pattern}'"

    try:
        from clawscope.sandbox import get_sandbox_manager

        manager = get_sandbox_manager()
        result = await manager.execute(
            command=command,
            session_id=session_id,
            timeout=timeout,
            env=env,
            cwd=cwd,
        )

        return result.output

    except ImportError:
        return "Error: Sandbox module not available. Install with: pip install clawscope[sandbox]"
    except Exception as e:
        return f"Error: Sandbox execution failed: {str(e)}"


__all__ = ["execute_shell", "execute_shell_sandboxed"]
