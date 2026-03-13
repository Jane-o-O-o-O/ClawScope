"""Shell execution tool for ClawScope."""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from clawscope.config import ToolsConfig


# Dangerous patterns to block
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
) -> str:
    """
    Execute a shell command.

    Args:
        command: Shell command to execute
        timeout: Timeout in seconds
        config: Tool configuration

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


__all__ = ["execute_shell"]
