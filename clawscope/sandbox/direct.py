"""Helpers for direct command execution outside Docker sandbox."""

from __future__ import annotations

import asyncio
import os
import subprocess
from datetime import datetime
from pathlib import Path

from clawscope.sandbox.base import SandboxResult, SandboxStatus


def resolve_direct_cwd(
    cwd: str | None,
    workspace_path: Path | None,
    sandbox_working_dir: str,
) -> str | None:
    """Resolve a sandbox-style working directory to a host path."""
    if workspace_path is None:
        return cwd

    workspace_root = workspace_path.expanduser().resolve()

    if cwd is None or cwd in {"", ".", sandbox_working_dir}:
        return str(workspace_root)

    normalized_working_dir = sandbox_working_dir.rstrip("/")
    if cwd == normalized_working_dir or cwd.startswith(f"{normalized_working_dir}/"):
        relative = cwd[len(normalized_working_dir) :].lstrip("/")
        if not relative:
            return str(workspace_root)
        return str((workspace_root / Path(relative)).resolve())

    host_path = Path(cwd)
    if host_path.is_absolute():
        return str(host_path)

    return str((workspace_root / host_path).resolve())


async def execute_direct_command(
    command: str,
    timeout: int | None,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    workspace_path: Path | None = None,
    sandbox_working_dir: str = "/workspace",
) -> SandboxResult:
    """Execute a command directly on the host while mirroring sandbox defaults."""
    timeout = timeout or 60
    started_at = datetime.now()
    resolved_cwd = resolve_direct_cwd(cwd, workspace_path, sandbox_working_dir)

    direct_env = os.environ.copy()
    if env:
        direct_env.update(env)

    try:
        try:
            completed = await asyncio.to_thread(
                subprocess.run,
                command,
                shell=True,
                capture_output=True,
                cwd=resolved_cwd,
                env=direct_env,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            finished_at = datetime.now()
            return SandboxResult(
                status=SandboxStatus.TIMEOUT,
                error=f"Command timed out after {timeout} seconds",
                started_at=started_at,
                finished_at=finished_at,
                duration_ms=(finished_at - started_at).total_seconds() * 1000,
                metadata={
                    "command": command,
                    "cwd": resolved_cwd,
                    "mode": "direct",
                },
            )

        finished_at = datetime.now()
        return SandboxResult(
            stdout=completed.stdout.decode("utf-8", errors="replace"),
            stderr=completed.stderr.decode("utf-8", errors="replace"),
            exit_code=completed.returncode or 0,
            status=SandboxStatus.COMPLETED,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=(finished_at - started_at).total_seconds() * 1000,
            metadata={
                "command": command,
                "cwd": resolved_cwd,
                "mode": "direct",
            },
        )

    except Exception as exc:
        finished_at = datetime.now()
        return SandboxResult(
            status=SandboxStatus.ERROR,
            error=str(exc),
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=(finished_at - started_at).total_seconds() * 1000,
            metadata={
                "command": command,
                "cwd": resolved_cwd,
                "mode": "direct",
            },
        )


__all__ = ["execute_direct_command", "resolve_direct_cwd"]
