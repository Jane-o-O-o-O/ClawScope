"""Sandbox manager for ClawScope."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from clawscope.sandbox.base import Sandbox, SandboxResult, SandboxStatus
from clawscope.sandbox.config import SandboxConfig
from clawscope.sandbox.docker import DockerSandbox, DOCKER_AVAILABLE

if TYPE_CHECKING:
    pass


class SandboxManager:
    """
    Manages sandbox instances for different sessions.

    Features:
    - Session-based sandbox isolation
    - Automatic container lifecycle management
    - Pool management for reusable containers
    - Graceful shutdown
    """

    def __init__(
        self,
        config: SandboxConfig | None = None,
        workspace: Path | None = None,
    ):
        """
        Initialize sandbox manager.

        Args:
            config: Default sandbox configuration
            workspace: Default workspace path
        """
        self.config = config or SandboxConfig()
        self.workspace = workspace

        if workspace:
            self.config.workspace_path = workspace

        self._sandboxes: dict[str, Sandbox] = {}
        self._lock = asyncio.Lock()

    @property
    def is_available(self) -> bool:
        """Check if sandbox functionality is available."""
        return DOCKER_AVAILABLE and self.config.enabled

    async def get_sandbox(
        self,
        session_id: str,
        config: SandboxConfig | None = None,
    ) -> Sandbox:
        """
        Get or create a sandbox for a session.

        Args:
            session_id: Session identifier
            config: Optional session-specific config

        Returns:
            Sandbox instance
        """
        async with self._lock:
            if session_id in self._sandboxes:
                sandbox = self._sandboxes[session_id]
                if await sandbox.is_running():
                    return sandbox
                else:
                    # Cleanup dead sandbox
                    await sandbox.cleanup()

            # Create new sandbox
            sandbox_config = config or self.config
            sandbox = DockerSandbox(
                config=sandbox_config,
                session_id=session_id,
            )
            await sandbox.start()
            self._sandboxes[session_id] = sandbox
            return sandbox

    async def execute(
        self,
        command: str,
        session_id: str = "default",
        timeout: int | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> SandboxResult:
        """
        Execute a command in a sandboxed environment.

        Args:
            command: Command to execute
            session_id: Session identifier
            timeout: Timeout in seconds
            env: Environment variables
            cwd: Working directory

        Returns:
            SandboxResult with execution details
        """
        if not self.is_available:
            # Fall back to direct execution
            return await self._execute_direct(command, timeout)

        try:
            sandbox = await self.get_sandbox(session_id)
            return await sandbox.execute(
                command=command,
                timeout=timeout,
                env=env,
                cwd=cwd,
            )
        except Exception as e:
            logger.error(f"Sandbox execution failed: {e}")
            return SandboxResult(
                status=SandboxStatus.ERROR,
                error=str(e),
            )

    async def release_sandbox(self, session_id: str) -> None:
        """
        Release a sandbox for a session.

        Args:
            session_id: Session identifier
        """
        async with self._lock:
            if session_id in self._sandboxes:
                sandbox = self._sandboxes.pop(session_id)
                await sandbox.cleanup()
                logger.debug(f"Released sandbox for session: {session_id}")

    async def cleanup_all(self) -> None:
        """Clean up all sandboxes."""
        async with self._lock:
            for session_id, sandbox in list(self._sandboxes.items()):
                try:
                    await sandbox.cleanup()
                except Exception as e:
                    logger.warning(f"Error cleaning up sandbox {session_id}: {e}")
            self._sandboxes.clear()
            logger.info("Cleaned up all sandboxes")

    async def get_stats(self) -> dict:
        """Get sandbox manager statistics."""
        stats = {
            "enabled": self.config.enabled,
            "docker_available": DOCKER_AVAILABLE,
            "active_sandboxes": len(self._sandboxes),
            "sessions": list(self._sandboxes.keys()),
        }
        return stats

    async def _execute_direct(
        self, command: str, timeout: int | None
    ) -> SandboxResult:
        """Execute command directly without sandbox."""
        from datetime import datetime

        timeout = timeout or 60
        started_at = datetime.now()

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                finished_at = datetime.now()
                return SandboxResult(
                    status=SandboxStatus.TIMEOUT,
                    error=f"Command timed out after {timeout} seconds",
                    started_at=started_at,
                    finished_at=finished_at,
                    duration_ms=(finished_at - started_at).total_seconds() * 1000,
                )

            finished_at = datetime.now()
            return SandboxResult(
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
                exit_code=process.returncode or 0,
                status=SandboxStatus.COMPLETED,
                started_at=started_at,
                finished_at=finished_at,
                duration_ms=(finished_at - started_at).total_seconds() * 1000,
            )

        except Exception as e:
            from datetime import datetime
            finished_at = datetime.now()
            return SandboxResult(
                status=SandboxStatus.ERROR,
                error=str(e),
                started_at=started_at,
                finished_at=finished_at,
                duration_ms=(finished_at - started_at).total_seconds() * 1000,
            )


# Global sandbox manager instance
_sandbox_manager: SandboxManager | None = None


def get_sandbox_manager() -> SandboxManager:
    """Get the global sandbox manager instance."""
    global _sandbox_manager
    if _sandbox_manager is None:
        _sandbox_manager = SandboxManager()
    return _sandbox_manager


def configure_sandbox(
    config: SandboxConfig | None = None,
    workspace: Path | None = None,
) -> SandboxManager:
    """Configure the global sandbox manager."""
    global _sandbox_manager
    _sandbox_manager = SandboxManager(config=config, workspace=workspace)
    return _sandbox_manager


__all__ = [
    "SandboxManager",
    "get_sandbox_manager",
    "configure_sandbox",
]
