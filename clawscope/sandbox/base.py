"""Base sandbox interface for ClawScope."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class SandboxStatus(str, Enum):
    """Sandbox execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    ERROR = "error"
    KILLED = "killed"


@dataclass
class SandboxResult:
    """Result from sandbox execution."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    status: SandboxStatus = SandboxStatus.COMPLETED
    duration_ms: float = 0.0
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if execution was successful."""
        return self.status == SandboxStatus.COMPLETED and self.exit_code == 0

    @property
    def output(self) -> str:
        """Get combined output."""
        parts = []
        if self.stdout:
            parts.append(f"STDOUT:\n{self.stdout}")
        if self.stderr:
            parts.append(f"STDERR:\n{self.stderr}")
        if self.exit_code != 0:
            parts.append(f"Exit code: {self.exit_code}")
        if self.error:
            parts.append(f"Error: {self.error}")
        return "\n".join(parts) if parts else "Command completed with no output"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "error": self.error,
            "metadata": self.metadata,
        }


class Sandbox(ABC):
    """Abstract base class for sandbox implementations."""

    @abstractmethod
    async def start(self) -> None:
        """Start the sandbox environment."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the sandbox environment."""
        pass

    @abstractmethod
    async def execute(
        self,
        command: str,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> SandboxResult:
        """
        Execute a command in the sandbox.

        Args:
            command: Command to execute
            timeout: Timeout in seconds
            env: Environment variables
            cwd: Working directory

        Returns:
            SandboxResult with execution details
        """
        pass

    @abstractmethod
    async def write_file(self, path: str, content: str | bytes) -> bool:
        """Write a file to the sandbox."""
        pass

    @abstractmethod
    async def read_file(self, path: str) -> str | bytes | None:
        """Read a file from the sandbox."""
        pass

    @abstractmethod
    async def is_running(self) -> bool:
        """Check if sandbox is running."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up sandbox resources."""
        pass


__all__ = ["Sandbox", "SandboxResult", "SandboxStatus"]
