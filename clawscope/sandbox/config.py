"""Sandbox configuration for ClawScope."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SandboxConfig:
    """Configuration for sandbox execution."""

    # Enable sandbox mode
    enabled: bool = False

    # Docker settings
    image: str = "python:3.11-slim"
    container_name_prefix: str = "clawscope-sandbox"

    # Resource limits
    memory_limit: str = "512m"
    cpu_limit: float = 1.0
    pids_limit: int = 100

    # Timeout settings
    default_timeout: int = 60
    max_timeout: int = 300

    # Network settings
    network_enabled: bool = False
    network_mode: str = "none"  # none, bridge, host

    # Volume settings
    workspace_path: Path | None = None
    mount_workspace: bool = True
    read_only_root: bool = True

    # Security settings
    privileged: bool = False
    cap_drop: list[str] = field(default_factory=lambda: ["ALL"])
    cap_add: list[str] = field(default_factory=list)
    security_opt: list[str] = field(default_factory=lambda: ["no-new-privileges"])

    # Allowed commands (if empty, all non-dangerous commands allowed)
    allowed_commands: list[str] = field(default_factory=list)
    blocked_commands: list[str] = field(default_factory=lambda: [
        "rm -rf /",
        "mkfs",
        "dd if=",
        "chmod -R 777 /",
    ])

    # Container lifecycle
    auto_remove: bool = True
    keep_container: bool = False
    reuse_container: bool = True

    # Working directory inside container
    working_dir: str = "/workspace"

    def to_docker_config(self) -> dict:
        """Convert to Docker API configuration."""
        config = {
            "Image": self.image,
            "WorkingDir": self.working_dir,
            "Tty": False,
            "AttachStdout": True,
            "AttachStderr": True,
            "NetworkDisabled": not self.network_enabled,
            "HostConfig": {
                "Memory": self._parse_memory(self.memory_limit),
                "NanoCpus": int(self.cpu_limit * 1e9),
                "PidsLimit": self.pids_limit,
                "Privileged": self.privileged,
                "CapDrop": self.cap_drop,
                "CapAdd": self.cap_add,
                "SecurityOpt": self.security_opt,
                "AutoRemove": self.auto_remove,
                "ReadonlyRootfs": self.read_only_root,
            },
        }

        if self.network_enabled:
            config["HostConfig"]["NetworkMode"] = self.network_mode

        return config

    def _parse_memory(self, memory: str) -> int:
        """Parse memory string to bytes."""
        units = {"b": 1, "k": 1024, "m": 1024**2, "g": 1024**3}
        memory = memory.lower().strip()

        if memory[-1] in units:
            return int(memory[:-1]) * units[memory[-1]]
        return int(memory)


__all__ = ["SandboxConfig"]
