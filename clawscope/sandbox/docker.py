"""Docker sandbox implementation for ClawScope."""

from __future__ import annotations

import asyncio
import tarfile
import io
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from clawscope.sandbox.base import Sandbox, SandboxResult, SandboxStatus
from clawscope.sandbox.config import SandboxConfig

try:
    import aiodocker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False


class DockerSandbox(Sandbox):
    """
    Docker-based sandbox for secure command execution.

    Features:
    - Container isolation
    - Resource limits (CPU, memory, PIDs)
    - Network isolation
    - Read-only root filesystem
    - Capability dropping
    - Automatic cleanup
    """

    def __init__(
        self,
        config: SandboxConfig | None = None,
        session_id: str | None = None,
    ):
        """
        Initialize Docker sandbox.

        Args:
            config: Sandbox configuration
            session_id: Unique session identifier
        """
        if not DOCKER_AVAILABLE:
            raise ImportError(
                "aiodocker is required for Docker sandbox. "
                "Install with: pip install aiodocker"
            )

        self.config = config or SandboxConfig()
        self.session_id = session_id or "default"
        self._docker: aiodocker.Docker | None = None
        self._container: Any = None
        self._container_id: str | None = None

    @property
    def container_name(self) -> str:
        """Generate unique container name."""
        return f"{self.config.container_name_prefix}-{self.session_id}"

    async def start(self) -> None:
        """Start the Docker sandbox container."""
        if not self.config.enabled:
            logger.debug("Sandbox disabled, skipping start")
            return

        try:
            self._docker = aiodocker.Docker()

            # Check if we can reuse existing container
            if self.config.reuse_container:
                existing = await self._find_existing_container()
                if existing:
                    self._container = existing
                    self._container_id = existing.id

                    # Start if not running
                    info = await existing.show()
                    if info["State"]["Status"] != "running":
                        await existing.start()

                    logger.info(f"Reusing container: {self.container_name}")
                    return

            # Pull image if needed
            await self._ensure_image()

            # Create container configuration
            container_config = self._build_container_config()

            # Create and start container
            self._container = await self._docker.containers.create(
                config=container_config,
                name=self.container_name,
            )
            self._container_id = self._container.id

            await self._container.start()
            logger.info(f"Started sandbox container: {self.container_name}")

        except Exception as e:
            logger.error(f"Failed to start sandbox: {e}")
            raise

    async def stop(self) -> None:
        """Stop the sandbox container."""
        if self._container:
            try:
                await self._container.stop()
                logger.info(f"Stopped sandbox container: {self.container_name}")
            except Exception as e:
                logger.warning(f"Error stopping container: {e}")

    async def execute(
        self,
        command: str,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> SandboxResult:
        """
        Execute a command in the Docker sandbox.

        Args:
            command: Command to execute
            timeout: Timeout in seconds
            env: Environment variables
            cwd: Working directory

        Returns:
            SandboxResult with execution details
        """
        if not self.config.enabled:
            # Fall back to direct execution with warning
            logger.warning("Sandbox disabled, executing directly (unsafe)")
            return await self._execute_direct(command, timeout)

        if not self._container:
            await self.start()

        timeout = timeout or self.config.default_timeout
        timeout = min(timeout, self.config.max_timeout)

        started_at = datetime.now()

        try:
            # Build exec configuration
            exec_config = {
                "Cmd": ["/bin/sh", "-c", command],
                "AttachStdout": True,
                "AttachStderr": True,
                "Tty": False,
            }

            if env:
                exec_config["Env"] = [f"{k}={v}" for k, v in env.items()]

            if cwd:
                exec_config["WorkingDir"] = cwd

            # Create exec instance
            exec_instance = await self._container.exec(exec_config)

            # Start exec with timeout
            try:
                stream = exec_instance.start(detach=False)

                stdout_data = []
                stderr_data = []

                async def read_output():
                    async for chunk in stream:
                        if chunk.get("stdout"):
                            stdout_data.append(chunk["stdout"])
                        if chunk.get("stderr"):
                            stderr_data.append(chunk["stderr"])

                await asyncio.wait_for(read_output(), timeout=timeout)

            except asyncio.TimeoutError:
                finished_at = datetime.now()
                return SandboxResult(
                    status=SandboxStatus.TIMEOUT,
                    error=f"Command timed out after {timeout} seconds",
                    started_at=started_at,
                    finished_at=finished_at,
                    duration_ms=(finished_at - started_at).total_seconds() * 1000,
                )

            # Get exit code
            exec_info = await exec_instance.inspect()
            exit_code = exec_info.get("ExitCode", 0)

            finished_at = datetime.now()

            stdout = b"".join(stdout_data).decode("utf-8", errors="replace")
            stderr = b"".join(stderr_data).decode("utf-8", errors="replace")

            return SandboxResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                status=SandboxStatus.COMPLETED,
                started_at=started_at,
                finished_at=finished_at,
                duration_ms=(finished_at - started_at).total_seconds() * 1000,
                metadata={
                    "container_id": self._container_id,
                    "command": command,
                },
            )

        except Exception as e:
            finished_at = datetime.now()
            logger.error(f"Sandbox execution error: {e}")
            return SandboxResult(
                status=SandboxStatus.ERROR,
                error=str(e),
                started_at=started_at,
                finished_at=finished_at,
                duration_ms=(finished_at - started_at).total_seconds() * 1000,
            )

    async def write_file(self, path: str, content: str | bytes) -> bool:
        """Write a file to the sandbox."""
        if not self._container:
            return False

        try:
            # Create tar archive with file
            if isinstance(content, str):
                content = content.encode("utf-8")

            tar_buffer = io.BytesIO()
            with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
                file_data = io.BytesIO(content)
                info = tarfile.TarInfo(name=Path(path).name)
                info.size = len(content)
                tar.addfile(info, file_data)

            tar_buffer.seek(0)

            # Put archive in container
            parent_dir = str(Path(path).parent)
            await self._container.put_archive(parent_dir, tar_buffer.read())
            return True

        except Exception as e:
            logger.error(f"Failed to write file to sandbox: {e}")
            return False

    async def read_file(self, path: str) -> str | bytes | None:
        """Read a file from the sandbox."""
        if not self._container:
            return None

        try:
            # Get archive from container
            archive_data = await self._container.get_archive(path)

            # Extract file content
            tar_buffer = io.BytesIO(archive_data)
            with tarfile.open(fileobj=tar_buffer, mode="r") as tar:
                for member in tar.getmembers():
                    f = tar.extractfile(member)
                    if f:
                        return f.read()
            return None

        except Exception as e:
            logger.error(f"Failed to read file from sandbox: {e}")
            return None

    async def is_running(self) -> bool:
        """Check if sandbox container is running."""
        if not self._container:
            return False

        try:
            info = await self._container.show()
            return info["State"]["Status"] == "running"
        except Exception:
            return False

    async def cleanup(self) -> None:
        """Clean up sandbox resources."""
        if self._container and not self.config.keep_container:
            try:
                await self._container.stop()
                await self._container.delete(force=True)
                logger.info(f"Cleaned up container: {self.container_name}")
            except Exception as e:
                logger.warning(f"Error cleaning up container: {e}")

        if self._docker:
            await self._docker.close()
            self._docker = None

        self._container = None
        self._container_id = None

    async def _ensure_image(self) -> None:
        """Ensure Docker image is available."""
        try:
            await self._docker.images.inspect(self.config.image)
        except aiodocker.DockerError:
            logger.info(f"Pulling image: {self.config.image}")
            await self._docker.images.pull(self.config.image)

    async def _find_existing_container(self) -> Any | None:
        """Find existing container by name."""
        try:
            containers = await self._docker.containers.list(all=True)
            for container in containers:
                info = await container.show()
                if info["Name"].lstrip("/") == self.container_name:
                    return container
        except Exception:
            pass
        return None

    def _build_container_config(self) -> dict[str, Any]:
        """Build Docker container configuration."""
        config = {
            "Image": self.config.image,
            "Cmd": ["/bin/sh", "-c", "tail -f /dev/null"],  # Keep alive
            "WorkingDir": self.config.working_dir,
            "Tty": False,
            "OpenStdin": False,
            "NetworkDisabled": not self.config.network_enabled,
            "HostConfig": {
                "Memory": self._parse_memory(self.config.memory_limit),
                "NanoCPUs": int(self.config.cpu_limit * 1e9),
                "PidsLimit": self.config.pids_limit,
                "Privileged": self.config.privileged,
                "CapDrop": self.config.cap_drop,
                "CapAdd": self.config.cap_add if self.config.cap_add else None,
                "SecurityOpt": self.config.security_opt,
                "ReadonlyRootfs": self.config.read_only_root,
            },
        }

        # Add volume mounts
        if self.config.mount_workspace and self.config.workspace_path:
            workspace = str(self.config.workspace_path.absolute())
            config["HostConfig"]["Binds"] = [
                f"{workspace}:{self.config.working_dir}:rw"
            ]
            # Need writable tmpfs for read-only root
            config["HostConfig"]["Tmpfs"] = {
                "/tmp": "rw,noexec,nosuid,size=64m",
                "/var/tmp": "rw,noexec,nosuid,size=64m",
            }

        if self.config.network_enabled:
            config["HostConfig"]["NetworkMode"] = self.config.network_mode

        return config

    def _parse_memory(self, memory: str) -> int:
        """Parse memory string to bytes."""
        units = {"b": 1, "k": 1024, "m": 1024**2, "g": 1024**3}
        memory = memory.lower().strip()
        if memory[-1] in units:
            return int(memory[:-1]) * units[memory[-1]]
        return int(memory)

    async def _execute_direct(
        self, command: str, timeout: int | None
    ) -> SandboxResult:
        """Execute command directly without sandbox (fallback)."""
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
            finished_at = datetime.now()
            return SandboxResult(
                status=SandboxStatus.ERROR,
                error=str(e),
                started_at=started_at,
                finished_at=finished_at,
                duration_ms=(finished_at - started_at).total_seconds() * 1000,
            )


__all__ = ["DockerSandbox", "DOCKER_AVAILABLE"]
