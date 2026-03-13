"""ClawScope main application class."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from clawscope.config import Config

if TYPE_CHECKING:
    from clawscope.bus import MessageBus
    from clawscope.channels import ChannelManager
    from clawscope.orchestration import SessionRouter
    from clawscope.services import SchedulerService


class ClawScope:
    """
    ClawScope - Unified AI Agent Platform.

    Main application class that orchestrates all components:
    - MessageBus for channel-agent communication
    - ChannelManager for multi-platform support
    - SessionRouter for message routing
    - Agent instances for conversation handling
    - Background services for scheduling
    """

    def __init__(self, config: Config):
        """
        Initialize ClawScope application.

        Args:
            config: Application configuration
        """
        self.config = config
        self._bus: MessageBus | None = None
        self._channels: ChannelManager | None = None
        self._router: SessionRouter | None = None
        self._scheduler: SchedulerService | None = None
        self._running = False
        self._tasks: list[asyncio.Task] = []

    @classmethod
    def from_config(cls, path: str | Path) -> "ClawScope":
        """
        Create ClawScope instance from config file.

        Args:
            path: Path to YAML or JSON config file

        Returns:
            Configured ClawScope instance
        """
        config = Config.from_file(path)
        return cls(config)

    async def start(self) -> None:
        """Start the ClawScope platform."""
        if self._running:
            logger.warning("ClawScope is already running")
            return

        logger.info("Starting ClawScope platform...")

        # Ensure workspace exists
        self.config.ensure_workspace()

        # Initialize components
        await self._init_components()

        # Start components
        await self._start_components()

        self._running = True
        logger.info("ClawScope platform started successfully")

    async def stop(self) -> None:
        """Stop the ClawScope platform."""
        if not self._running:
            return

        logger.info("Stopping ClawScope platform...")

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()

        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks.clear()

        # Stop components
        await self._stop_components()

        self._running = False
        logger.info("ClawScope platform stopped")

    async def _init_components(self) -> None:
        """Initialize all components."""
        from clawscope.bus import MessageBus
        from clawscope.channels import ChannelManager
        from clawscope.memory import SessionManager
        from clawscope.model import ModelRegistry
        from clawscope.orchestration import SessionRouter
        from clawscope.services import SchedulerService
        from clawscope.tool import ToolRegistry

        # Message bus
        self._bus = MessageBus()

        # Model registry
        self._model_registry = ModelRegistry(self.config.model)

        # Tool registry
        self._tool_registry = ToolRegistry(self.config.tools)
        await self._tool_registry.load_builtin_tools()

        # Session manager
        self._session_manager = SessionManager(
            workspace=self.config.workspace,
            backend=self.config.memory.session,
        )

        # Channel manager
        self._channels = ChannelManager(
            bus=self._bus,
            config=self.config.channels,
        )

        # Session router
        self._router = SessionRouter(
            bus=self._bus,
            sessions=self._session_manager,
            model_registry=self._model_registry,
            tool_registry=self._tool_registry,
            config=self.config.agent,
        )

        # Scheduler service
        if self.config.services.cron_enabled or self.config.services.heartbeat_enabled:
            self._scheduler = SchedulerService(
                workspace=self.config.workspace,
                bus=self._bus,
                config=self.config.services,
            )

    async def _start_components(self) -> None:
        """Start all components."""
        # Start channel manager
        if self._channels:
            task = asyncio.create_task(self._channels.start())
            self._tasks.append(task)

        # Start session router
        if self._router:
            task = asyncio.create_task(self._router.run())
            self._tasks.append(task)

        # Start scheduler
        if self._scheduler:
            task = asyncio.create_task(self._scheduler.start())
            self._tasks.append(task)

    async def _stop_components(self) -> None:
        """Stop all components."""
        if self._channels:
            await self._channels.stop()

        if self._router:
            self._router.stop()

        if self._scheduler:
            await self._scheduler.stop()

    async def run_forever(self) -> None:
        """Run the platform until interrupted."""
        await self.start()
        try:
            # Wait for all tasks
            if self._tasks:
                await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    @property
    def is_running(self) -> bool:
        """Check if platform is running."""
        return self._running

    @property
    def bus(self) -> "MessageBus":
        """Get message bus instance."""
        if self._bus is None:
            raise RuntimeError("ClawScope not initialized")
        return self._bus


__all__ = ["ClawScope"]
