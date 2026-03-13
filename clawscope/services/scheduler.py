"""Scheduler service for ClawScope."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from clawscope.services.cron import CronService
from clawscope.services.heartbeat import HeartbeatService

if TYPE_CHECKING:
    from clawscope.bus import MessageBus
    from clawscope.config import ServicesConfig


class SchedulerService:
    """
    Unified scheduler service combining Cron and Heartbeat.

    Manages scheduled tasks and periodic wake-ups.
    """

    def __init__(
        self,
        workspace: Path,
        bus: "MessageBus",
        config: "ServicesConfig",
    ):
        """
        Initialize scheduler service.

        Args:
            workspace: Workspace directory
            bus: Message bus instance
            config: Services configuration
        """
        self.workspace = workspace
        self.bus = bus
        self.config = config

        self._cron: CronService | None = None
        self._heartbeat: HeartbeatService | None = None
        self._running = False
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        """Start scheduler services."""
        self._running = True
        logger.info("Starting SchedulerService")

        # Start cron service
        if self.config.cron_enabled:
            self._cron = CronService(
                store_path=self.workspace / "cron" / "jobs.json",
                on_job=self._handle_cron_job,
            )
            task = asyncio.create_task(self._cron.run())
            self._tasks.append(task)

        # Start heartbeat service
        if self.config.heartbeat_enabled:
            self._heartbeat = HeartbeatService(
                workspace=self.workspace,
                bus=self.bus,
                interval=self.config.heartbeat_interval,
            )
            task = asyncio.create_task(self._heartbeat.run())
            self._tasks.append(task)

    async def stop(self) -> None:
        """Stop scheduler services."""
        self._running = False

        for task in self._tasks:
            task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks.clear()

        logger.info("SchedulerService stopped")

    async def _handle_cron_job(self, job: dict) -> None:
        """Handle a triggered cron job."""
        from clawscope.bus import InboundMessage

        message = job.get("message", "Scheduled task triggered")
        channel = job.get("channel", "system")
        chat_id = job.get("chat_id", "system")

        inbound = InboundMessage(
            channel=channel,
            sender_id=f"cron:{job.get('id', 'unknown')}",
            chat_id=chat_id,
            content=message,
            metadata={"cron_job": job},
        )
        await self.bus.publish_inbound(inbound)


__all__ = ["SchedulerService"]
