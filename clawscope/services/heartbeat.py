"""Heartbeat service for ClawScope."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import aiofiles
from loguru import logger

if TYPE_CHECKING:
    from clawscope.bus import MessageBus


class HeartbeatService:
    """
    Heartbeat service for proactive tasks.

    Periodically wakes up and executes tasks defined
    in HEARTBEAT.md.
    """

    def __init__(
        self,
        workspace: Path,
        bus: "MessageBus",
        interval: int = 1800,  # 30 minutes
    ):
        """
        Initialize heartbeat service.

        Args:
            workspace: Workspace directory
            bus: Message bus instance
            interval: Heartbeat interval in seconds
        """
        self.workspace = workspace
        self.bus = bus
        self.interval = interval
        self.heartbeat_file = workspace / "HEARTBEAT.md"
        self._running = False

    async def run(self) -> None:
        """Run the heartbeat service."""
        self._running = True
        logger.info(f"HeartbeatService started (interval: {self.interval}s)")

        while self._running:
            try:
                await asyncio.sleep(self.interval)

                if not self._running:
                    break

                await self._execute_heartbeat()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    async def stop(self) -> None:
        """Stop the heartbeat service."""
        self._running = False

    async def _execute_heartbeat(self) -> None:
        """Execute heartbeat tasks."""
        tasks = await self._load_tasks()
        if not tasks:
            return

        logger.info(f"Executing heartbeat with {len(tasks)} tasks")

        # Publish heartbeat message
        from clawscope.bus import InboundMessage

        inbound = InboundMessage(
            channel="system",
            sender_id="heartbeat",
            chat_id="system:heartbeat",
            content=f"Heartbeat triggered. Tasks:\n{tasks}",
            metadata={"heartbeat": True},
        )
        await self.bus.publish_inbound(inbound)

    async def _load_tasks(self) -> str:
        """Load tasks from HEARTBEAT.md."""
        if not self.heartbeat_file.exists():
            return ""

        try:
            async with aiofiles.open(self.heartbeat_file, "r", encoding="utf-8") as f:
                return await f.read()
        except Exception as e:
            logger.error(f"Error loading heartbeat tasks: {e}")
            return ""

    async def save_tasks(self, content: str) -> None:
        """Save tasks to HEARTBEAT.md."""
        try:
            async with aiofiles.open(self.heartbeat_file, "w", encoding="utf-8") as f:
                await f.write(content)
        except Exception as e:
            logger.error(f"Error saving heartbeat tasks: {e}")


__all__ = ["HeartbeatService"]
