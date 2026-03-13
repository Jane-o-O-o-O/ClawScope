"""Cron service for ClawScope."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Awaitable

import aiofiles
from loguru import logger


@dataclass
class CronJob:
    """Cron job definition."""

    id: str
    name: str
    schedule: str  # Cron expression
    message: str
    channel: str = "system"
    chat_id: str = "system"
    enabled: bool = True
    last_run: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "schedule": self.schedule,
            "message": self.message,
            "channel": self.channel,
            "chat_id": self.chat_id,
            "enabled": self.enabled,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CronJob":
        """Create from dictionary."""
        last_run = data.get("last_run")
        if isinstance(last_run, str):
            last_run = datetime.fromisoformat(last_run)

        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            schedule=data.get("schedule", ""),
            message=data.get("message", ""),
            channel=data.get("channel", "system"),
            chat_id=data.get("chat_id", "system"),
            enabled=data.get("enabled", True),
            last_run=last_run,
            metadata=data.get("metadata", {}),
        )


class CronService:
    """
    Cron job scheduling service.

    Features:
    - Cron expression parsing
    - Persistent job storage
    - Async execution callbacks
    """

    def __init__(
        self,
        store_path: Path,
        on_job: Callable[[dict], Awaitable[None]] | None = None,
    ):
        """
        Initialize cron service.

        Args:
            store_path: Path to job storage file
            on_job: Callback when job triggers
        """
        self.store_path = store_path
        self.on_job = on_job
        self._jobs: dict[str, CronJob] = {}
        self._running = False

    async def run(self) -> None:
        """Run the cron service."""
        self._running = True
        await self._load_jobs()
        logger.info("CronService started")

        while self._running:
            try:
                await self._check_jobs()
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cron error: {e}")
                await asyncio.sleep(60)

    async def stop(self) -> None:
        """Stop the cron service."""
        self._running = False
        await self._save_jobs()

    async def add_job(self, job: CronJob) -> None:
        """Add a cron job."""
        self._jobs[job.id] = job
        await self._save_jobs()
        logger.info(f"Added cron job: {job.name}")

    async def remove_job(self, job_id: str) -> None:
        """Remove a cron job."""
        if job_id in self._jobs:
            del self._jobs[job_id]
            await self._save_jobs()

    def list_jobs(self) -> list[CronJob]:
        """List all cron jobs."""
        return list(self._jobs.values())

    async def _check_jobs(self) -> None:
        """Check and trigger due jobs."""
        try:
            from croniter import croniter
        except ImportError:
            logger.warning("croniter not installed, cron disabled")
            return

        now = datetime.now()

        for job in self._jobs.values():
            if not job.enabled:
                continue

            try:
                cron = croniter(job.schedule, job.last_run or now)
                next_run = cron.get_next(datetime)

                if next_run <= now:
                    await self._trigger_job(job)
                    job.last_run = now

            except Exception as e:
                logger.error(f"Cron job error ({job.name}): {e}")

    async def _trigger_job(self, job: CronJob) -> None:
        """Trigger a cron job."""
        logger.info(f"Triggering cron job: {job.name}")

        if self.on_job:
            await self.on_job(job.to_dict())

    async def _load_jobs(self) -> None:
        """Load jobs from storage."""
        if not self.store_path.exists():
            return

        try:
            async with aiofiles.open(self.store_path, "r") as f:
                data = json.loads(await f.read())

            for job_data in data:
                job = CronJob.from_dict(job_data)
                self._jobs[job.id] = job

            logger.debug(f"Loaded {len(self._jobs)} cron jobs")
        except Exception as e:
            logger.error(f"Error loading cron jobs: {e}")

    async def _save_jobs(self) -> None:
        """Save jobs to storage."""
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            data = [job.to_dict() for job in self._jobs.values()]
            async with aiofiles.open(self.store_path, "w") as f:
                await f.write(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Error saving cron jobs: {e}")


__all__ = ["CronService", "CronJob"]
