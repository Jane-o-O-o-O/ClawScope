"""Base memory interface for ClawScope."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from clawscope.message import Msg


class MemoryBase(ABC):
    """Abstract base class for memory systems."""

    @abstractmethod
    async def add(self, messages: list[Msg], mark: str | None = None) -> None:
        """
        Add messages to memory.

        Args:
            messages: Messages to add
            mark: Optional marker for the messages
        """
        pass

    @abstractmethod
    async def get(
        self,
        mark: str | None = None,
        limit: int | None = None,
    ) -> list[Msg]:
        """
        Get messages from memory.

        Args:
            mark: Optional marker to filter by
            limit: Maximum number of messages to return

        Returns:
            List of messages
        """
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all messages from memory."""
        pass

    @abstractmethod
    async def size(self) -> int:
        """Get number of messages in memory."""
        pass


__all__ = ["MemoryBase"]
