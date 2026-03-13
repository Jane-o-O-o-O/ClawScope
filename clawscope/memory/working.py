"""Working memory implementations for ClawScope."""

from __future__ import annotations

from typing import Any

from clawscope.memory.base import MemoryBase
from clawscope.message import Msg


class InMemoryMemory(MemoryBase):
    """
    Simple in-memory message storage.

    Messages are stored in a list and lost when the process ends.
    Suitable for testing and short-lived sessions.
    """

    def __init__(self, max_messages: int = 1000):
        """
        Initialize in-memory storage.

        Args:
            max_messages: Maximum messages to retain
        """
        self.max_messages = max_messages
        self._messages: list[Msg] = []
        self._marks: dict[str, int] = {}  # mark -> index

    async def add(self, messages: list[Msg], mark: str | None = None) -> None:
        """Add messages to memory."""
        if mark:
            self._marks[mark] = len(self._messages)

        self._messages.extend(messages)

        # Trim if exceeds max
        if len(self._messages) > self.max_messages:
            excess = len(self._messages) - self.max_messages
            self._messages = self._messages[excess:]
            # Adjust marks
            self._marks = {
                k: max(0, v - excess)
                for k, v in self._marks.items()
            }

    async def get(
        self,
        mark: str | None = None,
        limit: int | None = None,
    ) -> list[Msg]:
        """Get messages from memory."""
        if mark and mark in self._marks:
            start = self._marks[mark]
            messages = self._messages[start:]
        else:
            messages = self._messages

        if limit:
            messages = messages[-limit:]

        return list(messages)

    async def clear(self) -> None:
        """Clear all messages."""
        self._messages.clear()
        self._marks.clear()

    async def size(self) -> int:
        """Get number of messages."""
        return len(self._messages)

    async def get_since_mark(self, mark: str) -> list[Msg]:
        """Get messages since a specific mark."""
        if mark not in self._marks:
            return []
        start = self._marks[mark]
        return self._messages[start:]


__all__ = ["InMemoryMemory"]
