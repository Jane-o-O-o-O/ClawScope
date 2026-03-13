"""Base channel interface for ClawScope."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from clawscope.bus import MessageBus, OutboundMessage


class BaseChannel(ABC):
    """
    Abstract base class for chat channels.

    Channels handle communication with external platforms
    (Telegram, Discord, Slack, etc.) and bridge them to
    the ClawScope message bus.
    """

    def __init__(
        self,
        name: str,
        bus: "MessageBus",
        config: Any,
    ):
        """
        Initialize channel.

        Args:
            name: Channel name
            bus: Message bus instance
            config: Channel-specific configuration
        """
        self.name = name
        self.bus = bus
        self.config = config
        self._running = False

    @abstractmethod
    async def start(self) -> None:
        """
        Start the channel.

        Begin listening for messages from the platform.
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """
        Stop the channel.

        Clean up connections and stop listening.
        """
        pass

    @abstractmethod
    async def send(self, message: "OutboundMessage") -> None:
        """
        Send a message to the platform.

        Args:
            message: Outbound message to send
        """
        pass

    def is_allowed(self, sender_id: str) -> bool:
        """
        Check if sender is allowed.

        Args:
            sender_id: Sender identifier

        Returns:
            True if sender is allowed
        """
        allow_from = getattr(self.config, "allow_from", ["*"])
        if "*" in allow_from:
            return True
        return sender_id in allow_from

    @property
    def is_running(self) -> bool:
        """Check if channel is running."""
        return self._running

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, running={self._running})"


__all__ = ["BaseChannel"]
