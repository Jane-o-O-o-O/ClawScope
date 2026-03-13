"""Message bus event types for ClawScope."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import shortuuid


@dataclass
class InboundMessage:
    """
    Inbound message from a channel.

    Represents a message received from any chat platform
    (Telegram, Discord, Slack, etc.) heading to the agent.
    """

    channel: str
    sender_id: str
    chat_id: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    media: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: shortuuid.uuid())
    session_key_override: str | None = None

    @property
    def session_key(self) -> str:
        """Get session key for this message."""
        if self.session_key_override:
            return self.session_key_override
        return f"{self.channel}:{self.chat_id}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "channel": self.channel,
            "sender_id": self.sender_id,
            "chat_id": self.chat_id,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "media": self.media,
            "metadata": self.metadata,
            "id": self.id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InboundMessage":
        """Create from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now()

        return cls(
            channel=data.get("channel", "unknown"),
            sender_id=data.get("sender_id", "unknown"),
            chat_id=data.get("chat_id", "unknown"),
            content=data.get("content", ""),
            timestamp=timestamp,
            media=data.get("media", []),
            metadata=data.get("metadata", {}),
            id=data.get("id", shortuuid.uuid()),
            session_key_override=data.get("session_key_override"),
        )


@dataclass
class OutboundMessage:
    """
    Outbound message to a channel.

    Represents a message from the agent heading to a chat platform.
    """

    channel: str
    chat_id: str
    content: str
    media: list[str] = field(default_factory=list)
    reply_to: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: shortuuid.uuid())
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "channel": self.channel,
            "chat_id": self.chat_id,
            "content": self.content,
            "media": self.media,
            "reply_to": self.reply_to,
            "metadata": self.metadata,
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OutboundMessage":
        """Create from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now()

        return cls(
            channel=data.get("channel", "unknown"),
            chat_id=data.get("chat_id", "unknown"),
            content=data.get("content", ""),
            media=data.get("media", []),
            reply_to=data.get("reply_to"),
            metadata=data.get("metadata", {}),
            id=data.get("id", shortuuid.uuid()),
            timestamp=timestamp,
        )


@dataclass
class SystemEvent:
    """
    System-level event for internal communication.

    Used for coordination between components (e.g., shutdown, health checks).
    """

    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    id: str = field(default_factory=lambda: shortuuid.uuid())


__all__ = [
    "InboundMessage",
    "OutboundMessage",
    "SystemEvent",
]
