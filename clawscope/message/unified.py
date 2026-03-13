"""Unified message protocol for ClawScope."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

import shortuuid

from clawscope.message.base import Msg, ContentBlock, TextBlock, ImageBlock
from clawscope.types import Role, JSONValue


@dataclass
class UnifiedMessage:
    """
    ClawScope Unified Message Format.

    Bridges AgentScope Msg and Nanobot InboundMessage/OutboundMessage
    formats, providing a single interface for the entire platform.
    """

    # Core fields (AgentScope compatible)
    name: str
    content: str | Sequence[ContentBlock]
    role: Role = "user"

    # Channel information (Nanobot compatible)
    channel: str | None = None
    sender_id: str | None = None
    chat_id: str | None = None

    # Metadata
    id: str = field(default_factory=lambda: shortuuid.uuid())
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, JSONValue] = field(default_factory=dict)
    media: list[str] = field(default_factory=list)

    # Reply context
    reply_to: str | None = None

    @property
    def session_key(self) -> str:
        """Get session key for this message."""
        if self.channel and self.chat_id:
            return f"{self.channel}:{self.chat_id}"
        return f"default:{self.id}"

    def get_text_content(self) -> str:
        """Extract text content from message."""
        if isinstance(self.content, str):
            return self.content

        text_parts = []
        for block in self.content:
            if isinstance(block, TextBlock):
                text_parts.append(block.text)
            elif isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))

        return "\n".join(text_parts)

    def to_agentscope_msg(self) -> Msg:
        """Convert to AgentScope Msg format."""
        metadata = dict(self.metadata)
        if self.channel:
            metadata["channel"] = self.channel
        if self.chat_id:
            metadata["chat_id"] = self.chat_id
        if self.sender_id:
            metadata["sender_id"] = self.sender_id

        return Msg(
            name=self.name,
            content=self.content,
            role=self.role,
            id=self.id,
            timestamp=self.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        content = self.content
        if isinstance(content, list):
            content = [
                block.to_dict() if isinstance(block, ContentBlock) else block
                for block in content
            ]

        return {
            "name": self.name,
            "content": content,
            "role": self.role,
            "channel": self.channel,
            "sender_id": self.sender_id,
            "chat_id": self.chat_id,
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "media": self.media,
            "reply_to": self.reply_to,
        }

    @classmethod
    def from_agentscope_msg(
        cls,
        msg: Msg,
        channel: str | None = None,
        chat_id: str | None = None,
    ) -> "UnifiedMessage":
        """Create from AgentScope Msg."""
        metadata = dict(msg.metadata)
        channel = channel or metadata.pop("channel", None)
        chat_id = chat_id or metadata.pop("chat_id", None)
        sender_id = metadata.pop("sender_id", None)

        # Parse timestamp
        try:
            timestamp = datetime.strptime(msg.timestamp, "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            timestamp = datetime.now()

        # Extract media URLs from content blocks
        media = []
        if isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, ImageBlock):
                    media.append(block.source)
                elif isinstance(block, dict) and block.get("type") == "image":
                    source = block.get("source", {})
                    if source.get("type") == "url":
                        media.append(source.get("url", ""))

        return cls(
            name=msg.name,
            content=msg.content,
            role=msg.role,
            channel=channel,
            sender_id=sender_id,
            chat_id=chat_id,
            id=msg.id,
            timestamp=timestamp,
            metadata=metadata,
            media=media,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UnifiedMessage":
        """Create from dictionary."""
        content = data.get("content", "")
        if isinstance(content, list):
            content = [
                ContentBlock.from_dict(block) if isinstance(block, dict) else block
                for block in content
            ]

        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp)
            except ValueError:
                timestamp = datetime.now()
        elif timestamp is None:
            timestamp = datetime.now()

        return cls(
            name=data.get("name", ""),
            content=content,
            role=data.get("role", "user"),
            channel=data.get("channel"),
            sender_id=data.get("sender_id"),
            chat_id=data.get("chat_id"),
            id=data.get("id", shortuuid.uuid()),
            timestamp=timestamp,
            metadata=data.get("metadata", {}),
            media=data.get("media", []),
            reply_to=data.get("reply_to"),
        )

    def __str__(self) -> str:
        text = self.get_text_content()[:50]
        return f"UnifiedMessage(channel={self.channel}, name={self.name}, content={text}...)"


__all__ = ["UnifiedMessage"]
