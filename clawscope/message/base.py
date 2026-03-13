"""Base message classes for ClawScope."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Sequence

import shortuuid

from clawscope.types import Role, JSONValue


@dataclass
class ContentBlock:
    """Base class for message content blocks."""

    type: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {"type": self.type}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContentBlock":
        """Create from dictionary."""
        block_type = data.get("type", "text")
        if block_type == "text":
            return TextBlock.from_dict(data)
        elif block_type == "image":
            return ImageBlock.from_dict(data)
        elif block_type == "audio":
            return AudioBlock.from_dict(data)
        elif block_type == "tool_use":
            return ToolUseBlock.from_dict(data)
        elif block_type == "tool_result":
            return ToolResultBlock.from_dict(data)
        else:
            return ContentBlock(type=block_type)


@dataclass
class TextBlock(ContentBlock):
    """Text content block."""

    type: str = "text"
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "text": self.text}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TextBlock":
        return cls(text=data.get("text", ""))


@dataclass
class ImageBlock(ContentBlock):
    """Image content block."""

    type: str = "image"
    source_type: Literal["url", "base64"] = "url"
    source: str = ""
    media_type: str = "image/png"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "source": {
                "type": self.source_type,
                "data" if self.source_type == "base64" else "url": self.source,
                "media_type": self.media_type,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImageBlock":
        source = data.get("source", {})
        source_type = source.get("type", "url")
        return cls(
            source_type=source_type,
            source=source.get("data" if source_type == "base64" else "url", ""),
            media_type=source.get("media_type", "image/png"),
        )


@dataclass
class AudioBlock(ContentBlock):
    """Audio content block."""

    type: str = "audio"
    source_type: Literal["url", "base64"] = "url"
    source: str = ""
    media_type: str = "audio/mp3"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "source": {
                "type": self.source_type,
                "data" if self.source_type == "base64" else "url": self.source,
                "media_type": self.media_type,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AudioBlock":
        source = data.get("source", {})
        source_type = source.get("type", "url")
        return cls(
            source_type=source_type,
            source=source.get("data" if source_type == "base64" else "url", ""),
            media_type=source.get("media_type", "audio/mp3"),
        )


@dataclass
class ToolUseBlock(ContentBlock):
    """Tool use request block."""

    type: str = "tool_use"
    id: str = ""
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "id": self.id,
            "name": self.name,
            "input": self.input,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolUseBlock":
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            input=data.get("input", {}),
        )


@dataclass
class ToolResultBlock(ContentBlock):
    """Tool result block."""

    type: str = "tool_result"
    tool_use_id: str = ""
    content: str = ""
    is_error: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "tool_use_id": self.tool_use_id,
            "content": self.content,
            "is_error": self.is_error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolResultBlock":
        return cls(
            tool_use_id=data.get("tool_use_id", ""),
            content=data.get("content", ""),
            is_error=data.get("is_error", False),
        )


@dataclass
class Msg:
    """
    Core message class for ClawScope.

    Compatible with AgentScope Msg format while supporting
    additional channel metadata.
    """

    name: str
    content: str | Sequence[ContentBlock]
    role: Role = "user"
    id: str = field(default_factory=lambda: shortuuid.uuid())
    timestamp: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    )
    metadata: dict[str, JSONValue] = field(default_factory=dict)
    invocation_id: str | None = None

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

    def get_content_blocks(
        self, block_types: list[str] | None = None
    ) -> list[ContentBlock | dict]:
        """Get content blocks, optionally filtered by type."""
        if isinstance(self.content, str):
            return [TextBlock(text=self.content)]

        if block_types is None:
            return list(self.content)

        result = []
        for block in self.content:
            if isinstance(block, ContentBlock) and block.type in block_types:
                result.append(block)
            elif isinstance(block, dict) and block.get("type") in block_types:
                result.append(block)

        return result

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
            "id": self.id,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "invocation_id": self.invocation_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Msg":
        """Create from dictionary."""
        content = data.get("content", "")
        if isinstance(content, list):
            content = [
                ContentBlock.from_dict(block) if isinstance(block, dict) else block
                for block in content
            ]

        return cls(
            name=data.get("name", ""),
            content=content,
            role=data.get("role", "user"),
            id=data.get("id", shortuuid.uuid()),
            timestamp=data.get(
                "timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            ),
            metadata=data.get("metadata", {}),
            invocation_id=data.get("invocation_id"),
        )

    def __str__(self) -> str:
        return f"Msg(name={self.name}, role={self.role}, content={self.get_text_content()[:50]}...)"


__all__ = [
    "Msg",
    "ContentBlock",
    "TextBlock",
    "ImageBlock",
    "AudioBlock",
    "ToolUseBlock",
    "ToolResultBlock",
]
