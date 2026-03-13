"""Base classes for model providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal

from clawscope.message import Msg


@dataclass
class ToolCall:
    """Represents a tool call request from the model."""

    id: str
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolCall":
        """Create from dictionary."""
        func = data.get("function", {})
        return cls(
            id=data.get("id", ""),
            name=func.get("name", ""),
            arguments=func.get("arguments", {}),
        )


@dataclass
class UsageInfo:
    """Token usage information."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
        }


@dataclass
class ChatResponse:
    """Response from a chat model."""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: Literal["stop", "tool_calls", "length", "content_filter"] | None = None
    usage: UsageInfo = field(default_factory=UsageInfo)
    raw_response: Any = None

    # Extended thinking support (Claude, DeepSeek-R1, etc.)
    thinking_content: str | None = None
    thinking_blocks: list[dict[str, Any]] = field(default_factory=list)

    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return bool(self.tool_calls)

    def to_msg(self, name: str = "assistant") -> Msg:
        """Convert to Msg."""
        from clawscope.message import TextBlock, ToolUseBlock

        content_blocks = []

        # Add thinking if present
        if self.thinking_content:
            content_blocks.append({
                "type": "thinking",
                "thinking": self.thinking_content,
            })

        # Add text content
        if self.content:
            content_blocks.append(TextBlock(text=self.content))

        # Add tool calls
        for tool_call in self.tool_calls:
            content_blocks.append(ToolUseBlock(
                id=tool_call.id,
                name=tool_call.name,
                input=tool_call.arguments,
            ))

        return Msg(
            name=name,
            content=content_blocks if content_blocks else (self.content or ""),
            role="assistant",
        )


class ChatModelBase(ABC):
    """
    Abstract base class for chat models.

    Provides a unified interface for all LLM providers.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        api_base: str | None = None,
        stream: bool = True,
        timeout: int = 120,
        max_retries: int = 3,
        **kwargs: Any,
    ):
        """
        Initialize chat model.

        Args:
            model_name: Name of the model to use
            api_key: API key for authentication
            api_base: Base URL for API
            stream: Whether to stream responses
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            **kwargs: Additional provider-specific options
        """
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.stream = stream
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_options = kwargs

    @abstractmethod
    async def chat(
        self,
        messages: list[Msg],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """
        Send chat request to model.

        Args:
            messages: List of messages
            tools: Optional list of tool definitions
            tool_choice: Tool choice strategy ("auto", "none", "required", or specific tool)
            **kwargs: Additional options

        Returns:
            ChatResponse with model output
        """
        pass

    @abstractmethod
    async def stream_chat(
        self,
        messages: list[Msg],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatResponse]:
        """
        Stream chat response from model.

        Args:
            messages: List of messages
            tools: Optional list of tool definitions
            tool_choice: Tool choice strategy
            **kwargs: Additional options

        Yields:
            ChatResponse chunks
        """
        pass

    async def __call__(
        self,
        messages: list[Msg],
        **kwargs: Any,
    ) -> ChatResponse:
        """Shorthand for chat method."""
        return await self.chat(messages, **kwargs)

    def _format_messages(self, messages: list[Msg]) -> list[dict[str, Any]]:
        """
        Format messages for API request.

        Override in subclasses for provider-specific formatting.
        """
        formatted = []
        for msg in messages:
            content = msg.content
            if isinstance(content, list):
                # Convert content blocks to API format
                content = [
                    block.to_dict() if hasattr(block, "to_dict") else block
                    for block in content
                ]

            formatted.append({
                "role": msg.role,
                "content": content,
                "name": msg.name if msg.role != "assistant" else None,
            })

        return formatted

    def _validate_tool_choice(
        self,
        tool_choice: str | dict | None,
        tools: list[dict[str, Any]] | None,
    ) -> str | dict | None:
        """Validate tool choice against available tools."""
        if tool_choice is None or tools is None:
            return None

        if isinstance(tool_choice, str):
            if tool_choice in ("auto", "none", "required"):
                return tool_choice
            # Assume it's a tool name
            tool_names = [t.get("function", {}).get("name") for t in tools]
            if tool_choice in tool_names:
                return {"type": "function", "function": {"name": tool_choice}}
            raise ValueError(f"Unknown tool: {tool_choice}")

        return tool_choice


__all__ = [
    "ChatModelBase",
    "ChatResponse",
    "ToolCall",
    "UsageInfo",
]
