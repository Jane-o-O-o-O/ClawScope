"""ClawScope type definitions."""

from typing import Any, Literal, TypeAlias, TypeVar
from collections.abc import Sequence, Callable, Awaitable

# Role types
Role: TypeAlias = Literal["user", "assistant", "system", "tool"]

# JSON serializable types
JSONPrimitive: TypeAlias = str | int | float | bool | None
JSONObject: TypeAlias = dict[str, "JSONValue"]
JSONArray: TypeAlias = list["JSONValue"]
JSONValue: TypeAlias = JSONPrimitive | JSONObject | JSONArray

# Message content types
TextContent: TypeAlias = str
ContentBlock: TypeAlias = dict[str, Any]
MessageContent: TypeAlias = TextContent | Sequence[ContentBlock]

# Callback types
T = TypeVar("T")
AsyncCallback: TypeAlias = Callable[..., Awaitable[T]]
SyncCallback: TypeAlias = Callable[..., T]

# Provider types
ProviderType: TypeAlias = Literal["agentscope", "litellm", "direct"]

# Tool types
ToolSchema: TypeAlias = dict[str, Any]
ToolResult: TypeAlias = str | dict[str, Any]

# Channel types
ChannelType: TypeAlias = Literal[
    "telegram",
    "discord",
    "slack",
    "feishu",
    "dingtalk",
    "wecom",
    "whatsapp",
    "qq",
    "matrix",
    "email",
    "cli",
    "api",
]

# Memory types
MemoryBackend: TypeAlias = Literal["in_memory", "redis", "sqlalchemy"]
SessionBackend: TypeAlias = Literal["jsonl", "redis", "sqlite"]

__all__ = [
    "Role",
    "JSONPrimitive",
    "JSONObject",
    "JSONArray",
    "JSONValue",
    "TextContent",
    "ContentBlock",
    "MessageContent",
    "AsyncCallback",
    "SyncCallback",
    "ProviderType",
    "ToolSchema",
    "ToolResult",
    "ChannelType",
    "MemoryBackend",
    "SessionBackend",
]
