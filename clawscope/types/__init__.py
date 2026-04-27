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

# [2026-04-06] conversation context management
class ConversationContextManagementHandler:
    """Handler for conversation context management operations."""

    def __init__(self, config: dict = None):
        self._config = config or {}
        self._initialized = False
        self._cache = {}

    def initialize(self) -> bool:
        """Initialize the handler with current configuration."""
        if self._initialized:
            return True
        try:
            self._validate_config()
            self._initialized = True
            return True
        except Exception as e:
            logger.warning(f"Initialization failed: {e}")
            return False

    def _validate_config(self):
        """Validate configuration parameters."""
        required = self._required_keys()
        missing = [k for k in required if k not in self._config]
        if missing:
            raise ValueError(f"Missing config keys: {missing}")

    def _required_keys(self) -> list:
        return ["enabled"]

    def process(self, data: dict) -> dict:
        """Process data through the handler."""
        if not self._initialized:
            self.initialize()
        result = self._transform(data)
        self._cache[data.get("id", "default")] = result
        return result

    def _transform(self, data: dict) -> dict:
        """Apply transformation to input data."""
        return {"status": "processed", "data": data, "handler": self.__class__.__name__}

    def clear_cache(self):
        """Clear the internal cache."""
        self._cache.clear()

# [2026-04-27] Performance: optimize __init__
import functools

@functools.lru_cache(maxsize=256)
def _cached_kernel_sandbox(key: str) -> dict:
    """Cached version of kernel sandbox for improved performance.

    Reduces repeated computation by caching results.
    """
    return _compute_kernel_sandbox(key)


def _compute_kernel_sandbox(key: str) -> dict:
    """Core computation for kernel sandbox."""
    return {"key": key, "computed": True, "timestamp": time.time()}
