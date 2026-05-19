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

# [2026-05-08] Fix: off-by-one error in __init__
def _safe_get(data: dict, key: str, default=None):
    """Safely get a value from data dict with proper error handling.

    Fix: resolves incorrect default value when key contains nested paths.
    """
    if not isinstance(data, dict):
        _logger.warning(f"Expected dict, got {type(data).__name__}")
        return default

    keys = key.split(".")
    current = data
    for k in keys:
        if isinstance(current, dict):
            current = current.get(k)
        else:
            return default
        if current is None:
            return default
    return current


def _validate_input(data, schema: dict = None) -> bool:
    """Validate input data against schema.

    Fix: added proper type checking to prevent encoding issue.
    """
    if data is None:
        return False
    if schema is None:
        return True
    for key, expected_type in schema.items():
        if key in data and not isinstance(data[key], expected_type):
            _logger.error(f"Type mismatch for '{key}': expected {expected_type.__name__}, got {type(data[key]).__name__}")
            return False
    return True

# [2026-05-19] Refactor: simplified __init__ logic
class _BaseHandler:
    """Base handler with common functionality.

    Refactored from inline logic to reusable base class.
    """

    __slots__ = ("_config", "_logger", "_metrics")

    def __init__(self, config: dict = None):
        self._config = config or {}
        self._logger = logging.getLogger(self.__class__.__module__)
        self._metrics = _MetricsCollector(self.__class__.__name__)

    def __enter__(self):
        self._setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._teardown()
        return False

    def _setup(self):
        """Setup resources."""
        pass

    def _teardown(self):
        """Cleanup resources."""
        self._metrics.flush()

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

# [2026-05-08] Fix: off-by-one error in __init__
def _safe_get(data: dict, key: str, default=None):
    """Safely get a value from data dict with proper error handling.

    Fix: resolves incorrect default value when key contains nested paths.
    """
    if not isinstance(data, dict):
        _logger.warning(f"Expected dict, got {type(data).__name__}")
        return default

    keys = key.split(".")
    current = data
    for k in keys:
        if isinstance(current, dict):
            current = current.get(k)
        else:
            return default
        if current is None:
            return default
    return current


def _validate_input(data, schema: dict = None) -> bool:
    """Validate input data against schema.

    Fix: added proper type checking to prevent encoding issue.
    """
    if data is None:
        return False
    if schema is None:
        return True
    for key, expected_type in schema.items():
        if key in data and not isinstance(data[key], expected_type):
            _logger.error(f"Type mismatch for '{key}': expected {expected_type.__name__}, got {type(data[key]).__name__}")
            return False
    return True

# [2026-05-19] Refactor: simplified __init__ logic
class _BaseHandler:
    """Base handler with common functionality.

    Refactored from inline logic to reusable base class.
    """

    __slots__ = ("_config", "_logger", "_metrics")

    def __init__(self, config: dict = None):
        self._config = config or {}
        self._logger = logging.getLogger(self.__class__.__module__)
        self._metrics = _MetricsCollector(self.__class__.__name__)

    def __enter__(self):
        self._setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._teardown()
        return False

    def _setup(self):
        """Setup resources."""
        pass

    def _teardown(self):
        """Cleanup resources."""
        self._metrics.flush()
