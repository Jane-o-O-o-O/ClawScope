"""ClawScope exception definitions."""

from typing import Any


class ClawScopeError(Exception):
    """Base exception for ClawScope."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(ClawScopeError):
    """Configuration related errors."""

    pass


class ChannelError(ClawScopeError):
    """Channel related errors."""

    pass


class ChannelConnectionError(ChannelError):
    """Channel connection failed."""

    pass


class ChannelSendError(ChannelError):
    """Failed to send message through channel."""

    pass


class ModelError(ClawScopeError):
    """Model related errors."""

    pass


class ModelNotFoundError(ModelError):
    """Model provider not found."""

    pass


class ModelInvocationError(ModelError):
    """Model invocation failed."""

    pass


class AgentError(ClawScopeError):
    """Agent related errors."""

    pass


class AgentTimeoutError(AgentError):
    """Agent execution timeout."""

    pass


class ToolError(ClawScopeError):
    """Tool related errors."""

    pass


class ToolNotFoundError(ToolError):
    """Tool not found in registry."""

    pass


class ToolExecutionError(ToolError):
    """Tool execution failed."""

    pass


class MemoryError(ClawScopeError):
    """Memory related errors."""

    pass


class SessionError(ClawScopeError):
    """Session related errors."""

    pass


class MessageError(ClawScopeError):
    """Message related errors."""

    pass


class MessageAdapterError(MessageError):
    """Message adaptation failed."""

    pass
