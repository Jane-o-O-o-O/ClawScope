"""ClawScope message system."""

from importlib import import_module
from typing import Any

from clawscope.message.base import (
    AudioBlock,
    ContentBlock,
    ImageBlock,
    Msg,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)

__all__ = [
    "Msg",
    "ContentBlock",
    "TextBlock",
    "ImageBlock",
    "AudioBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    "UnifiedMessage",
    "MessageAdapter",
]


_LAZY_IMPORTS = {
    "UnifiedMessage": ("clawscope.message.unified", "UnifiedMessage"),
    "MessageAdapter": ("clawscope.message.adapters", "MessageAdapter"),
}


def __getattr__(name: str) -> Any:
    """Lazily import higher-level message helpers."""
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module 'clawscope.message' has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
