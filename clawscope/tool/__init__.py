"""ClawScope tool system."""

from clawscope.tool.registry import ToolRegistry, Tool
from clawscope.tool.decorator import tool

__all__ = [
    "ToolRegistry",
    "Tool",
    "tool",
]
