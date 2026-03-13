"""Tool registry for ClawScope."""

from __future__ import annotations

import inspect
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from loguru import logger

from clawscope.config import ToolsConfig
from clawscope.exception import ToolNotFoundError, ToolExecutionError


@dataclass
class ToolParameter:
    """Tool parameter definition."""

    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    enum: list[str] | None = None


@dataclass
class Tool:
    """Tool definition."""

    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    func: Callable[..., Awaitable[str]] | None = None
    enabled: bool = True

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format."""
        properties = {}
        required = []

        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


class ToolRegistry:
    """
    Registry for managing tools.

    Features:
    - Dynamic tool registration
    - Built-in tool loading
    - Tool execution with validation
    - OpenAI-compatible schema generation
    """

    def __init__(self, config: ToolsConfig | None = None):
        """
        Initialize tool registry.

        Args:
            config: Tool configuration
        """
        self.config = config or ToolsConfig()
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def register_function(
        self,
        func: Callable[..., Awaitable[str]],
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        """
        Register a function as a tool.

        Args:
            func: Async function to register
            name: Tool name (defaults to function name)
            description: Tool description (from docstring if not provided)
        """
        tool_name = name or func.__name__
        tool_description = description or (func.__doc__ or "").strip()

        # Parse parameters from signature
        sig = inspect.signature(func)
        parameters = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            # Determine type
            param_type = "string"
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"

            parameters.append(ToolParameter(
                name=param_name,
                type=param_type,
                description=f"Parameter: {param_name}",
                required=param.default == inspect.Parameter.empty,
                default=None if param.default == inspect.Parameter.empty else param.default,
            ))

        tool = Tool(
            name=tool_name,
            description=tool_description,
            parameters=parameters,
            func=func,
        )
        self.register(tool)

    def unregister(self, name: str) -> None:
        """Unregister a tool."""
        if name in self._tools:
            del self._tools[name]
            logger.debug(f"Unregistered tool: {name}")

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_definitions(self) -> list[dict[str, Any]]:
        """Get all tool definitions in OpenAI format."""
        definitions = []
        for tool in self._tools.values():
            if tool.enabled:
                definitions.append(tool.to_openai_schema())
        return definitions

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """
        Execute a tool.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool execution result

        Raises:
            ToolNotFoundError: If tool not found
            ToolExecutionError: If execution fails
        """
        tool = self.get(name)
        if not tool:
            raise ToolNotFoundError(f"Tool not found: {name}")

        if not tool.func:
            raise ToolExecutionError(f"Tool has no function: {name}")

        try:
            logger.info(f"Executing tool: {name}")
            result = await tool.func(**arguments)

            # Truncate long results
            max_length = self.config.max_output_length
            if len(result) > max_length:
                result = result[:max_length] + "\n... [truncated]"

            return result

        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            raise ToolExecutionError(f"Tool execution failed: {str(e)}")

    async def load_builtin_tools(self) -> None:
        """Load built-in tools based on configuration."""
        from clawscope.tool.builtin import get_builtin_tools

        for tool in get_builtin_tools(self.config):
            if tool.name in self.config.enabled:
                self.register(tool)

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())


__all__ = ["ToolRegistry", "Tool", "ToolParameter"]
