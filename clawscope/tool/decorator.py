"""Tool decorator for ClawScope."""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Awaitable, TypeVar

from clawscope.tool.registry import Tool, ToolParameter

T = TypeVar("T")


def tool(
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Callable[..., Awaitable[str]]], Callable[..., Awaitable[str]]]:
    """
    Decorator to mark a function as a tool.

    Usage:
        @tool(name="search", description="Search the web")
        async def search(query: str) -> str:
            '''Search for information.

            Args:
                query: Search query
            '''
            return "results..."

    Args:
        name: Tool name (defaults to function name)
        description: Tool description (defaults to docstring)

    Returns:
        Decorated function with tool metadata
    """
    def decorator(func: Callable[..., Awaitable[str]]) -> Callable[..., Awaitable[str]]:
        tool_name = name or func.__name__
        tool_description = description or _parse_description(func)

        # Parse parameters
        parameters = _parse_parameters(func)

        # Create tool and attach to function
        tool_obj = Tool(
            name=tool_name,
            description=tool_description,
            parameters=parameters,
            func=func,
        )

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> str:
            return await func(*args, **kwargs)

        # Attach tool metadata
        wrapper._tool = tool_obj
        wrapper._is_tool = True

        return wrapper

    return decorator


def _parse_description(func: Callable) -> str:
    """Parse description from docstring."""
    doc = func.__doc__
    if not doc:
        return f"Tool: {func.__name__}"

    # Get first paragraph
    lines = doc.strip().split("\n\n")[0].split("\n")
    return " ".join(line.strip() for line in lines)


def _parse_parameters(func: Callable) -> list[ToolParameter]:
    """Parse parameters from function signature and docstring."""
    sig = inspect.signature(func)
    doc = func.__doc__ or ""

    # Parse docstring for parameter descriptions
    param_docs = _parse_docstring_params(doc)

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
            elif param.annotation == list:
                param_type = "array"
            elif param.annotation == dict:
                param_type = "object"

        parameters.append(ToolParameter(
            name=param_name,
            type=param_type,
            description=param_docs.get(param_name, f"The {param_name} parameter"),
            required=param.default == inspect.Parameter.empty,
            default=None if param.default == inspect.Parameter.empty else param.default,
        ))

    return parameters


def _parse_docstring_params(doc: str) -> dict[str, str]:
    """Parse parameter descriptions from docstring."""
    params = {}

    # Look for Args: section
    in_args = False
    current_param = None
    current_desc = []

    for line in doc.split("\n"):
        stripped = line.strip()

        if stripped.lower().startswith("args:"):
            in_args = True
            continue
        elif stripped.lower().startswith(("returns:", "raises:", "yields:", "examples:")):
            in_args = False
            if current_param:
                params[current_param] = " ".join(current_desc).strip()
            continue

        if in_args:
            # Check for new parameter
            if ":" in stripped and not stripped.startswith(" "):
                # Save previous parameter
                if current_param:
                    params[current_param] = " ".join(current_desc).strip()

                # Parse new parameter
                parts = stripped.split(":", 1)
                current_param = parts[0].strip()
                current_desc = [parts[1].strip()] if len(parts) > 1 else []
            elif current_param and stripped:
                # Continuation of description
                current_desc.append(stripped)

    # Save last parameter
    if current_param:
        params[current_param] = " ".join(current_desc).strip()

    return params


__all__ = ["tool"]
