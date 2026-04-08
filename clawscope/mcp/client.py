"""
MCP (Model Context Protocol) client for ClawScope.

Connects to any MCP-compliant server (filesystem, databases, APIs, etc.)
and imports their tools into the ClawScope ToolRegistry so agents can
use them transparently alongside built-in tools.

Usage::

    from clawscope.mcp import MCPClient, StdioServerConfig

    client = MCPClient(StdioServerConfig(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        name="filesystem",
    ))

    await client.connect()
    count = await client.register_tools(tool_registry)
    print(f"Registered {count} MCP tools")

    # Later
    await client.disconnect()
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from clawscope.tool import ToolRegistry


# ---------------------------------------------------------------------------
# Server configuration
# ---------------------------------------------------------------------------


@dataclass
class StdioServerConfig:
    """
    Configuration for a stdio-based MCP server (most common).

    The server is launched as a subprocess; ClawScope communicates with it
    over its stdin/stdout.
    """

    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    name: str = "mcp"
    description: str = ""


@dataclass
class HttpServerConfig:
    """
    Configuration for an HTTP/SSE-based MCP server.
    """

    url: str
    headers: dict[str, str] = field(default_factory=dict)
    name: str = "mcp-http"
    description: str = ""


ServerConfig = StdioServerConfig | HttpServerConfig


# ---------------------------------------------------------------------------
# Tool info (mirrors mcp.types.Tool but avoids mandatory SDK dependency at
# import time — falls back gracefully when mcp is not installed)
# ---------------------------------------------------------------------------


@dataclass
class MCPToolInfo:
    """Lightweight descriptor for a tool exposed by an MCP server."""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mcp(cls, mcp_tool: Any) -> "MCPToolInfo":
        """Convert an mcp.types.Tool to MCPToolInfo."""
        return cls(
            name=mcp_tool.name,
            description=mcp_tool.description or "",
            input_schema=dict(mcp_tool.inputSchema) if mcp_tool.inputSchema else {},
        )


# ---------------------------------------------------------------------------
# MCPClient
# ---------------------------------------------------------------------------


class MCPClient:
    """
    Client that connects to an MCP server and integrates it with ClawScope.

    Lifecycle::

        client = MCPClient(config)
        await client.connect()           # opens transport + initialises session
        tools = await client.list_tools()
        result = await client.call_tool("read_file", {"path": "/tmp/a.txt"})
        count = await client.register_tools(tool_registry)
        await client.disconnect()

    The client can also be used as an async context manager::

        async with MCPClient(config) as client:
            await client.register_tools(registry)
    """

    def __init__(self, config: ServerConfig):
        self.config = config
        self.name: str = config.name
        self._session = None
        self._ctx = None          # async context manager stack
        self._connected = False

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the transport and initialise the MCP session."""
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            raise ImportError(
                "mcp package not installed. "
                "Run: pip install clawscope[mcp]"
            )

        if self._connected:
            return

        if isinstance(self.config, StdioServerConfig):
            params = StdioServerParameters(
                command=self.config.command,
                args=self.config.args,
                env=self.config.env,
            )
            self._ctx = stdio_client(params)
            read, write = await self._ctx.__aenter__()
            self._session_ctx = ClientSession(read, write)
            self._session = await self._session_ctx.__aenter__()
            await self._session.initialize()

        elif isinstance(self.config, HttpServerConfig):
            try:
                from mcp.client.sse import sse_client  # type: ignore[import]
            except ImportError:
                raise ImportError("HTTP/SSE MCP transport requires mcp[cli]")
            self._ctx = sse_client(
                self.config.url,
                headers=self.config.headers,
            )
            read, write = await self._ctx.__aenter__()
            self._session_ctx = ClientSession(read, write)
            self._session = await self._session_ctx.__aenter__()
            await self._session.initialize()

        else:
            raise ValueError(f"Unknown server config type: {type(self.config)}")

        self._connected = True
        logger.info(f"MCP client '{self.name}' connected")

    async def disconnect(self) -> None:
        """Close the session and transport."""
        if not self._connected:
            return
        try:
            if self._session_ctx:
                await self._session_ctx.__aexit__(None, None, None)
            if self._ctx:
                await self._ctx.__aexit__(None, None, None)
        except Exception as exc:
            logger.warning(f"MCP client '{self.name}' disconnect error: {exc}")
        finally:
            self._session = None
            self._ctx = None
            self._connected = False
            logger.info(f"MCP client '{self.name}' disconnected")

    async def __aenter__(self) -> "MCPClient":
        await self.connect()
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.disconnect()

    # ------------------------------------------------------------------
    # Tool operations
    # ------------------------------------------------------------------

    async def list_tools(self) -> list[MCPToolInfo]:
        """Return all tools exposed by the connected MCP server."""
        self._require_connected()
        result = await self._session.list_tools()
        tools = [MCPToolInfo.from_mcp(t) for t in result.tools]
        logger.debug(f"MCP '{self.name}': {len(tools)} tools available")
        return tools

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> str:
        """
        Call an MCP tool and return its output as a plain string.

        Concatenates all text content blocks from the response.
        If the response is an error, raises RuntimeError.
        """
        self._require_connected()
        result = await self._session.call_tool(name, arguments or {})

        if result.isError:
            # Extract error text
            error_text = _extract_text(result.content)
            raise RuntimeError(
                f"MCP tool '{name}' returned an error: {error_text}"
            )

        return _extract_text(result.content)

    # ------------------------------------------------------------------
    # ToolRegistry integration
    # ------------------------------------------------------------------

    async def register_tools(
        self,
        tool_registry: "ToolRegistry",
        prefix: str | None = None,
    ) -> int:
        """
        Discover and register all MCP server tools into *tool_registry*.

        Args:
            tool_registry: ClawScope ToolRegistry to populate.
            prefix: Optional name prefix (e.g. ``"fs_"`` → ``"fs_read_file"``).
                    Defaults to ``"{server_name}_"`` when *server_name* is set
                    and ``""`` otherwise.

        Returns:
            Number of tools registered.
        """
        tools = await self.list_tools()
        if prefix is None:
            prefix = f"{self.name}_" if self.name != "mcp" else ""

        client_ref = self
        count = 0

        for tool_info in tools:
            tool_name = f"{prefix}{tool_info.name}"
            _register_one(tool_registry, client_ref, tool_info, tool_name)
            count += 1

        logger.info(
            f"MCP client '{self.name}': registered {count} tools "
            f"(prefix={prefix!r})"
        )
        return count

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_connected(self) -> None:
        if not self._connected or self._session is None:
            raise RuntimeError(
                f"MCP client '{self.name}' is not connected. "
                "Call await client.connect() first."
            )

    @property
    def is_connected(self) -> bool:
        return self._connected


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _extract_text(content: list[Any]) -> str:
    """Concatenate all text content blocks into a single string."""
    parts = []
    for block in content:
        if hasattr(block, "text"):
            parts.append(block.text)
        elif isinstance(block, dict):
            parts.append(block.get("text", str(block)))
        else:
            parts.append(str(block))
    return "\n".join(parts)


def _register_one(
    tool_registry: "ToolRegistry",
    client: MCPClient,
    tool_info: MCPToolInfo,
    registered_name: str,
) -> None:
    """Register a single MCPToolInfo as a ClawScope tool."""
    from clawscope.tool.registry import Tool, ToolParameter

    # Build parameter list from JSON Schema
    params: list[ToolParameter] = []
    schema = tool_info.input_schema or {}
    properties = schema.get("properties", {})
    required_set = set(schema.get("required", []))

    for param_name, param_def in properties.items():
        params.append(
            ToolParameter(
                name=param_name,
                type=param_def.get("type", "string"),
                description=param_def.get("description", ""),
                required=param_name in required_set,
                default=param_def.get("default"),
                enum=param_def.get("enum"),
            )
        )

    # Async wrapper that calls the MCP server
    async def _call(**kwargs: Any) -> str:
        return await client.call_tool(tool_info.name, kwargs)

    _call.__name__ = registered_name
    _call.__doc__ = tool_info.description

    mcp_tool = Tool(
        name=registered_name,
        description=tool_info.description,
        parameters=params,
        func=_call,
        enabled=True,
    )

    tool_registry._tools[registered_name] = mcp_tool
    logger.debug(f"Registered MCP tool: {registered_name}")


__all__ = [
    "MCPClient",
    "MCPToolInfo",
    "StdioServerConfig",
    "HttpServerConfig",
    "ServerConfig",
]
