"""
MCP Server for ClawScope.

Exposes ClawScope's ToolRegistry (and optionally SkillRegistry) as an
MCP-compliant server so that any MCP client — Claude Desktop, Continue,
Cursor, other agents — can discover and invoke ClawScope tools.

Two runtime modes:

stdio (default)::

    server = MCPServer(tool_registry, name="my-agent")
    await server.run_stdio()          # blocks; communicates over stdin/stdout

HTTP/SSE (experimental)::

    server = MCPServer(tool_registry)
    await server.run_http(host="0.0.0.0", port=8765)

You can also get the raw ``mcp.server.Server`` object and embed it in your
own transport layer::

    mcp_server = server.build()

CLI entry-point
---------------
Add to ``pyproject.toml``::

    [project.scripts]
    clawscope-mcp = "clawscope.mcp.server:run_stdio_cli"

Then users can configure it in Claude Desktop's ``claude_desktop_config.json``::

    {
      "mcpServers": {
        "clawscope": {
          "command": "clawscope-mcp",
          "args": ["--config", "~/.clawscope/config.yaml"]
        }
      }
    }
"""

from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from clawscope.tool import ToolRegistry
    from clawscope.skills import SkillRegistry


class MCPServer:
    """
    Wraps ClawScope's ToolRegistry as an MCP server.

    Args:
        tool_registry: ClawScope ToolRegistry to expose.
        skill_registry: Optional SkillRegistry; skills are exposed as
                        additional MCP tools prefixed with ``skill_``.
        name: Server name advertised to MCP clients.
        version: Server version string.
    """

    def __init__(
        self,
        tool_registry: "ToolRegistry",
        skill_registry: "SkillRegistry | None" = None,
        name: str = "clawscope",
        version: str = "0.1.0",
    ):
        self.tool_registry = tool_registry
        self.skill_registry = skill_registry
        self.name = name
        self.version = version

    # ------------------------------------------------------------------
    # Build the mcp.server.Server object
    # ------------------------------------------------------------------

    def build(self) -> Any:
        """
        Construct and return a configured ``mcp.server.Server``.

        Registers ``list_tools`` and ``call_tool`` handlers that delegate
        to ClawScope's ToolRegistry (and SkillRegistry if provided).
        """
        try:
            from mcp.server import Server
            import mcp.types as types
        except ImportError:
            raise ImportError(
                "mcp package not installed. "
                "Run: pip install clawscope[mcp]"
            )

        server = Server(self.name)
        registry_ref = self.tool_registry
        skill_ref = self.skill_registry

        @server.list_tools()
        async def handle_list_tools() -> list[types.Tool]:
            return _build_tool_list(registry_ref, skill_ref)

        @server.call_tool()
        async def handle_call_tool(
            name: str,
            arguments: dict[str, Any] | None,
        ) -> list[types.TextContent]:
            result = await _dispatch_tool(
                name, arguments or {}, registry_ref, skill_ref
            )
            return [types.TextContent(type="text", text=result)]

        logger.debug(
            f"MCPServer '{self.name}' built with "
            f"{len(registry_ref.list_tools())} tools"
        )
        return server

    # ------------------------------------------------------------------
    # Runtime modes
    # ------------------------------------------------------------------

    async def run_stdio(self) -> None:
        """
        Run the MCP server over stdin/stdout (the standard deployment mode).

        Blocks until the client disconnects or the process is terminated.
        """
        try:
            from mcp.server.stdio import stdio_server
        except ImportError:
            raise ImportError("mcp package not installed. Run: pip install clawscope[mcp]")

        mcp_server = self.build()
        init_options = mcp_server.create_initialization_options()

        logger.info(f"MCPServer '{self.name}' starting on stdio")
        async with stdio_server() as (read_stream, write_stream):
            await mcp_server.run(read_stream, write_stream, init_options)

    async def run_http(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> None:
        """
        Run the MCP server over HTTP/SSE.

        Requires ``mcp[cli]`` or ``uvicorn`` to be installed.

        Args:
            host: Bind address.
            port: Port to listen on.
        """
        try:
            from mcp.server.sse import SseServerTransport  # type: ignore[import]
            from starlette.applications import Starlette
            from starlette.routing import Route, Mount
            import uvicorn
        except ImportError:
            raise ImportError(
                "HTTP transport requires additional packages. "
                "Run: pip install mcp[cli] uvicorn"
            )

        mcp_server = self.build()
        sse_transport = SseServerTransport("/messages/")

        async def handle_sse(request):
            async with sse_transport.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await mcp_server.run(
                    streams[0],
                    streams[1],
                    mcp_server.create_initialization_options(),
                )

        starlette_app = Starlette(
            routes=[
                Route("/sse", endpoint=handle_sse),
                Mount("/messages/", app=sse_transport.handle_post_message),
            ]
        )

        logger.info(f"MCPServer '{self.name}' starting on http://{host}:{port}")
        config = uvicorn.Config(starlette_app, host=host, port=port, log_level="warning")
        server = uvicorn.Server(config)
        await server.serve()

    # ------------------------------------------------------------------
    # Convenience: stats
    # ------------------------------------------------------------------

    def get_exposed_tools(self) -> list[str]:
        """List all tool names that will be advertised to MCP clients."""
        names = list(self.tool_registry.list_tools())
        if self.skill_registry:
            names += [f"skill_{s.name}" for s in self.skill_registry.list_skills()]
        return names


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_tool_list(
    tool_registry: "ToolRegistry",
    skill_registry: "SkillRegistry | None",
) -> list[Any]:
    """Build the mcp.types.Tool list from registry contents."""
    from mcp import types

    result: list[types.Tool] = []

    # Regular tools
    for claws_tool in tool_registry._tools.values():
        if not claws_tool.enabled:
            continue
        schema = claws_tool.to_openai_schema()
        func_def = schema.get("function", {})
        result.append(
            types.Tool(
                name=claws_tool.name,
                description=claws_tool.description,
                inputSchema=func_def.get("parameters", {"type": "object", "properties": {}}),
            )
        )

    # Skills (exposed as "skill_{name}" tools)
    if skill_registry:
        for skill in skill_registry.list_skills():
            result.append(
                types.Tool(
                    name=f"skill_{skill.name}",
                    description=getattr(skill.config, "description", skill.name),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "Input message for the skill",
                            }
                        },
                        "required": ["message"],
                    },
                )
            )

    return result


async def _dispatch_tool(
    name: str,
    arguments: dict[str, Any],
    tool_registry: "ToolRegistry",
    skill_registry: "SkillRegistry | None",
) -> str:
    """Route an MCP call_tool request to the appropriate ClawScope handler."""
    # Skill call?
    if name.startswith("skill_") and skill_registry:
        skill_name = name[len("skill_"):]
        from clawscope.skills.base import SkillContext
        from clawscope.message import Msg

        ctx = SkillContext(
            message=Msg(
                name="user",
                content=arguments.get("message", ""),
                role="user",
            )
        )
        response = await skill_registry.execute(skill_name, ctx)
        if response:
            return response.get_text_content() if hasattr(response, "get_text_content") else str(response)
        return "(no response)"

    # Regular tool call
    return await tool_registry.execute(name, arguments)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def run_stdio_cli() -> None:
    """
    Synchronous entry-point for use as a CLI command (``clawscope-mcp``).

    Reads configuration from the standard ClawScope config file, initialises
    the ToolRegistry with built-in tools, and starts the stdio MCP server.
    """
    import asyncio
    from pathlib import Path

    from clawscope.config import Config
    from clawscope.tool import ToolRegistry

    config_path = Path.home() / ".clawscope" / "config.yaml"
    config = Config.from_file(config_path) if config_path.exists() else Config()

    async def _main() -> None:
        tool_registry = ToolRegistry(config.tools)
        await tool_registry.load_builtin_tools()
        server = MCPServer(tool_registry, name="clawscope")
        await server.run_stdio()

    asyncio.run(_main())


__all__ = ["MCPServer", "run_stdio_cli"]
