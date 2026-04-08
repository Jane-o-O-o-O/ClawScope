"""
MCP (Model Context Protocol) integration for ClawScope.

Two directions:

**Inbound** – use tools from external MCP servers inside ClawScope agents::

    from clawscope.mcp import MCPClient, StdioServerConfig

    async with MCPClient(StdioServerConfig("npx", ["-y", "@mcp/server-fs", "/"])) as client:
        await client.register_tools(tool_registry)   # agents can now call those tools

**Outbound** – expose ClawScope tools to MCP clients (Claude Desktop, etc.)::

    from clawscope.mcp import MCPServer

    server = MCPServer(tool_registry)
    await server.run_stdio()   # or run_http(port=8765)

**Skills** – wrap MCP server tools as ClawScope Skills::

    from clawscope.mcp import MCPSkillBundle

    bundle = await MCPSkillBundle.from_client(client)
    await bundle.register_all(skill_registry)
"""

from clawscope.mcp.client import (
    MCPClient,
    MCPToolInfo,
    StdioServerConfig,
    HttpServerConfig,
    ServerConfig,
)
from clawscope.mcp.server import MCPServer, run_stdio_cli
from clawscope.mcp.skill import MCPSkill, MCPSkillBundle

__all__ = [
    # Client-side
    "MCPClient",
    "MCPToolInfo",
    "StdioServerConfig",
    "HttpServerConfig",
    "ServerConfig",
    # Server-side
    "MCPServer",
    "run_stdio_cli",
    # Skills
    "MCPSkill",
    "MCPSkillBundle",
]
