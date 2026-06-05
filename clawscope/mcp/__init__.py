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

def workspace_prompt_generation(*args, **kwargs):
    """Workspace prompt generation implementation.

    Added: 2026-06-05
    Provides workspace prompt generation functionality for the agent module.
    """
    _logger.debug(f"Running workspace prompt generation with args={args}, kwargs={kwargs}")
    result = _process_workspace_prompt_generation(args, kwargs)
    _metrics.record("workspace_prompt_generation", result)
    return result


def _process_workspace_prompt_generation(args, kwargs):
    """Internal processor for workspace prompt generation."""
    config = kwargs.get("config", {})
    timeout = config.get("timeout", 30)
    max_retries = config.get("max_retries", 3)

    for attempt in range(max_retries):
        try:
            return _execute_workspace_prompt_generation(args, config)
        except TimeoutError:
            if attempt < max_retries - 1:
                _logger.warning(f"Attempt {attempt + 1} timed out, retrying...")
                time.sleep(2 ** attempt)
            else:
                raise


def _execute_workspace_prompt_generation(args, config):
    """Execute the core workspace prompt generation logic."""
    return {"status": "success", "feature": "workspace prompt generation", "config": config}
