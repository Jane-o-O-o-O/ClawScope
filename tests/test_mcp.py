"""Tests for clawscope.mcp — MCPClient, MCPServer, MCPSkill, MCPSkillBundle.

All MCP SDK calls are mocked so that the mcp package does not need to be
installed in the test environment.
"""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Minimal mcp SDK stubs (injected before any import of clawscope.mcp)
# ---------------------------------------------------------------------------


def _make_mcp_stubs():
    """Inject minimal mcp SDK modules into sys.modules."""

    # --- mcp.types ---
    mcp_types = types.ModuleType("mcp.types")

    class TextContent:
        def __init__(self, *, type: str = "text", text: str = ""):
            self.type = type
            self.text = text

    class Tool:
        def __init__(self, *, name: str, description: str = "", inputSchema=None):
            self.name = name
            self.description = description
            self.inputSchema = inputSchema or {}

    mcp_types.TextContent = TextContent
    mcp_types.Tool = Tool

    # --- mcp (top-level) ---
    mcp_top = types.ModuleType("mcp")
    mcp_top.types = mcp_types

    class _FakeClientSession:
        pass

    class _FakeStdioServerParameters:
        def __init__(self, command, args, env):
            self.command = command
            self.args = args
            self.env = env

    mcp_top.ClientSession = _FakeClientSession
    mcp_top.StdioServerParameters = _FakeStdioServerParameters

    # --- mcp.client.stdio ---
    mcp_client = types.ModuleType("mcp.client")
    mcp_client_stdio = types.ModuleType("mcp.client.stdio")
    mcp_client_stdio.stdio_client = MagicMock()
    mcp_client.stdio = mcp_client_stdio

    # --- mcp.server ---
    mcp_server_mod = types.ModuleType("mcp.server")

    class _FakeMCPServerServer:
        def __init__(self, name):
            self.name = name
            self._list_tools_handler = None
            self._call_tool_handler = None

        def list_tools(self):
            def decorator(fn):
                self._list_tools_handler = fn
                return fn
            return decorator

        def call_tool(self):
            def decorator(fn):
                self._call_tool_handler = fn
                return fn
            return decorator

        def create_initialization_options(self):
            return {}

        async def run(self, read, write, options):
            pass

    mcp_server_mod.Server = _FakeMCPServerServer

    # --- mcp.server.stdio ---
    mcp_server_stdio = types.ModuleType("mcp.server.stdio")

    class _AsyncCMStdio:
        async def __aenter__(self):
            return (AsyncMock(), AsyncMock())

        async def __aexit__(self, *a):
            pass

    mcp_server_stdio.stdio_server = MagicMock(return_value=_AsyncCMStdio())
    mcp_server_mod.stdio = mcp_server_stdio

    # Register all modules
    sys.modules.setdefault("mcp", mcp_top)
    sys.modules.setdefault("mcp.types", mcp_types)
    sys.modules.setdefault("mcp.client", mcp_client)
    sys.modules.setdefault("mcp.client.stdio", mcp_client_stdio)
    sys.modules.setdefault("mcp.server", mcp_server_mod)
    sys.modules.setdefault("mcp.server.stdio", mcp_server_stdio)

    return {
        "mcp": mcp_top,
        "mcp.types": mcp_types,
        "mcp.server": mcp_server_mod,
        "mcp.client.stdio": mcp_client_stdio,
    }


_STUBS = _make_mcp_stubs()

# Now import ClawScope MCP modules (stubs already registered)
from clawscope.mcp.client import (  # noqa: E402
    MCPClient,
    MCPToolInfo,
    StdioServerConfig,
    HttpServerConfig,
    _extract_text,
    _register_one,
)
from clawscope.mcp.server import MCPServer, _build_tool_list, _dispatch_tool  # noqa: E402
from clawscope.mcp.skill import MCPSkill, MCPSkillBundle  # noqa: E402
from clawscope.mcp import (  # noqa: E402
    MCPClient as MCPClientPkg,
    MCPServer as MCPServerPkg,
    MCPSkill as MCPSkillPkg,
    MCPSkillBundle as MCPSkillBundlePkg,
    StdioServerConfig as StdioServerConfigPkg,
    HttpServerConfig as HttpServerConfigPkg,
    MCPToolInfo as MCPToolInfoPkg,
)
from clawscope.skills.base import SkillCategory, SkillConfig, SkillContext
from clawscope.message import Msg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_registry():
    """Create a minimal ToolRegistry with zero tools."""
    from clawscope.tool.registry import ToolRegistry
    from clawscope.config import ToolsConfig
    reg = ToolRegistry(ToolsConfig())
    return reg


def _make_skill_registry():
    from clawscope.skills import SkillRegistry
    return SkillRegistry()


def _fake_mcp_tool(name="echo", description="Echo tool", schema=None):
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.inputSchema = schema or {"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]}
    return tool


def _fake_list_tools_result(tools):
    result = MagicMock()
    result.tools = tools
    return result


def _fake_call_result(text="ok", is_error=False):
    block = MagicMock()
    block.text = text
    result = MagicMock()
    result.isError = is_error
    result.content = [block]
    return result


def _make_connected_client(name="test", tools=None):
    """Create an MCPClient that appears connected with a mocked session."""
    cfg = StdioServerConfig(command="echo", args=[], name=name)
    client = MCPClient(cfg)
    client._connected = True
    client._session = AsyncMock()
    if tools is None:
        tools = [_fake_mcp_tool()]
    client._session.list_tools = AsyncMock(return_value=_fake_list_tools_result(tools))
    client._session.call_tool = AsyncMock(return_value=_fake_call_result("result text"))
    return client


# ===========================================================================
# MCPToolInfo
# ===========================================================================


class TestMCPToolInfo:
    def test_basic_fields(self):
        info = MCPToolInfo(name="read_file", description="Read a file", input_schema={"x": 1})
        assert info.name == "read_file"
        assert info.description == "Read a file"
        assert info.input_schema == {"x": 1}

    def test_defaults(self):
        info = MCPToolInfo(name="t", description="")
        assert info.description == ""
        assert info.input_schema == {}

    def test_from_mcp(self):
        fake = _fake_mcp_tool("my_tool", "desc", {"type": "object"})
        info = MCPToolInfo.from_mcp(fake)
        assert info.name == "my_tool"
        assert info.description == "desc"
        assert info.input_schema == {"type": "object"}

    def test_from_mcp_no_schema(self):
        fake = MagicMock()
        fake.name = "t"
        fake.description = None
        fake.inputSchema = None
        info = MCPToolInfo.from_mcp(fake)
        assert info.description == ""
        assert info.input_schema == {}


# ===========================================================================
# StdioServerConfig / HttpServerConfig
# ===========================================================================


class TestServerConfigs:
    def test_stdio_defaults(self):
        cfg = StdioServerConfig(command="npx")
        assert cfg.args == []
        assert cfg.env is None
        assert cfg.name == "mcp"

    def test_stdio_full(self):
        cfg = StdioServerConfig(command="uv", args=["run", "server.py"], env={"X": "1"}, name="fs")
        assert cfg.command == "uv"
        assert cfg.env == {"X": "1"}
        assert cfg.name == "fs"

    def test_http_defaults(self):
        cfg = HttpServerConfig(url="http://localhost:8765")
        assert cfg.headers == {}
        assert cfg.name == "mcp-http"

    def test_http_with_headers(self):
        cfg = HttpServerConfig(url="http://x", headers={"Authorization": "Bearer tok"}, name="remote")
        assert cfg.headers["Authorization"] == "Bearer tok"
        assert cfg.name == "remote"


# ===========================================================================
# MCPClient — connection guard
# ===========================================================================


class TestMCPClientGuard:
    def test_not_connected_raises(self):
        client = MCPClient(StdioServerConfig(command="echo"))
        with pytest.raises(RuntimeError, match="not connected"):
            client._require_connected()

    def test_is_connected_false_initially(self):
        client = MCPClient(StdioServerConfig(command="echo"))
        assert client.is_connected is False

    def test_is_connected_true_after_flag(self):
        client = MCPClient(StdioServerConfig(command="echo"))
        client._connected = True
        client._session = MagicMock()
        assert client.is_connected is True


# ===========================================================================
# MCPClient — list_tools
# ===========================================================================


class TestMCPClientListTools:
    async def test_list_tools_returns_tool_infos(self):
        client = _make_connected_client(
            tools=[
                _fake_mcp_tool("tool_a", "A"),
                _fake_mcp_tool("tool_b", "B"),
            ]
        )
        tools = await client.list_tools()
        assert len(tools) == 2
        assert tools[0].name == "tool_a"
        assert tools[1].name == "tool_b"

    async def test_list_tools_not_connected_raises(self):
        client = MCPClient(StdioServerConfig(command="x"))
        with pytest.raises(RuntimeError):
            await client.list_tools()

    async def test_list_tools_empty_server(self):
        client = _make_connected_client(tools=[])
        tools = await client.list_tools()
        assert tools == []


# ===========================================================================
# MCPClient — call_tool
# ===========================================================================


class TestMCPClientCallTool:
    async def test_call_tool_returns_text(self):
        client = _make_connected_client()
        client._session.call_tool = AsyncMock(return_value=_fake_call_result("hello world"))
        result = await client.call_tool("echo", {"input": "hello"})
        assert result == "hello world"

    async def test_call_tool_passes_arguments(self):
        client = _make_connected_client()
        client._session.call_tool = AsyncMock(return_value=_fake_call_result("ok"))
        await client.call_tool("my_tool", {"a": 1, "b": 2})
        client._session.call_tool.assert_called_once_with("my_tool", {"a": 1, "b": 2})

    async def test_call_tool_empty_args_defaults_to_empty_dict(self):
        client = _make_connected_client()
        client._session.call_tool = AsyncMock(return_value=_fake_call_result("ok"))
        await client.call_tool("t")
        client._session.call_tool.assert_called_once_with("t", {})

    async def test_call_tool_error_raises(self):
        client = _make_connected_client()
        client._session.call_tool = AsyncMock(return_value=_fake_call_result("boom", is_error=True))
        with pytest.raises(RuntimeError, match="error"):
            await client.call_tool("bad_tool", {})

    async def test_call_tool_not_connected_raises(self):
        client = MCPClient(StdioServerConfig(command="x"))
        with pytest.raises(RuntimeError):
            await client.call_tool("t")


# ===========================================================================
# MCPClient — register_tools
# ===========================================================================


class TestMCPClientRegisterTools:
    async def test_register_tools_adds_to_registry(self):
        client = _make_connected_client(
            name="fs",
            tools=[_fake_mcp_tool("read_file", "Read a file")],
        )
        registry = _make_tool_registry()
        count = await client.register_tools(registry)
        assert count == 1
        assert "fs_read_file" in registry._tools

    async def test_register_tools_custom_prefix(self):
        client = _make_connected_client(tools=[_fake_mcp_tool("get")])
        registry = _make_tool_registry()
        await client.register_tools(registry, prefix="api_")
        assert "api_get" in registry._tools

    async def test_register_tools_no_prefix_when_default_name(self):
        cfg = StdioServerConfig(command="x", name="mcp")  # default name
        client = MCPClient(cfg)
        client._connected = True
        client._session = AsyncMock()
        client._session.list_tools = AsyncMock(
            return_value=_fake_list_tools_result([_fake_mcp_tool("ping")])
        )
        registry = _make_tool_registry()
        await client.register_tools(registry)
        assert "ping" in registry._tools

    async def test_registered_tool_is_callable(self):
        client = _make_connected_client(tools=[_fake_mcp_tool("echo")])
        registry = _make_tool_registry()
        await client.register_tools(registry, prefix="")
        tool = registry._tools["echo"]
        assert tool.func is not None
        assert callable(tool.func)


# ===========================================================================
# _extract_text helper
# ===========================================================================


class TestExtractText:
    def test_extracts_from_text_attribute(self):
        block = MagicMock()
        block.text = "hello"
        assert _extract_text([block]) == "hello"

    def test_concatenates_multiple_blocks(self):
        b1, b2 = MagicMock(), MagicMock()
        b1.text, b2.text = "foo", "bar"
        assert _extract_text([b1, b2]) == "foo\nbar"

    def test_dict_block(self):
        result = _extract_text([{"text": "from dict"}])
        assert result == "from dict"

    def test_fallback_to_str(self):
        result = _extract_text(["plain string"])
        assert result == "plain string"

    def test_empty_list(self):
        assert _extract_text([]) == ""


# ===========================================================================
# MCPServer
# ===========================================================================


class TestMCPServer:
    def test_build_returns_server_object(self):
        registry = _make_tool_registry()
        server = MCPServer(registry, name="test")
        mcp_srv = server.build()
        assert mcp_srv is not None

    def test_get_exposed_tools_empty_registry(self):
        registry = _make_tool_registry()
        server = MCPServer(registry)
        assert server.get_exposed_tools() == []

    def test_get_exposed_tools_with_tools(self):
        from clawscope.tool.registry import Tool
        registry = _make_tool_registry()
        registry._tools["my_tool"] = Tool(name="my_tool", description="d", enabled=True)
        server = MCPServer(registry)
        assert "my_tool" in server.get_exposed_tools()

    def test_get_exposed_tools_with_skills(self):
        registry = _make_tool_registry()
        skill_registry = _make_skill_registry()

        from clawscope.skills.base import SkillConfig

        class _DummySkill(MCPSkill):
            pass

        cfg = SkillConfig(name="my_skill")
        client = _make_connected_client()
        skill = MCPSkill(client, "echo", cfg)
        skill_registry.register(skill)

        server = MCPServer(registry, skill_registry=skill_registry)
        assert "skill_my_skill" in server.get_exposed_tools()

    def test_build_tool_list_disabled_tool_excluded(self):
        from clawscope.tool.registry import Tool
        registry = _make_tool_registry()
        registry._tools["active"] = Tool(name="active", description="d", enabled=True)
        registry._tools["inactive"] = Tool(name="inactive", description="d", enabled=False)

        result = _build_tool_list(registry, None)
        names = [t.name for t in result]
        assert "active" in names
        assert "inactive" not in names

    async def test_dispatch_tool_calls_tool_registry(self):
        registry = _make_tool_registry()
        registry.execute = AsyncMock(return_value="tool result")
        result = await _dispatch_tool("my_tool", {"x": 1}, registry, None)
        assert result == "tool result"
        registry.execute.assert_called_once_with("my_tool", {"x": 1})

    async def test_dispatch_tool_skill_prefix_routes_to_skill_registry(self):
        from clawscope.skills.base import SkillContext
        registry = _make_tool_registry()
        skill_registry = _make_skill_registry()

        mock_response = Msg(name="skill", content="skill output", role="assistant")
        skill_registry.execute = AsyncMock(return_value=mock_response)

        result = await _dispatch_tool("skill_greeter", {"message": "hi"}, registry, skill_registry)
        assert "skill output" in result
        skill_registry.execute.assert_called_once()

    async def test_run_stdio_calls_mcp_run(self):
        # The mcp.server.stdio.stdio_server stub is already registered in
        # sys.modules; _FakeMCPServerServer.run is a no-op async method.
        registry = _make_tool_registry()
        server = MCPServer(registry)
        # Should complete without raising; stubs handle transport
        await server.run_stdio()


# ===========================================================================
# MCPSkill
# ===========================================================================


class TestMCPSkill:
    def _make_skill(self, tool_name="echo", triggers=None):
        client = _make_connected_client()
        cfg = SkillConfig(name="test_skill", triggers=triggers or ["!echo"])
        return MCPSkill(client, tool_name, cfg), client

    async def test_execute_returns_msg(self):
        skill, client = self._make_skill()
        client._session.call_tool = AsyncMock(return_value=_fake_call_result("pong"))
        ctx = SkillContext(message=Msg(name="user", content="ping", role="user"))
        result = await skill.execute(ctx)
        assert isinstance(result, Msg)
        assert "pong" in result.get_text_content()

    async def test_execute_default_argument_mapper(self):
        skill, client = self._make_skill()
        client._session.call_tool = AsyncMock(return_value=_fake_call_result("ok"))
        ctx = SkillContext(message=Msg(name="user", content="hello world", role="user"))
        await skill.execute(ctx)
        client._session.call_tool.assert_called_once_with("echo", {"input": "hello world"})

    async def test_execute_custom_argument_mapper(self):
        client = _make_connected_client()
        cfg = SkillConfig(name="s")
        skill = MCPSkill(client, "echo", cfg, argument_mapper=lambda ctx: {"q": "custom"})
        client._session.call_tool = AsyncMock(return_value=_fake_call_result("ok"))
        ctx = SkillContext(message=Msg(name="user", content="anything", role="user"))
        await skill.execute(ctx)
        client._session.call_tool.assert_called_once_with("echo", {"q": "custom"})

    async def test_execute_tool_error_returns_error_msg(self):
        skill, client = self._make_skill()
        client._session.call_tool = AsyncMock(return_value=_fake_call_result("bad", is_error=True))
        ctx = SkillContext(message=Msg(name="user", content="x", role="user"))
        result = await skill.execute(ctx)
        assert result is not None
        assert result.metadata.get("is_error") is True

    async def test_execute_not_connected_returns_none(self):
        client = MCPClient(StdioServerConfig(command="x"))
        cfg = SkillConfig(name="s")
        skill = MCPSkill(client, "echo", cfg)
        ctx = SkillContext(message=Msg(name="user", content="x", role="user"))
        result = await skill.execute(ctx)
        assert result is None

    def test_from_client_factory(self):
        client = _make_connected_client(name="myserver")
        skill = MCPSkill.from_client(client, "my_tool", description="Does things")
        assert skill.config.name == "myserver_my_tool"
        assert skill._tool_name == "my_tool"
        assert skill._client is client

    def test_from_client_custom_skill_name(self):
        client = _make_connected_client()
        skill = MCPSkill.from_client(client, "tool", skill_name="my_custom_name")
        assert skill.config.name == "my_custom_name"

    def test_from_client_stores_client_and_tool(self):
        client = _make_connected_client(name="srv")
        skill = MCPSkill.from_client(client, "t")
        assert skill._client is client
        assert skill._tool_name == "t"
        # Tags encode MCP identity
        assert any("srv" in tag for tag in skill.config.tags)
        assert any("t" in tag for tag in skill.config.tags)

    async def test_from_tool_info_factory(self):
        client = _make_connected_client()
        info = MCPToolInfo(name="read", description="Read a file")
        skill = await MCPSkill.from_tool_info(client, info)
        assert skill._tool_name == "read"
        assert "Read a file" in skill.config.description

    def test_matches_trigger(self):
        skill, _ = self._make_skill(triggers=["!echo"])
        msg = Msg(name="user", content="!echo hello", role="user")
        assert skill.matches(msg)

    def test_no_match_wrong_trigger(self):
        skill, _ = self._make_skill(triggers=["!echo"])
        msg = Msg(name="user", content="hello world", role="user")
        assert not skill.matches(msg)


# ===========================================================================
# MCPSkillBundle
# ===========================================================================


class TestMCPSkillBundle:
    async def test_from_client_creates_one_skill_per_tool(self):
        client = _make_connected_client(
            name="srv",
            tools=[_fake_mcp_tool("tool_a", "A"), _fake_mcp_tool("tool_b", "B")],
        )
        bundle = await MCPSkillBundle.from_client(client)
        assert len(bundle) == 2
        names = [s.config.name for s in bundle]
        assert "srv_tool_a" in names
        assert "srv_tool_b" in names

    async def test_from_client_custom_prefix(self):
        client = _make_connected_client(tools=[_fake_mcp_tool("ping")])
        bundle = await MCPSkillBundle.from_client(client, prefix="x_")
        assert bundle.skills[0].config.name == "x_ping"

    async def test_from_client_empty_server(self):
        client = _make_connected_client(tools=[])
        bundle = await MCPSkillBundle.from_client(client)
        assert len(bundle) == 0

    async def test_register_all_adds_to_registry(self):
        client = _make_connected_client(tools=[_fake_mcp_tool("t1"), _fake_mcp_tool("t2")])
        bundle = await MCPSkillBundle.from_client(client, prefix="")
        skill_registry = _make_skill_registry()
        count = await bundle.register_all(skill_registry)
        assert count == 2
        assert skill_registry.get("t1") is not None
        assert skill_registry.get("t2") is not None

    async def test_iter_yields_skills(self):
        client = _make_connected_client(tools=[_fake_mcp_tool("a"), _fake_mcp_tool("b")])
        bundle = await MCPSkillBundle.from_client(client, prefix="")
        skills = list(bundle)
        assert len(skills) == 2

    async def test_bundle_client_attribute(self):
        client = _make_connected_client()
        bundle = await MCPSkillBundle.from_client(client)
        assert bundle.client is client


# ===========================================================================
# Package-level imports (__init__.py)
# ===========================================================================


class TestPackageExports:
    def test_all_exports_available(self):
        assert MCPClientPkg is MCPClient
        assert MCPServerPkg is MCPServer
        assert MCPSkillPkg is MCPSkill
        assert MCPSkillBundlePkg is MCPSkillBundle
        assert StdioServerConfigPkg is StdioServerConfig
        assert HttpServerConfigPkg is HttpServerConfig
        assert MCPToolInfoPkg is MCPToolInfo
