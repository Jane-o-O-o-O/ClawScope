"""Tests for OrchestratorAgent and SessionRouter orchestration mode."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clawscope.agent.orchestrator import OrchestratorAgent, create_orchestrator
from clawscope.agent.base import AgentBase
from clawscope.message import Msg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_msg(text: str, name: str = "user") -> Msg:
    return Msg(name=name, content=text, role="user")


def _make_sub_agent(name: str, reply_text: str) -> AgentBase:
    """Create a mock sub-agent that always returns the given text."""
    response = Msg(name=name, content=reply_text, role="assistant")
    # Plain AsyncMock: await agent(msg) → response; agent.call_args tracks calls.
    agent = AsyncMock()
    agent.name = name
    agent.sys_prompt = f"I am the {name} agent."
    agent.return_value = response
    return agent


def _make_orchestrator(
    sub_agents=None,
    model=None,
    sys_prompt="",
) -> OrchestratorAgent:
    from clawscope.memory import InMemoryMemory
    return OrchestratorAgent(
        name="Orch",
        sys_prompt=sys_prompt,
        model=model,
        memory=InMemoryMemory(),
        sub_agents=sub_agents,
    )


# ---------------------------------------------------------------------------
# OrchestratorAgent – construction
# ---------------------------------------------------------------------------


class TestOrchestratorConstruction:
    def test_default_sys_prompt_applied(self):
        orch = _make_orchestrator()
        assert "orchestrator" in orch.sys_prompt.lower()

    def test_custom_sys_prompt_preserved(self):
        orch = _make_orchestrator(sys_prompt="Custom prompt here.")
        assert orch.sys_prompt == "Custom prompt here."

    def test_no_sub_agents_by_default(self):
        orch = _make_orchestrator()
        assert orch.sub_agents == {}

    def test_sub_agents_list_registered(self):
        a = _make_sub_agent("alpha", "A")
        b = _make_sub_agent("beta", "B")
        orch = _make_orchestrator(sub_agents=[a, b])
        assert "alpha" in orch.sub_agents
        assert "beta" in orch.sub_agents

    def test_sub_agents_dict_registered(self):
        a = _make_sub_agent("x", "X")
        orch = _make_orchestrator(sub_agents={"my_alias": a})
        assert "my_alias" in orch.sub_agents
        assert "x" not in orch.sub_agents

    def test_tools_created_for_sub_agents(self):
        a = _make_sub_agent("search", "results")
        orch = _make_orchestrator(sub_agents=[a])
        assert "ask_search" in orch.tools._tools

    def test_inherits_from_react_agent(self):
        from clawscope.agent.react import ReActAgent
        orch = _make_orchestrator()
        assert isinstance(orch, ReActAgent)


# ---------------------------------------------------------------------------
# OrchestratorAgent – register / unregister
# ---------------------------------------------------------------------------


class TestSubAgentRegistration:
    def test_register_adds_tool(self):
        orch = _make_orchestrator()
        agent = _make_sub_agent("writer", "text")
        orch.register_sub_agent(agent)
        assert "ask_writer" in orch.tools._tools
        assert "writer" in orch.sub_agents

    def test_register_with_alias(self):
        orch = _make_orchestrator()
        agent = _make_sub_agent("WriterInternal", "text")
        orch.register_sub_agent(agent, alias="writer")
        assert "ask_writer" in orch.tools._tools
        assert "writer" in orch.sub_agents
        assert "WriterInternal" not in orch.sub_agents

    def test_unregister_removes_tool_and_agent(self):
        agent = _make_sub_agent("calc", "42")
        orch = _make_orchestrator(sub_agents=[agent])
        assert orch.unregister_sub_agent("calc") is True
        assert "calc" not in orch.sub_agents
        assert "ask_calc" not in orch.tools._tools

    def test_unregister_missing_returns_false(self):
        orch = _make_orchestrator()
        assert orch.unregister_sub_agent("nonexistent") is False

    def test_sub_agents_property_is_copy(self):
        agent = _make_sub_agent("a", "a")
        orch = _make_orchestrator(sub_agents=[agent])
        snapshot = orch.sub_agents
        orch.register_sub_agent(_make_sub_agent("b", "b"))
        assert "b" not in snapshot  # snapshot is not mutated


# ---------------------------------------------------------------------------
# OrchestratorAgent – tool call mechanics
# ---------------------------------------------------------------------------


class TestAgentToolExecution:
    async def test_tool_calls_sub_agent(self):
        sub = _make_sub_agent("greeter", "Hello!")
        orch = _make_orchestrator(sub_agents=[sub])
        tool = orch.tools._tools["ask_greeter"]
        result = await tool.func(message="Say hello")
        assert "Hello!" in result
        sub.assert_called_once()

    async def test_tool_passes_message_text(self):
        sub = _make_sub_agent("echo", "echoed")
        orch = _make_orchestrator(sub_agents=[sub])
        tool = orch.tools._tools["ask_echo"]
        await tool.func(message="specific task")
        passed_msg = sub.call_args[0][0]
        assert "specific task" in passed_msg.get_text_content()

    async def test_tool_handles_sub_agent_error_gracefully(self):
        sub = AsyncMock()
        sub.name = "broken"
        sub.sys_prompt = ""
        sub.side_effect = RuntimeError("service unavailable")
        orch = _make_orchestrator(sub_agents=[sub])
        tool = orch.tools._tools["ask_broken"]
        result = await tool.func(message="do something")
        assert "error" in result.lower()
        assert "service unavailable" in result

    async def test_tool_handles_none_response(self):
        sub = AsyncMock()
        sub.name = "silent"
        sub.sys_prompt = ""
        sub.return_value = None
        orch = _make_orchestrator(sub_agents=[sub])
        tool = orch.tools._tools["ask_silent"]
        result = await tool.func(message="hello")
        assert "no response" in result.lower()


# ---------------------------------------------------------------------------
# OrchestratorAgent – tool schema
# ---------------------------------------------------------------------------


class TestAgentToolSchema:
    def test_tool_has_message_parameter(self):
        agent = _make_sub_agent("search", "found")
        orch = _make_orchestrator(sub_agents=[agent])
        tool = orch.tools._tools["ask_search"]
        assert len(tool.parameters) == 1
        assert tool.parameters[0].name == "message"
        assert tool.parameters[0].required is True

    def test_tool_description_contains_agent_name(self):
        agent = _make_sub_agent("researcher", "research")
        orch = _make_orchestrator(sub_agents=[agent])
        tool = orch.tools._tools["ask_researcher"]
        assert "researcher" in tool.description

    def test_tool_description_truncates_long_sys_prompt(self):
        agent = AsyncMock()
        agent.name = "verbose"
        agent.sys_prompt = "x" * 500  # very long
        agent.return_value = Msg(name="v", content="ok", role="assistant")
        orch = _make_orchestrator(sub_agents=[agent])
        tool = orch.tools._tools["ask_verbose"]
        assert len(tool.description) < 700  # should be truncated


# ---------------------------------------------------------------------------
# OrchestratorAgent – catalogue / sys-prompt helpers
# ---------------------------------------------------------------------------


class TestCatalogue:
    def test_catalogue_empty_when_no_sub_agents(self):
        orch = _make_orchestrator()
        catalogue = orch._build_agent_catalogue()
        assert "no sub-agents" in catalogue

    def test_catalogue_lists_all_agents(self):
        a = _make_sub_agent("alpha", "A")
        b = _make_sub_agent("beta", "B")
        orch = _make_orchestrator(sub_agents=[a, b])
        catalogue = orch._build_agent_catalogue()
        assert "alpha" in catalogue
        assert "beta" in catalogue

    def test_effective_sys_prompt_includes_catalogue(self):
        a = _make_sub_agent("worker", "result")
        orch = _make_orchestrator(sub_agents=[a])
        effective = orch.get_effective_sys_prompt()
        assert "worker" in effective
        assert "Registered sub-agents" in effective


# ---------------------------------------------------------------------------
# create_orchestrator factory
# ---------------------------------------------------------------------------


class TestCreateOrchestratorFactory:
    def test_creates_orchestrator_agent(self):
        orch = create_orchestrator(name="MainBot")
        assert isinstance(orch, OrchestratorAgent)
        assert orch.name == "MainBot"

    def test_factory_with_sub_agents(self):
        a = _make_sub_agent("a", "a_reply")
        orch = create_orchestrator(sub_agents=[a])
        assert "a" in orch.sub_agents

    def test_factory_has_memory(self):
        orch = create_orchestrator()
        assert orch.memory is not None


# ---------------------------------------------------------------------------
# SessionRouter – orchestrator mode
# ---------------------------------------------------------------------------


class TestSessionRouterOrchestratorMode:
    def _make_router(self):
        from clawscope.orchestration.router import SessionRouter
        from clawscope.config import AgentConfig

        bus = MagicMock()
        bus.consume_inbound = AsyncMock()
        bus.publish_outbound = AsyncMock()

        sessions = MagicMock()
        session = MagicMock()
        session.session_key = "test:session"
        sessions.get_or_create = AsyncMock(return_value=session)

        kernel = MagicMock()
        kernel.model_registry = MagicMock()
        kernel.model_registry.get_model = MagicMock(return_value=None)
        kernel.create_agent = MagicMock(
            return_value=MagicMock(spec=AgentBase)
        )

        config = AgentConfig(name="TestAgent")

        router = SessionRouter(bus=bus, sessions=sessions, kernel=kernel, config=config)
        return router

    async def test_no_sub_agents_uses_kernel(self):
        router = self._make_router()
        agent = await router._create_agent("k", "ch", "id")
        router.kernel.create_agent.assert_called_once()
        assert not isinstance(agent, OrchestratorAgent)

    async def test_with_sub_agents_creates_orchestrator(self):
        router = self._make_router()
        sub = _make_sub_agent("worker", "done")
        router.register_sub_agent("worker", sub)
        agent = await router._create_agent("k", "ch", "id")
        assert isinstance(agent, OrchestratorAgent)

    async def test_orchestrator_has_registered_sub_agents(self):
        router = self._make_router()
        sub_a = _make_sub_agent("alpha", "a")
        sub_b = _make_sub_agent("beta", "b")
        router.register_sub_agent("alpha", sub_a)
        router.register_sub_agent("beta", sub_b)
        agent = await router._create_agent("k", "ch", "id")
        assert isinstance(agent, OrchestratorAgent)
        assert "alpha" in agent.sub_agents
        assert "beta" in agent.sub_agents

    def test_register_sub_agent_logged(self):
        router = self._make_router()
        sub = _make_sub_agent("x", "x")
        router.register_sub_agent("x", sub)
        assert "x" in router._sub_agents

    def test_unregister_sub_agent(self):
        router = self._make_router()
        sub = _make_sub_agent("x", "x")
        router.register_sub_agent("x", sub)
        assert router.unregister_sub_agent("x") is True
        assert "x" not in router._sub_agents

    def test_unregister_missing_returns_false(self):
        router = self._make_router()
        assert router.unregister_sub_agent("ghost") is False


# ---------------------------------------------------------------------------
# ClawScope.register_sub_agent integration
# ---------------------------------------------------------------------------


class TestClawScopeRegisterSubAgent:
    def _make_app(self):
        from clawscope.app import ClawScope
        from clawscope.config import Config
        app = ClawScope(Config())
        return app

    def test_register_before_start_stored_as_pending(self):
        app = self._make_app()
        sub = _make_sub_agent("helper", "help")
        app.register_sub_agent(sub)
        assert "helper" in getattr(app, "_pending_sub_agents", {})

    def test_register_after_start_goes_to_router(self):
        app = self._make_app()
        from clawscope.orchestration.router import SessionRouter
        from clawscope.config import AgentConfig

        mock_router = MagicMock(spec=SessionRouter)
        app._router = mock_router

        sub = _make_sub_agent("searcher", "found")
        app.register_sub_agent(sub, name="my_searcher")
        mock_router.register_sub_agent.assert_called_once_with("my_searcher", sub)

    def test_register_uses_agent_name_as_default_key(self):
        app = self._make_app()
        from clawscope.orchestration.router import SessionRouter
        mock_router = MagicMock(spec=SessionRouter)
        app._router = mock_router

        sub = _make_sub_agent("autoname", "ok")
        app.register_sub_agent(sub)
        mock_router.register_sub_agent.assert_called_once_with("autoname", sub)
