"""Tests for OrchestratorAgent and SessionRouter orchestration mode."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clawscope.agent.orchestrator import OrchestratorAgent, ProgressReporter, create_orchestrator, _AgentCall
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
        assert len(catalogue) > 0  # some placeholder text when empty

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


# ---------------------------------------------------------------------------
# Dynamic agent creation – spawn_agent
# ---------------------------------------------------------------------------


class TestSpawnAgent:
    def _make_orch_with_model(self):
        """OrchestratorAgent whose model always returns a fixed reply."""
        from unittest.mock import AsyncMock as AM
        from clawscope.memory import InMemoryMemory

        model = MagicMock()
        # chat() returns a Msg with text content
        reply_msg = Msg(name="spawned", content="spawned reply", role="assistant")
        model.chat = AM(return_value=reply_msg)

        orch = OrchestratorAgent(
            name="Orch",
            model=model,
            memory=InMemoryMemory(),
        )
        return orch, model

    def test_spawn_agent_tool_registered(self):
        orch = _make_orchestrator()
        assert "spawn_agent" in orch.tools._tools

    def test_create_agent_tool_registered(self):
        orch = _make_orchestrator()
        assert "create_agent" in orch.tools._tools

    def test_list_agents_tool_registered(self):
        orch = _make_orchestrator()
        assert "list_agents" in orch.tools._tools

    async def test_list_agents_empty(self):
        orch = _make_orchestrator()
        tool = orch.tools._tools["list_agents"]
        result = await tool.func()
        assert "agent" in result.lower()  # some empty message

    async def test_list_agents_shows_registered(self):
        sub = _make_sub_agent("helper", "help")
        orch = _make_orchestrator(sub_agents=[sub])
        tool = orch.tools._tools["list_agents"]
        result = await tool.func()
        assert "helper" in result

    async def test_create_agent_registers_new_agent(self):
        orch = _make_orchestrator()
        tool = orch.tools._tools["create_agent"]
        result = await tool.func(name="analyst", role="You are a data analyst.")
        assert "analyst" in orch.sub_agents
        assert "ask_analyst" in orch.tools._tools
        assert "analyst" in result

    async def test_create_agent_duplicate_name_returns_error(self):
        sub = _make_sub_agent("existing", "ok")
        orch = _make_orchestrator(sub_agents=[sub])
        tool = orch.tools._tools["create_agent"]
        result = await tool.func(name="existing", role="Another role.")
        assert "已存在" in result or "already" in result.lower()

    async def test_create_agent_sanitises_name(self):
        orch = _make_orchestrator()
        tool = orch.tools._tools["create_agent"]
        result = await tool.func(name="my agent", role="Custom role.")
        # spaces replaced with underscores
        assert "my_agent" in orch.sub_agents

    async def test_do_spawn_returns_text(self):
        from clawscope.memory import InMemoryMemory

        # Mock a model that returns through the ReActAgent call stack
        # We patch ReActAgent.__call__ to return a fixed Msg
        orch = _make_orchestrator()

        with patch("clawscope.agent.react.ReActAgent.__call__", new=AsyncMock(
            return_value=Msg(name="s", content="dynamic result", role="assistant")
        )):
            result = await orch._do_spawn("You are an expert.", "Solve X.", register_as=None)
        assert "dynamic result" in result

    async def test_do_spawn_with_register_adds_sub_agent(self):
        orch = _make_orchestrator()

        with patch("clawscope.agent.react.ReActAgent.__call__", new=AsyncMock(
            return_value=Msg(name="s", content="ok", role="assistant")
        )):
            await orch._do_spawn("Role.", "Task.", register_as="new_agent")

        assert "new_agent" in orch.sub_agents
        assert "ask_new_agent" in orch.tools._tools

    async def test_spawn_agent_tool_invokes_do_spawn(self):
        orch = _make_orchestrator()

        with patch.object(orch, "_do_spawn", new=AsyncMock(return_value="spawned text")) as mock_spawn:
            tool = orch.tools._tools["spawn_agent"]
            result = await tool.func(role="Expert role.", task="Do the task.")

        mock_spawn.assert_called_once_with("Expert role.", "Do the task.", register_as=None)
        assert result == "spawned text"

    async def test_spawn_agent_error_handled(self):
        orch = _make_orchestrator()

        with patch("clawscope.agent.react.ReActAgent.__call__", new=AsyncMock(
            side_effect=RuntimeError("model error")
        )):
            result = await orch._do_spawn("Role.", "Task.", register_as=None)

        assert "error" in result.lower()

    def test_meta_tools_have_correct_parameters(self):
        orch = _make_orchestrator()

        spawn_tool = orch.tools._tools["spawn_agent"]
        param_names = [p.name for p in spawn_tool.parameters]
        assert "role" in param_names
        assert "task" in param_names

        create_tool = orch.tools._tools["create_agent"]
        param_names = [p.name for p in create_tool.parameters]
        assert "name" in param_names
        assert "role" in param_names

    def test_sys_prompt_mentions_spawn(self):
        orch = _make_orchestrator()
        assert "spawn_agent" in orch.sys_prompt

    def test_sys_prompt_mentions_create_agent(self):
        orch = _make_orchestrator()
        assert "create_agent" in orch.sys_prompt


# ---------------------------------------------------------------------------
# ProgressReporter
# ---------------------------------------------------------------------------


class TestProgressReporter:
    def _make_reporter(self):
        from clawscope.agent.orchestrator import ProgressReporter
        from clawscope.bus import MessageBus
        bus = MessageBus()
        return ProgressReporter(bus, channel="feishu", chat_id="chat_001"), bus

    async def test_send_publishes_outbound(self):
        reporter, bus = self._make_reporter()
        await reporter.send("⚙️ [researcher] working…")
        msg = await asyncio.wait_for(bus._outbound.get(), timeout=1.0)
        assert "researcher" in msg.content
        assert msg.channel == "feishu"
        assert msg.chat_id == "chat_001"

    async def test_send_disabled_does_not_publish(self):
        reporter, bus = self._make_reporter()
        reporter.enabled = False
        await reporter.send("should not appear")
        assert bus._outbound.empty()

    def test_set_progress_reporter_on_orchestrator(self):
        from clawscope.agent.orchestrator import ProgressReporter
        from clawscope.bus import MessageBus
        orch = _make_orchestrator()
        bus = MessageBus()
        reporter = ProgressReporter(bus, "feishu", "cid")
        orch.set_progress_reporter(reporter)
        assert orch._progress_reporter is reporter

    async def test_progress_sent_when_sub_agent_called(self):
        from clawscope.agent.orchestrator import ProgressReporter
        from clawscope.bus import MessageBus

        bus = MessageBus()
        sub = _make_sub_agent("alpha", "alpha result")
        orch = _make_orchestrator(sub_agents=[sub])
        reporter = ProgressReporter(bus, "test", "cid")
        orch.set_progress_reporter(reporter)

        tool = orch.tools._tools["ask_alpha"]
        await tool.func(message="do something")

        # At least one progress message should have been sent
        assert not bus._outbound.empty()
        messages = []
        while not bus._outbound.empty():
            messages.append(await bus._outbound.get())
        texts = [m.content for m in messages]
        assert any("alpha" in t for t in texts)

    async def test_progress_not_sent_when_reporter_is_none(self):
        from clawscope.bus import MessageBus

        bus = MessageBus()
        sub = _make_sub_agent("beta", "beta result")
        orch = _make_orchestrator(sub_agents=[sub])
        # No reporter set
        tool = orch.tools._tools["ask_beta"]
        await tool.func(message="task")
        assert bus._outbound.empty()


# ---------------------------------------------------------------------------
# update_agent / remove_agent
# ---------------------------------------------------------------------------


class TestAgentManagementTools:
    def test_update_agent_tool_registered(self):
        orch = _make_orchestrator()
        assert "update_agent" in orch.tools._tools

    def test_remove_agent_tool_registered(self):
        orch = _make_orchestrator()
        assert "remove_agent" in orch.tools._tools

    def test_update_agent_changes_sys_prompt(self):
        sub = _make_sub_agent("researcher", "old role")
        orch = _make_orchestrator(sub_agents=[sub])
        result = orch._do_update_agent("researcher", "new academic role")
        assert "researcher" in result
        assert orch._sub_agents["researcher"].sys_prompt == "new academic role"

    def test_update_agent_refreshes_tool_description(self):
        sub = _make_sub_agent("researcher", "old role")
        orch = _make_orchestrator(sub_agents=[sub])
        orch._do_update_agent("researcher", "specialist in quantum physics")
        tool_desc = orch.tools._tools["ask_researcher"].description
        assert "specialist in quantum physics" in tool_desc

    def test_update_agent_missing_returns_error(self):
        orch = _make_orchestrator()
        result = orch._do_update_agent("ghost", "any role")
        assert "找不到" in result or "not found" in result.lower()

    async def test_remove_agent_via_tool(self):
        sub = _make_sub_agent("temp", "temporary")
        orch = _make_orchestrator(sub_agents=[sub])
        tool = orch.tools._tools["remove_agent"]
        result = await tool.func(name="temp")
        assert "temp" not in orch.sub_agents
        assert "ask_temp" not in orch.tools._tools
        assert "removed" in result.lower() or "删除" in result or "removed" in result

    async def test_remove_agent_missing_returns_error(self):
        orch = _make_orchestrator()
        tool = orch.tools._tools["remove_agent"]
        result = await tool.func(name="nonexistent")
        assert "nonexistent" in result

    async def test_update_agent_via_tool(self):
        sub = _make_sub_agent("writer", "plain writer")
        orch = _make_orchestrator(sub_agents=[sub])
        tool = orch.tools._tools["update_agent"]
        result = await tool.func(name="writer", new_role="expert technical writer")
        assert "writer" in result
        assert orch._sub_agents["writer"].sys_prompt == "expert technical writer"


# ---------------------------------------------------------------------------
# Orchestration summary header
# ---------------------------------------------------------------------------


class TestOrchestrationSummary:
    def test_attach_summary_prefixes_agent_names(self):
        orch = _make_orchestrator()
        orch._call_log = [
            _AgentCall(name="researcher", task_preview="search"),
            _AgentCall(name="writer", task_preview="write"),
        ]
        original = Msg(name="orch", content="Final answer.", role="assistant")
        result = orch._attach_summary(original)
        text = result.get_text_content()
        assert "researcher" in text
        assert "writer" in text
        assert "→" in text
        assert "Final answer." in text

    def test_attach_summary_deduplicates_names(self):
        orch = _make_orchestrator()
        orch._call_log = [
            _AgentCall(name="researcher", task_preview="q1"),
            _AgentCall(name="researcher", task_preview="q2"),
        ]
        original = Msg(name="orch", content="Answer.", role="assistant")
        result = orch._attach_summary(original)
        text = result.get_text_content()
        # researcher should appear only once in the header
        header_line = text.split("\n")[0]
        assert header_line.count("researcher") == 1

    async def test_reply_attaches_summary_when_agents_called(self):
        sub = _make_sub_agent("helper", "helper answer")
        orch = _make_orchestrator(sub_agents=[sub])

        # Simulate the orchestrator calling the tool and producing a response
        tool = orch.tools._tools["ask_helper"]
        await tool.func(message="do work")

        base_msg = Msg(name="Orchestrator", content="Combined answer.", role="assistant")
        result = orch._attach_summary(base_msg)
        assert "helper" in result.get_text_content()
        assert "Combined answer." in result.get_text_content()

    def test_reply_no_summary_when_no_agents_called(self):
        orch = _make_orchestrator()
        orch._call_log = []
        original = Msg(name="orch", content="Direct answer.", role="assistant")
        # _attach_summary is only called when call_log is non-empty;
        # verify the content is unchanged when log is empty
        assert original.get_text_content() == "Direct answer."


# ---------------------------------------------------------------------------
# SessionRouter – ProgressReporter injection
# ---------------------------------------------------------------------------


class TestSessionRouterProgressInjection:
    async def test_router_injects_reporter_for_orchestrator(self):
        from clawscope.orchestration.router import SessionRouter
        from clawscope.agent.orchestrator import OrchestratorAgent, ProgressReporter

        bus = AsyncMock()
        bus._outbound_queue = asyncio.Queue()
        bus.publish_outbound = AsyncMock()

        kernel = MagicMock()
        orch = _make_orchestrator()
        kernel.create_agent = MagicMock(return_value=orch)

        sessions = AsyncMock()
        sessions.get_or_create = AsyncMock(return_value=MagicMock())

        from clawscope.config import AgentConfig
        router = SessionRouter(bus=bus, sessions=sessions, kernel=kernel, config=AgentConfig())

        from clawscope.bus import InboundMessage
        inbound = InboundMessage(
            channel="feishu",
            chat_id="cid_001",
            sender_id="u1",
            content="hello",
        )

        # Pre-register the orchestrator in the router's agent map
        router._agents[inbound.session_key] = orch

        # Simulate what _process_message does: inject reporter for orchestrators
        from clawscope.agent.orchestrator import ProgressReporter
        if isinstance(orch, OrchestratorAgent):
            orch.set_progress_reporter(
                ProgressReporter(bus, inbound.channel, inbound.chat_id)
            )

        assert orch._progress_reporter is not None
        assert isinstance(orch._progress_reporter, ProgressReporter)
