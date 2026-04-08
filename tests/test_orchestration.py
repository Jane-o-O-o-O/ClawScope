"""Tests for multi-agent orchestration (MsgHub, Pipeline)."""

from unittest.mock import AsyncMock

import pytest

from clawscope.message import Msg
from clawscope.orchestration.msghub import MsgHub, MsgHubBuilder, Participant
from clawscope.orchestration.pipeline import (
    ConditionalPipeline,
    FanOutPipeline,
    SequentialPipeline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_agent(name: str, reply_text: str) -> AsyncMock:
    """Create a mock agent that always returns *reply_text*."""
    agent = AsyncMock()
    agent.name = name
    agent.return_value = Msg(name=name, content=reply_text, role="assistant")
    return agent


def user_msg(text: str) -> Msg:
    return Msg(name="user", content=text, role="user")


# ---------------------------------------------------------------------------
# MsgHub – participant management
# ---------------------------------------------------------------------------


def test_msghub_add_participant() -> None:
    hub = MsgHub()
    agent = make_agent("Alice", "hi")
    hub.add(agent)
    assert len(hub.active_participants) == 1


def test_msghub_add_multiple_participants() -> None:
    hub = MsgHub()
    hub.add(make_agent("Alice", "a"))
    hub.add(make_agent("Bob", "b"))
    assert len(hub.active_participants) == 2


def test_msghub_remove_participant() -> None:
    hub = MsgHub()
    agent = make_agent("Alice", "hi")
    hub.add(agent)
    removed = hub.remove("Alice")
    assert removed is True
    assert len(hub.active_participants) == 0


def test_msghub_remove_nonexistent_returns_false() -> None:
    hub = MsgHub()
    assert hub.remove("Nobody") is False


def test_msghub_deactivate_and_activate() -> None:
    hub = MsgHub()
    hub.add(make_agent("Bob", "hello"))
    hub.deactivate("Bob")
    assert len(hub.active_participants) == 0
    hub.activate("Bob")
    assert len(hub.active_participants) == 1


def test_msghub_initial_participants_from_constructor() -> None:
    agents = [make_agent(f"A{i}", f"reply{i}") for i in range(3)]
    hub = MsgHub(participants=agents)
    assert len(hub.active_participants) == 3


def test_msghub_announcement_in_messages() -> None:
    announcement = Msg(name="system", content="Let's begin", role="system")
    hub = MsgHub(announcement=announcement)
    assert len(hub.messages) == 1
    assert hub.messages[0].content == "Let's begin"


# ---------------------------------------------------------------------------
# MsgHub – run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_msghub_run_one_round() -> None:
    alice = make_agent("Alice", "I agree")
    hub = MsgHub(participants=[alice], max_rounds=1)
    messages = await hub.run(initial_message=user_msg("Start"))
    # initial + 1 response
    assert len(messages) >= 2
    alice.assert_called_once()


@pytest.mark.asyncio
async def test_msghub_run_multiple_agents() -> None:
    alice = make_agent("Alice", "Alice says hi")
    bob = make_agent("Bob", "Bob says hi")
    hub = MsgHub(participants=[alice, bob], max_rounds=1)
    messages = await hub.run(initial_message=user_msg("hello"))
    # initial + 2 responses
    assert len(messages) == 3


@pytest.mark.asyncio
async def test_msghub_run_custom_order() -> None:
    alice = make_agent("Alice", "A")
    bob = make_agent("Bob", "B")
    hub = MsgHub(participants=[alice, bob], max_rounds=1)
    await hub.run(initial_message=user_msg("go"), order=["Bob", "Alice"])
    # Order shouldn't affect count – both called once
    alice.assert_called_once()
    bob.assert_called_once()


@pytest.mark.asyncio
async def test_msghub_no_participants_returns_initial() -> None:
    hub = MsgHub(max_rounds=3)
    initial = user_msg("hello")
    messages = await hub.run(initial_message=initial)
    assert len(messages) == 1
    assert messages[0].content == "hello"


@pytest.mark.asyncio
async def test_msghub_termination_condition() -> None:
    counter = {"n": 0}

    agent = AsyncMock()
    agent.name = "Agent"

    async def side_effect(msg):
        counter["n"] += 1
        return Msg(name="Agent", content="STOP" if counter["n"] >= 2 else "continue", role="assistant")

    agent.side_effect = side_effect

    def stop_on_stop(msgs):
        return any(
            "STOP" in (m.get_text_content() or "")
            for m in msgs
            if m.role == "assistant"
        )

    hub = MsgHub(participants=[agent], max_rounds=10, termination_condition=stop_on_stop)
    await hub.run(initial_message=user_msg("go"))
    # Should have stopped at round 2
    assert counter["n"] == 2


@pytest.mark.asyncio
async def test_msghub_broadcast() -> None:
    hub = MsgHub()
    msg = Msg(name="admin", content="broadcast!", role="user")
    hub.broadcast(msg)
    assert len(hub.messages) == 1


# ---------------------------------------------------------------------------
# MsgHubBuilder
# ---------------------------------------------------------------------------


def test_msghub_builder_creates_hub() -> None:
    agent = make_agent("X", "y")
    hub = (
        MsgHubBuilder()
        .add_participant(agent)
        .with_announcement("Start!")
        .max_rounds(5)
        .build()
    )
    assert hub.max_rounds == 5
    assert len(hub.active_participants) == 1
    assert len(hub.messages) == 1  # announcement


def test_msghub_builder_terminate_when() -> None:
    hub = (
        MsgHubBuilder()
        .add_participant(make_agent("A", "done"))
        .max_rounds(3)
        .terminate_when(lambda msgs: len(msgs) >= 2)
        .build()
    )
    assert hub.termination_condition is not None
    assert hub.termination_condition([Msg(name="a", content="x", role="user"),
                                      Msg(name="b", content="y", role="user")]) is True


def test_msghub_builder_add_participants_list() -> None:
    agents = [make_agent(f"A{i}", "r") for i in range(4)]
    hub = MsgHubBuilder().add_participants(agents).build()
    assert len(hub.active_participants) == 4


# ---------------------------------------------------------------------------
# SequentialPipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sequential_pipeline_passes_output_to_next() -> None:
    """Each agent receives the output of the previous one."""
    calls = []

    async def make_agent_fn(name: str, suffix: str):
        agent = AsyncMock()
        agent.name = name

        async def side_effect(msg):
            calls.append((name, msg.get_text_content()))
            return Msg(name=name, content=msg.get_text_content() + suffix, role="assistant")

        agent.side_effect = side_effect
        return agent

    a1 = await make_agent_fn("A1", "-A1")
    a2 = await make_agent_fn("A2", "-A2")
    a3 = await make_agent_fn("A3", "-A3")

    pipeline = SequentialPipeline(agents=[a1, a2, a3])
    result = await pipeline.run(user_msg("start"))

    assert result is not None
    text = result.get_text_content()
    assert "A1" in text
    assert "A2" in text
    assert "A3" in text
    # Each agent was called once
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_sequential_pipeline_empty_returns_input() -> None:
    pipeline = SequentialPipeline(agents=[])
    initial = user_msg("hello")
    result = await pipeline.run(initial)
    assert result.content == "hello"


# ---------------------------------------------------------------------------
# FanOutPipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fanout_pipeline_calls_all_agents() -> None:
    agents = [make_agent(f"A{i}", f"reply{i}") for i in range(3)]
    pipeline = FanOutPipeline(agents=agents)
    results = await pipeline.run(user_msg("question"))
    assert len(results) == 3
    for agent in agents:
        agent.assert_called_once()
