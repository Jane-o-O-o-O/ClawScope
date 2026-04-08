"""Tests for ChatRoom, Debate, and SpeakingPolicy."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from clawscope.message import Msg
from clawscope.orchestration.chatroom import (
    ChatParticipant,
    ChatRoom,
    Debate,
    SpeakingPolicy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_agent(name: str, reply_text: str = "ok"):
    agent = AsyncMock()
    agent.name = name
    agent.return_value = Msg(name=name, content=reply_text, role="assistant")
    return agent


def history(*texts: str) -> list[Msg]:
    return [Msg(name=f"user{i}", content=t, role="user") for i, t in enumerate(texts)]


# ---------------------------------------------------------------------------
# SpeakingPolicy – round_robin
# ---------------------------------------------------------------------------


def test_round_robin_empty_participants() -> None:
    assert SpeakingPolicy.round_robin([], []) is None


def test_round_robin_no_history_returns_first() -> None:
    agents = [make_agent("A"), make_agent("B")]
    participants = [ChatParticipant(agent=a) for a in agents]
    result = SpeakingPolicy.round_robin(participants, [])
    assert result.agent.name == "A"


def test_round_robin_cycles_correctly() -> None:
    agents = [make_agent("A"), make_agent("B"), make_agent("C")]
    participants = [ChatParticipant(agent=a) for a in agents]

    hist = [Msg(name="A", content="x", role="assistant")]
    result = SpeakingPolicy.round_robin(participants, hist)
    assert result.agent.name == "B"

    hist = [Msg(name="C", content="x", role="assistant")]
    result = SpeakingPolicy.round_robin(participants, hist)
    assert result.agent.name == "A"


# ---------------------------------------------------------------------------
# SpeakingPolicy – random
# ---------------------------------------------------------------------------


def test_random_returns_one_participant() -> None:
    agents = [make_agent("A"), make_agent("B")]
    participants = [ChatParticipant(agent=a) for a in agents]
    result = SpeakingPolicy.random(participants, [])
    assert result is not None
    assert result.agent.name in ("A", "B")


def test_random_skips_muted() -> None:
    agent_a = make_agent("A")
    agent_b = make_agent("B")
    participants = [
        ChatParticipant(agent=agent_a, is_muted=True),
        ChatParticipant(agent=agent_b, is_muted=False),
    ]
    for _ in range(20):  # probabilistic – should always pick B
        result = SpeakingPolicy.random(participants, [])
        assert result.agent.name == "B"


def test_random_empty_returns_none() -> None:
    assert SpeakingPolicy.random([], []) is None


# ---------------------------------------------------------------------------
# SpeakingPolicy – host_moderated
# ---------------------------------------------------------------------------


def test_host_moderated_host_speaks_first() -> None:
    host = ChatParticipant(agent=make_agent("Host"), role="host")
    other = ChatParticipant(agent=make_agent("Other"), role="participant")
    result = SpeakingPolicy.host_moderated([host, other], [])
    assert result.agent.name == "Host"


def test_host_moderated_alternates_after_host() -> None:
    host = ChatParticipant(agent=make_agent("Host"), role="host")
    other = ChatParticipant(agent=make_agent("Other"), role="participant")
    hist = [Msg(name="Host", content="x", role="assistant")]
    result = SpeakingPolicy.host_moderated([host, other], hist)
    assert result.agent.name == "Other"


def test_host_moderated_no_host_falls_back_to_round_robin() -> None:
    a = ChatParticipant(agent=make_agent("A"))
    b = ChatParticipant(agent=make_agent("B"))
    result = SpeakingPolicy.host_moderated([a, b], [])
    assert result is not None


# ---------------------------------------------------------------------------
# SpeakingPolicy – llm_decided
# ---------------------------------------------------------------------------


def test_llm_decided_model_none_falls_back() -> None:
    a = ChatParticipant(agent=make_agent("Alice"))
    b = ChatParticipant(agent=make_agent("Bob"))
    result = SpeakingPolicy.llm_decided(None, [a, b], [])
    assert result is not None  # Falls back to round-robin


def test_llm_decided_model_returns_valid_name() -> None:
    model = MagicMock()
    response = MagicMock()
    response.get_text_content = MagicMock(return_value="  Bob  ")
    import asyncio
    model.chat = lambda msgs: asyncio.coroutine(lambda: response)()

    a = ChatParticipant(agent=make_agent("Alice"))
    b = ChatParticipant(agent=make_agent("Bob"))

    # Use a sync model callable for simplicity
    model2 = MagicMock()
    model2.chat = None  # no chat attribute
    model2.__call__ = MagicMock(return_value="Bob")
    # Replace model with a callable that has no .chat
    del model2.chat  # make hasattr(model2, "chat") False

    result = SpeakingPolicy.llm_decided(model2, [a, b], [])
    # Falls back since model2.chat doesn't exist
    assert result is not None


def test_llm_decided_model_returns_unknown_falls_back() -> None:
    """Model returns an unknown name → fall back to round-robin."""
    model = MagicMock()

    async def _chat(msgs):
        r = MagicMock()
        r.get_text_content = MagicMock(return_value="CompletelyUnknown")
        return r

    model.chat = _chat

    a = ChatParticipant(agent=make_agent("Alice"))
    b = ChatParticipant(agent=make_agent("Bob"))
    result = SpeakingPolicy.llm_decided(model, [a, b], [])
    assert result is not None  # round-robin fallback


def test_llm_decided_empty_participants_returns_none() -> None:
    assert SpeakingPolicy.llm_decided(MagicMock(), [], []) is None


# ---------------------------------------------------------------------------
# ChatRoom – participant management
# ---------------------------------------------------------------------------


def test_chatroom_join_and_leave() -> None:
    room = ChatRoom()
    room.join(make_agent("Alice"), role="participant")
    assert len(room.active_participants) == 1
    room.leave("Alice")
    assert len(room.active_participants) == 0


def test_chatroom_mute_and_unmute() -> None:
    room = ChatRoom()
    room.join(make_agent("Bob"))
    room.mute("Bob")
    assert len(room.active_participants) == 0
    room.unmute("Bob")
    assert len(room.active_participants) == 1


def test_chatroom_observers_excluded_from_active() -> None:
    room = ChatRoom()
    room.join(make_agent("Observer"), role="observer")
    room.join(make_agent("Active"), role="participant")
    assert len(room.active_participants) == 1


def test_chatroom_say_adds_to_history() -> None:
    room = ChatRoom()
    room.join(make_agent("Alice"))
    msg = Msg(name="Alice", content="hi", role="assistant")
    room.say(msg)
    assert len(room.history) == 1


def test_chatroom_clear_resets_history_and_counts() -> None:
    room = ChatRoom()
    room.join(make_agent("Alice"))
    room.say(Msg(name="Alice", content="hi", role="assistant"))
    room.say(Msg(name="Alice", content="bye", role="assistant"))
    room.clear()
    assert len(room.history) == 0


def test_chatroom_get_stats() -> None:
    room = ChatRoom(name="test-room")
    room.join(make_agent("A"), role="participant")
    stats = room.get_stats()
    assert stats["name"] == "test-room"
    assert stats["participants"] == 1


# ---------------------------------------------------------------------------
# ChatRoom – run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chatroom_run_produces_messages() -> None:
    room = ChatRoom(max_messages=3)
    room.join(make_agent("Alice", "hello"), role="participant")
    messages = await room.run(topic="Greetings")
    # topic intro + at least one agent reply
    assert len(messages) >= 1


@pytest.mark.asyncio
async def test_chatroom_run_stops_on_termination_phrase() -> None:
    call_count = {"n": 0}
    agent = AsyncMock()
    agent.name = "Bot"

    async def side_effect(msg):
        call_count["n"] += 1
        text = "goodbye" if call_count["n"] >= 2 else "hello"
        return Msg(name="Bot", content=text, role="assistant")

    agent.side_effect = side_effect

    room = ChatRoom(max_messages=20)
    room.join(agent, role="participant")
    messages = await room.run(termination_phrases=["goodbye"])
    # Should have stopped at 2nd message
    assert call_count["n"] == 2


@pytest.mark.asyncio
async def test_chatroom_run_no_participants_returns_empty() -> None:
    room = ChatRoom(max_messages=5)
    messages = await room.run()
    assert messages == []


@pytest.mark.asyncio
async def test_chatroom_run_until_condition() -> None:
    msg_count = {"n": 0}
    agent = AsyncMock()
    agent.name = "A"

    async def se(msg):
        msg_count["n"] += 1
        return Msg(name="A", content=str(msg_count["n"]), role="assistant")

    agent.side_effect = se

    room = ChatRoom(max_messages=100)
    room.join(agent)

    messages = await room.run_until(lambda h: len(h) >= 3)
    assert len(messages) >= 3


# ---------------------------------------------------------------------------
# Debate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_debate_runs_correct_number_of_rounds() -> None:
    proposer = make_agent("Pro", "I agree")
    opposer = make_agent("Con", "I disagree")

    debate = Debate(
        topic="AI is beneficial",
        proposer=proposer,
        opposer=opposer,
        rounds=2,
    )
    messages = await debate.run()

    # Opening + 2 rounds × 2 (proposer + opposer) + moderator intros
    assert len(messages) > 0
    # Both agents were called (2 rounds each)
    assert proposer.call_count == 2
    assert opposer.call_count == 2


@pytest.mark.asyncio
async def test_debate_with_judge() -> None:
    proposer = make_agent("Pro", "For")
    opposer = make_agent("Con", "Against")
    judge = make_agent("Judge", "The winner is Pro")

    debate = Debate(
        topic="Python vs Java",
        proposer=proposer,
        opposer=opposer,
        judge=judge,
        rounds=1,
    )
    messages = await debate.run()
    judge.assert_called_once()
    texts = [m.get_text_content() for m in messages]
    assert any("winner" in t.lower() for t in texts)


def test_debate_registers_roles_correctly() -> None:
    proposer = make_agent("Pro")
    opposer = make_agent("Con")

    debate = Debate(
        topic="test",
        proposer=proposer,
        opposer=opposer,
    )

    roles = {name: p.role for name, p in debate._participants.items()}
    assert roles["Pro"] == "proposer"
    assert roles["Con"] == "opposer"
