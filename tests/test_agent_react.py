"""Tests for ReActAgent with mocked model and tools."""

from __future__ import annotations

from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clawscope.agent.react import CompressionConfig, ReActAgent
from clawscope.memory import InMemoryMemory
from clawscope.message import Msg
from clawscope.model.base import ChatModelBase, ChatResponse, ToolCall


# ---------------------------------------------------------------------------
# Mock model helpers
# ---------------------------------------------------------------------------


def make_model(
    text: str = "Hello!",
    tool_calls: list[ToolCall] | None = None,
    finish_reason: str = "stop",
) -> ChatModelBase:
    """Return a mock ChatModelBase that always responds with *text* (and optional tool calls)."""
    model = MagicMock(spec=ChatModelBase)
    model.model_name = "mock-model"

    response = ChatResponse(
        content=text,
        tool_calls=tool_calls or [],
        finish_reason=finish_reason,
    )
    model.chat = AsyncMock(return_value=response)
    return model


def make_stream_model(chunks: list[ChatResponse]) -> ChatModelBase:
    """Return a mock that streams *chunks* from stream_chat()."""
    model = MagicMock(spec=ChatModelBase)
    model.model_name = "stream-mock"

    async def _stream(*args, **kwargs) -> AsyncIterator[ChatResponse]:
        for chunk in chunks:
            yield chunk

    model.stream_chat = _stream
    model.chat = AsyncMock(return_value=chunks[-1] if chunks else ChatResponse(content=""))
    return model


def user_msg(text: str) -> Msg:
    return Msg(name="user", content=text, role="user")


# ---------------------------------------------------------------------------
# Basic reply
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_react_reply_returns_msg() -> None:
    agent = ReActAgent(name="Bot", model=make_model("Hi there!"), memory=InMemoryMemory())
    result = await agent.reply(user_msg("Hello"))
    assert isinstance(result, Msg)
    assert result.role == "assistant"
    assert result.name == "Bot"


@pytest.mark.asyncio
async def test_react_reply_text_content() -> None:
    agent = ReActAgent(name="Bot", model=make_model("42"), memory=InMemoryMemory())
    result = await agent.reply(user_msg("What is 6×7?"))
    assert "42" in result.get_text_content()


@pytest.mark.asyncio
async def test_react_reply_stores_messages_in_memory() -> None:
    mem = InMemoryMemory()
    agent = ReActAgent(name="Bot", model=make_model("ok"), memory=mem)
    await agent.reply(user_msg("test"))
    # user + assistant
    assert await mem.size() >= 2


@pytest.mark.asyncio
async def test_react_reply_without_message() -> None:
    """reply(None) should still return a Msg."""
    agent = ReActAgent(name="Bot", model=make_model("empty"), memory=InMemoryMemory())
    result = await agent.reply(None)
    assert isinstance(result, Msg)


@pytest.mark.asyncio
async def test_react_no_model_raises() -> None:
    agent = ReActAgent(name="Bot", model=None, memory=InMemoryMemory())
    with pytest.raises(RuntimeError, match="No model configured"):
        await agent.reply(user_msg("hello"))


# ---------------------------------------------------------------------------
# Tool calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_react_executes_tool_then_responds() -> None:
    """
    First call returns a tool-use request; second call returns the final answer.
    """
    tool_call = ToolCall(id="tc1", name="calculator", arguments={"expr": "6*7"})

    model = MagicMock(spec=ChatModelBase)
    model.model_name = "mock"
    model.chat = AsyncMock(
        side_effect=[
            ChatResponse(content=None, tool_calls=[tool_call], finish_reason="tool_calls"),
            ChatResponse(content="The answer is 42", finish_reason="stop"),
        ]
    )

    tool_registry = MagicMock()
    tool_registry.get_definitions = MagicMock(return_value=[
        {"type": "function", "function": {"name": "calculator"}}
    ])
    tool_registry.execute = AsyncMock(return_value="42")

    agent = ReActAgent(
        name="Bot",
        model=model,
        memory=InMemoryMemory(),
        tools=tool_registry,
    )

    result = await agent.reply(user_msg("What is 6×7?"))
    tool_registry.execute.assert_awaited_once_with("calculator", {"expr": "6*7"})
    assert "42" in result.get_text_content()


@pytest.mark.asyncio
async def test_react_max_iterations_reached() -> None:
    """Agent should return a fallback message when max_iterations is exhausted."""
    tool_call = ToolCall(id="tc1", name="loop", arguments={})
    looping_response = ChatResponse(
        content=None, tool_calls=[tool_call], finish_reason="tool_calls"
    )

    model = MagicMock(spec=ChatModelBase)
    model.model_name = "mock"
    model.chat = AsyncMock(return_value=looping_response)

    tool_registry = MagicMock()
    tool_registry.get_definitions = MagicMock(return_value=[])
    tool_registry.execute = AsyncMock(return_value="still looping")

    agent = ReActAgent(
        name="Bot",
        model=model,
        memory=InMemoryMemory(),
        tools=tool_registry,
        max_iterations=3,
    )

    result = await agent.reply(user_msg("loop forever"))
    assert isinstance(result, Msg)
    assert model.chat.await_count == 3


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_react_pre_reply_hook_modifies_kwargs() -> None:
    """pre-reply hook can inject keyword args (not message)."""
    agent = ReActAgent(name="Bot", model=make_model("ok"), memory=InMemoryMemory())

    hook_called = []

    @agent.on_pre_reply
    async def hook(**kwargs):
        hook_called.append(True)
        return kwargs

    await agent.reply(user_msg("hi"))
    assert hook_called


@pytest.mark.asyncio
async def test_react_post_reply_hook_transforms_output() -> None:
    agent = ReActAgent(name="Bot", model=make_model("original"), memory=InMemoryMemory())

    @agent.on_post_reply
    async def hook(output: Msg) -> Msg:
        return Msg(name=output.name, content="TRANSFORMED", role="assistant")

    result = await agent.reply(user_msg("hi"))
    assert result.get_text_content() == "TRANSFORMED"


@pytest.mark.asyncio
async def test_react_observe_adds_to_memory() -> None:
    mem = InMemoryMemory()
    agent = ReActAgent(name="Bot", model=make_model("ok"), memory=mem)
    msg = Msg(name="user", content="observed", role="user")
    await agent.observe(msg)
    assert await mem.size() == 1


@pytest.mark.asyncio
async def test_react_clear_memory() -> None:
    mem = InMemoryMemory()
    agent = ReActAgent(name="Bot", model=make_model("ok"), memory=mem)
    await agent.observe(user_msg("remember this"))
    await agent.clear_memory()
    assert await mem.size() == 0


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_react_system_prompt_included_in_conversation() -> None:
    model = make_model("ok")
    agent = ReActAgent(
        name="Bot",
        sys_prompt="You are a helpful assistant.",
        model=model,
        memory=InMemoryMemory(),
    )
    await agent.reply(user_msg("hi"))

    call_args = model.chat.call_args
    messages = call_args[1].get("messages") or call_args[0][0]
    roles = [m.role for m in messages]
    assert "system" in roles


# ---------------------------------------------------------------------------
# stream_reply
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_react_stream_reply_yields_content_chunk() -> None:
    chunks = [
        ChatResponse(content="Hello", finish_reason=None),
        ChatResponse(content=" world", finish_reason="stop"),
    ]
    model = make_stream_model(chunks)
    agent = ReActAgent(name="Bot", model=model, memory=InMemoryMemory())

    events = []
    async for event in agent.stream_reply(user_msg("hi")):
        events.append(event)

    content_events = [e for e in events if e["type"] == "content"]
    assert len(content_events) >= 1
    combined = "".join(e["content"] for e in content_events)
    assert "Hello" in combined


@pytest.mark.asyncio
async def test_react_stream_reply_done_event_at_end() -> None:
    chunks = [ChatResponse(content="done!", finish_reason="stop")]
    model = make_stream_model(chunks)
    agent = ReActAgent(name="Bot", model=model, memory=InMemoryMemory())

    events = []
    async for event in agent.stream_reply(user_msg("hi")):
        events.append(event)

    assert events[-1]["type"] == "done"


@pytest.mark.asyncio
async def test_react_stream_reply_with_tool_call() -> None:
    """Tools should be executed exactly once and results included in the stream."""
    tool_call = ToolCall(id="tc1", name="calc", arguments={"x": 1})

    tool_chunk = ChatResponse(
        content=None,
        tool_calls=[tool_call],
        finish_reason="tool_calls",
    )
    final_chunk = ChatResponse(content="result=2", finish_reason="stop")

    call_count = {"n": 0}

    async def _stream(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            yield tool_chunk
        else:
            yield final_chunk

    model = MagicMock(spec=ChatModelBase)
    model.model_name = "mock"
    model.stream_chat = _stream

    tool_registry = MagicMock()
    tool_registry.get_definitions = MagicMock(return_value=[])
    tool_registry.execute = AsyncMock(return_value="2")

    agent = ReActAgent(
        name="Bot", model=model, memory=InMemoryMemory(), tools=tool_registry
    )

    events = []
    async for event in agent.stream_reply(user_msg("1+1")):
        events.append(event)

    # Tool executed exactly once
    tool_registry.execute.assert_awaited_once_with("calc", {"x": 1})

    types = [e["type"] for e in events]
    assert "tool_start" in types
    assert "tool_result" in types
    assert "done" in types


# ---------------------------------------------------------------------------
# __call__ / __str__
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_react_callable() -> None:
    agent = ReActAgent(name="Bot", model=make_model("hi"), memory=InMemoryMemory())
    result = await agent(user_msg("hello"))
    assert isinstance(result, Msg)


def test_react_str_representation() -> None:
    agent = ReActAgent(name="MyBot", model=None)
    assert "MyBot" in str(agent)
