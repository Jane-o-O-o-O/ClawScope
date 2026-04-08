"""Tests for the message bus (events + queue)."""

import asyncio

import pytest

from clawscope.bus import InboundMessage, MessageBus, OutboundMessage
from clawscope.bus.events import SystemEvent


# ---------------------------------------------------------------------------
# InboundMessage
# ---------------------------------------------------------------------------


def test_inbound_session_key_default() -> None:
    msg = InboundMessage(
        channel="telegram",
        chat_id="123",
        sender_id="user1",
        content="hello",
    )
    assert msg.session_key == "telegram:123"


def test_inbound_session_key_override() -> None:
    msg = InboundMessage(
        channel="telegram",
        chat_id="123",
        sender_id="user1",
        content="hello",
        session_key_override="custom:session",
    )
    assert msg.session_key == "custom:session"


def test_inbound_has_auto_id() -> None:
    msg = InboundMessage(channel="slack", chat_id="C1", sender_id="U1", content="x")
    assert msg.id
    assert len(msg.id) > 0


def test_inbound_to_dict_roundtrip() -> None:
    msg = InboundMessage(
        channel="discord",
        chat_id="999",
        sender_id="user2",
        content="test message",
        media=["https://example.com/img.png"],
    )
    d = msg.to_dict()
    assert d["channel"] == "discord"
    assert d["content"] == "test message"
    assert len(d["media"]) == 1

    restored = InboundMessage.from_dict(d)
    assert restored.channel == "discord"
    assert restored.content == "test message"
    assert restored.chat_id == "999"


def test_two_inbounds_have_different_ids() -> None:
    a = InboundMessage(channel="tg", chat_id="1", sender_id="u", content="a")
    b = InboundMessage(channel="tg", chat_id="1", sender_id="u", content="b")
    assert a.id != b.id


# ---------------------------------------------------------------------------
# OutboundMessage
# ---------------------------------------------------------------------------


def test_outbound_to_dict_roundtrip() -> None:
    msg = OutboundMessage(
        channel="feishu",
        chat_id="oc_123",
        content="reply text",
        reply_to="msg_99",
    )
    d = msg.to_dict()
    assert d["channel"] == "feishu"
    assert d["reply_to"] == "msg_99"

    restored = OutboundMessage.from_dict(d)
    assert restored.content == "reply text"
    assert restored.reply_to == "msg_99"


# ---------------------------------------------------------------------------
# MessageBus – inbound queue
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bus_publish_and_consume_inbound() -> None:
    bus = MessageBus()
    msg = InboundMessage(channel="telegram", chat_id="1", sender_id="u", content="ping")
    await bus.publish_inbound(msg)
    received = await bus.consume_inbound()
    assert received.content == "ping"
    assert received.channel == "telegram"


@pytest.mark.asyncio
async def test_bus_inbound_fifo_order() -> None:
    bus = MessageBus()
    for i in range(3):
        await bus.publish_inbound(
            InboundMessage(channel="tg", chat_id="1", sender_id="u", content=f"msg{i}")
        )
    contents = []
    for _ in range(3):
        m = await bus.consume_inbound()
        contents.append(m.content)
    assert contents == ["msg0", "msg1", "msg2"]


@pytest.mark.asyncio
async def test_bus_inbound_stats_count() -> None:
    bus = MessageBus()
    for _ in range(5):
        await bus.publish_inbound(
            InboundMessage(channel="slack", chat_id="c", sender_id="u", content="x")
        )
    assert bus.stats["inbound_count"] == 5


@pytest.mark.asyncio
async def test_bus_inbound_listener_called() -> None:
    bus = MessageBus()
    received = []

    async def listener(msg: InboundMessage) -> None:
        received.append(msg.content)

    bus.on_inbound(listener)
    await bus.publish_inbound(
        InboundMessage(channel="tg", chat_id="1", sender_id="u", content="hello")
    )
    assert received == ["hello"]


@pytest.mark.asyncio
async def test_bus_inbound_listener_unsubscribe() -> None:
    bus = MessageBus()
    received = []

    async def listener(msg: InboundMessage) -> None:
        received.append(msg.content)

    unsubscribe = bus.on_inbound(listener)
    await bus.publish_inbound(
        InboundMessage(channel="tg", chat_id="1", sender_id="u", content="first")
    )
    unsubscribe()
    await bus.publish_inbound(
        InboundMessage(channel="tg", chat_id="1", sender_id="u", content="second")
    )
    assert received == ["first"]


# ---------------------------------------------------------------------------
# MessageBus – outbound queue
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bus_publish_and_consume_outbound_global() -> None:
    bus = MessageBus()
    msg = OutboundMessage(channel="telegram", chat_id="1", content="pong")
    await bus.publish_outbound(msg)
    received = await bus.consume_outbound()
    assert received.content == "pong"


@pytest.mark.asyncio
async def test_bus_outbound_channel_routing() -> None:
    bus = MessageBus()
    await bus.publish_outbound(OutboundMessage(channel="telegram", chat_id="1", content="to tg"))
    await bus.publish_outbound(OutboundMessage(channel="discord", chat_id="2", content="to dc"))

    tg_msg = await bus.consume_outbound_for_channel("telegram")
    dc_msg = await bus.consume_outbound_for_channel("discord")
    assert tg_msg is not None and tg_msg.content == "to tg"
    assert dc_msg is not None and dc_msg.content == "to dc"


@pytest.mark.asyncio
async def test_bus_outbound_channel_timeout() -> None:
    bus = MessageBus()
    result = await bus.consume_outbound_for_channel("telegram", timeout=0.05)
    assert result is None


@pytest.mark.asyncio
async def test_bus_outbound_stats_count() -> None:
    bus = MessageBus()
    for _ in range(3):
        await bus.publish_outbound(OutboundMessage(channel="slack", chat_id="c", content="x"))
    assert bus.stats["outbound_count"] == 3


# ---------------------------------------------------------------------------
# MessageBus – system events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bus_system_event() -> None:
    bus = MessageBus()
    event = SystemEvent(event_type="health_check", payload={"status": "ok"})
    await bus.publish_system_event(event)
    received = await bus.consume_system_event()
    assert received.event_type == "health_check"
    assert received.payload["status"] == "ok"


# ---------------------------------------------------------------------------
# MessageBus – misc
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bus_queue_size_properties() -> None:
    bus = MessageBus()
    assert bus.inbound_size == 0
    assert bus.outbound_size == 0
    await bus.publish_inbound(
        InboundMessage(channel="tg", chat_id="1", sender_id="u", content="x")
    )
    assert bus.inbound_size == 1


@pytest.mark.asyncio
async def test_bus_clear() -> None:
    bus = MessageBus()
    await bus.publish_inbound(
        InboundMessage(channel="tg", chat_id="1", sender_id="u", content="x")
    )
    await bus.publish_outbound(OutboundMessage(channel="tg", chat_id="1", content="y"))
    bus.clear()
    assert bus.inbound_size == 0
    assert bus.outbound_size == 0
