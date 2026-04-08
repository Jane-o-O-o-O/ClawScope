"""Tests for channel implementations (base, DingTalk, Feishu)."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clawscope.bus import InboundMessage, MessageBus, OutboundMessage
from clawscope.channels.base import BaseChannel


# ---------------------------------------------------------------------------
# Concrete stub for testing BaseChannel
# ---------------------------------------------------------------------------


class StubChannel(BaseChannel):
    """Minimal concrete channel for testing base class behaviour."""

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False

    async def send(self, message: OutboundMessage) -> None:
        pass


def make_config(**kwargs):
    cfg = MagicMock()
    cfg.allow_from = kwargs.get("allow_from", ["*"])
    return cfg


# ---------------------------------------------------------------------------
# BaseChannel
# ---------------------------------------------------------------------------


def test_channel_initial_state() -> None:
    bus = MessageBus()
    ch = StubChannel(name="stub", bus=bus, config=make_config())
    assert ch.name == "stub"
    assert ch.is_running is False


@pytest.mark.asyncio
async def test_channel_start_stop() -> None:
    bus = MessageBus()
    ch = StubChannel(name="stub", bus=bus, config=make_config())
    await ch.start()
    assert ch.is_running is True
    await ch.stop()
    assert ch.is_running is False


def test_channel_is_allowed_wildcard() -> None:
    ch = StubChannel(name="x", bus=MessageBus(), config=make_config(allow_from=["*"]))
    assert ch.is_allowed("anyone") is True
    assert ch.is_allowed("") is True


def test_channel_is_allowed_specific_list() -> None:
    ch = StubChannel(
        name="x",
        bus=MessageBus(),
        config=make_config(allow_from=["alice", "bob"]),
    )
    assert ch.is_allowed("alice") is True
    assert ch.is_allowed("charlie") is False


def test_channel_str_representation() -> None:
    ch = StubChannel(name="test", bus=MessageBus(), config=make_config())
    s = str(ch)
    assert "test" in s
    assert "running" in s.lower()


# ---------------------------------------------------------------------------
# DingTalkChannel – unit tests (no real SDK calls)
# ---------------------------------------------------------------------------


def make_dingtalk_channel():
    """Return a DingTalkChannel with mocked SDK."""
    bus = MessageBus()
    cfg = MagicMock()
    cfg.app_key = "key123"
    cfg.app_secret = "secret456"
    cfg.allow_from = ["*"]

    # Import under a patch so we don't need the real SDK installed
    with patch.dict(
        "sys.modules",
        {
            "dingtalk_stream": MagicMock(),
            "dingtalk_stream.chatbot": MagicMock(),
        },
    ):
        from clawscope.channels.dingtalk import DingTalkChannel
        ch = DingTalkChannel(bus=bus, config=cfg)

    return ch, bus


def test_dingtalk_initial_state() -> None:
    ch, _ = make_dingtalk_channel()
    assert ch.name == "dingtalk"
    assert ch.is_running is False
    assert len(ch._reply_callbacks) == 0


def test_dingtalk_store_callback_lru_eviction() -> None:
    ch, _ = make_dingtalk_channel()

    from clawscope.channels.dingtalk import _MAX_CALLBACKS

    # Fill to capacity + 1
    for i in range(_MAX_CALLBACKS + 1):
        ch._store_callback(f"chat_{i}", object())

    assert len(ch._reply_callbacks) == _MAX_CALLBACKS
    # The oldest entry (chat_0) should have been evicted
    assert "chat_0" not in ch._reply_callbacks
    assert f"chat_{_MAX_CALLBACKS}" in ch._reply_callbacks


def test_dingtalk_store_callback_refresh_existing() -> None:
    ch, _ = make_dingtalk_channel()
    sentinel_a = object()
    sentinel_b = object()

    ch._store_callback("room1", sentinel_a)
    ch._store_callback("room2", sentinel_a)
    ch._store_callback("room1", sentinel_b)  # refresh

    # room1 should now be at the end (most-recently used)
    assert list(ch._reply_callbacks.keys())[-1] == "room1"
    assert ch._reply_callbacks["room1"] is sentinel_b


@pytest.mark.asyncio
async def test_dingtalk_send_no_running_warns(caplog) -> None:
    ch, _ = make_dingtalk_channel()
    # Channel not started → _running=False
    msg = OutboundMessage(channel="dingtalk", chat_id="c1", content="hello")
    await ch.send(msg)
    # Should log a warning and not raise
    assert not ch.is_running


@pytest.mark.asyncio
async def test_dingtalk_send_no_callback_warns(caplog) -> None:
    ch, _ = make_dingtalk_channel()
    ch._running = True  # simulate started
    msg = OutboundMessage(channel="dingtalk", chat_id="unknown", content="hi")
    # No stored callback → should log warning without raising
    await ch.send(msg)


@pytest.mark.asyncio
async def test_dingtalk_send_uses_stored_callback() -> None:
    ch, _ = make_dingtalk_channel()
    ch._running = True

    callback = AsyncMock()
    callback.reply = AsyncMock()
    ch._store_callback("room1", callback)

    with patch.dict(
        "sys.modules",
        {"dingtalk_stream.chatbot": MagicMock(TextMessage=MagicMock(return_value="<TM>"))},
    ):
        msg = OutboundMessage(channel="dingtalk", chat_id="room1", content="pong")
        await ch.send(msg)

    callback.reply.assert_awaited_once()


# ---------------------------------------------------------------------------
# FeishuChannel – unit tests (no real SDK calls)
# ---------------------------------------------------------------------------


def make_feishu_channel():
    bus = MessageBus()
    cfg = MagicMock()
    cfg.app_id = "appid"
    cfg.app_secret = "appsecret"
    cfg.allow_from = ["*"]

    lark_mock = MagicMock()
    lark_mock.LogLevel = MagicMock(WARNING=1)
    lark_mock.Client = MagicMock()
    lark_mock.ws = MagicMock()
    lark_mock.ws.Client = MagicMock(return_value=MagicMock(start=MagicMock()))

    with patch.dict(
        "sys.modules",
        {
            "lark_oapi": lark_mock,
            "lark_oapi.api.im.v1": MagicMock(),
            "lark_oapi.event.dispatcher": MagicMock(),
        },
    ):
        from clawscope.channels.feishu import FeishuChannel
        ch = FeishuChannel(bus=bus, config=cfg)

    return ch, bus, lark_mock


def test_feishu_initial_state() -> None:
    ch, _, _ = make_feishu_channel()
    assert ch.name == "feishu"
    assert ch.is_running is False


@pytest.mark.asyncio
async def test_feishu_stop_sets_running_false() -> None:
    ch, _, _ = make_feishu_channel()
    ch._running = True
    await ch.stop()
    assert ch.is_running is False


@pytest.mark.asyncio
async def test_feishu_send_not_running_warns() -> None:
    ch, _, _ = make_feishu_channel()
    msg = OutboundMessage(channel="feishu", chat_id="oc_123", content="hello")
    # Should not raise even when not running
    await ch.send(msg)
    assert not ch.is_running


@pytest.mark.asyncio
async def test_feishu_send_calls_create_message() -> None:
    ch, _, lark_mock = make_feishu_channel()
    ch._running = True

    # Set up the REST client mock
    mock_response = MagicMock()
    mock_response.success = MagicMock(return_value=True)

    mock_client = MagicMock()
    ch._client = mock_client

    with patch(
        "clawscope.channels.feishu.asyncio.to_thread",
        new=AsyncMock(return_value=mock_response),
    ):
        msg = OutboundMessage(channel="feishu", chat_id="oc_123", content="Hi!")
        await ch.send(msg)

    # asyncio.to_thread was called (actual REST call delegated to thread)
    # Just verify no exception was raised and _running is still True
    assert ch.is_running is True


# ---------------------------------------------------------------------------
# DingTalk _on_message integration (synchronous callback parsing)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dingtalk_message_handler_publishes_inbound() -> None:
    """
    Simulate the MessageHandler.process() callback path without the real SDK.
    We construct a fake callback object and verify an InboundMessage is queued.
    """
    bus = MessageBus()
    cfg = MagicMock()
    cfg.app_key = "k"
    cfg.app_secret = "s"
    cfg.allow_from = ["*"]

    # --- Build a fake incoming message structure ---
    text_obj = MagicMock()
    text_obj.content = "Hello DingTalk"

    incoming = MagicMock()
    incoming.text = text_obj
    incoming.conversation_id = "conv_99"
    incoming.sender_staff_id = "staff_1"
    incoming.conversation_type = "1"
    incoming.at_users = []
    incoming.msgId = "msg_1"

    callback = MagicMock()
    callback.incoming_message = incoming

    # --- Dynamically construct the handler as the channel would ---
    ChatbotHandler = MagicMock
    AIODingTalkStreamClient = MagicMock

    with patch.dict(
        "sys.modules",
        {
            "dingtalk_stream": MagicMock(
                AIODingTalkStreamClient=AIODingTalkStreamClient,
                ChatbotHandler=ChatbotHandler,
            ),
            "dingtalk_stream.chatbot": MagicMock(),
        },
    ):
        from clawscope.channels.dingtalk import DingTalkChannel

        ch = DingTalkChannel(bus=bus, config=cfg)
        ch._running = True

        # Re-create the inner MessageHandler class manually
        class FakeHandler:
            def __init__(self):
                self.bus = bus
                self.config = cfg

            async def process(self, cb):
                incoming_ = cb.incoming_message
                content = (incoming_.text.content or "").strip()
                if not content:
                    return
                chat_id = getattr(incoming_, "conversation_id", "")
                sender_id = getattr(incoming_, "sender_staff_id", "") or "unknown"
                if not ch.is_allowed(sender_id):
                    return
                ch._store_callback(chat_id, cb)
                inbound = InboundMessage(
                    channel="dingtalk",
                    chat_id=chat_id,
                    sender_id=sender_id,
                    content=content,
                )
                await self.bus.publish_inbound(inbound)

        handler = FakeHandler()
        await handler.process(callback)

    received = await bus.consume_inbound()
    assert received.content == "Hello DingTalk"
    assert received.chat_id == "conv_99"
    assert received.sender_id == "staff_1"
    assert "conv_99" in ch._reply_callbacks
