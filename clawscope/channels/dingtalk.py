"""DingTalk channel implementation for ClawScope."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import TYPE_CHECKING

from loguru import logger

from clawscope.channels.base import BaseChannel
from clawscope.bus import InboundMessage

if TYPE_CHECKING:
    from clawscope.bus import MessageBus, OutboundMessage
    from clawscope.config import DingTalkConfig

# Maximum number of reply callbacks kept in memory per channel instance.
# Older entries are evicted when this limit is exceeded (LRU-like).
_MAX_CALLBACKS = 500


class DingTalkChannel(BaseChannel):
    """
    DingTalk bot channel implementation.

    Uses dingtalk-stream library for WebSocket communication.

    Inbound flow:
      DingTalk cloud → WebSocket (dingtalk-stream) → MessageHandler.process()
      → InboundMessage → MessageBus

    Outbound flow:
      MessageBus → OutboundMessage → send()
      → callback.reply() for reactive messages (same conversation)
    """

    def __init__(
        self,
        bus: "MessageBus",
        config: "DingTalkConfig",
    ):
        """
        Initialize DingTalk channel.

        Args:
            bus: Message bus instance
            config: DingTalk configuration
        """
        super().__init__(name="dingtalk", bus=bus, config=config)
        self._client = None
        # Maps conversation_id → most-recent callback object so that
        # send() can reply back without extra REST calls.
        self._reply_callbacks: OrderedDict = OrderedDict()

    def _store_callback(self, chat_id: str, callback) -> None:
        """Store (or refresh) a reply callback, evicting the oldest if needed."""
        if chat_id in self._reply_callbacks:
            self._reply_callbacks.move_to_end(chat_id)
        self._reply_callbacks[chat_id] = callback
        while len(self._reply_callbacks) > _MAX_CALLBACKS:
            self._reply_callbacks.popitem(last=False)

    async def start(self) -> None:
        """Start DingTalk bot via WebSocket stream."""
        if not self.config.app_key or not self.config.app_secret:
            raise ValueError("DingTalk app_key and app_secret required")

        try:
            from dingtalk_stream import AIODingTalkStreamClient, ChatbotHandler

            channel_ref = self  # captured in nested class below

            class MessageHandler(ChatbotHandler):
                """Translates DingTalk messages into ClawScope InboundMessages."""

                def __init__(self, bus, config):
                    super().__init__()
                    self.bus = bus
                    self.config = config

                async def process(self, callback) -> None:
                    """Handle an incoming chatbot message."""
                    try:
                        incoming = callback.incoming_message

                        # Extract text content
                        content: str = ""
                        if hasattr(incoming, "text") and incoming.text:
                            content = (incoming.text.content or "").strip()
                        if not content:
                            logger.debug("DingTalk: received empty message, skipping")
                            return

                        chat_id: str = getattr(incoming, "conversation_id", "") or ""
                        sender_id: str = (
                            getattr(incoming, "sender_staff_id", "")
                            or getattr(incoming, "sender_id", "")
                            or "unknown"
                        )

                        # Check allow-list
                        if not channel_ref.is_allowed(sender_id):
                            logger.debug(
                                f"DingTalk: message from {sender_id} rejected by allow-list"
                            )
                            return

                        # Remember the callback so send() can reply later
                        channel_ref._store_callback(chat_id, callback)

                        inbound = InboundMessage(
                            channel="dingtalk",
                            chat_id=chat_id,
                            sender_id=sender_id,
                            content=content,
                            metadata={
                                "conversation_type": getattr(
                                    incoming, "conversation_type", ""
                                ),
                                "at_users": getattr(incoming, "at_users", []),
                                "msg_id": getattr(incoming, "msgId", ""),
                            },
                        )
                        await self.bus.publish_inbound(inbound)
                        logger.debug(
                            f"DingTalk inbound from {sender_id} in {chat_id}: "
                            f"{content[:60]!r}"
                        )

                    except Exception as exc:
                        logger.error(f"DingTalk MessageHandler.process error: {exc}")

            handler = MessageHandler(self.bus, self.config)

            self._client = AIODingTalkStreamClient(
                credential=dict(
                    client_id=self.config.app_key,
                    client_secret=self.config.app_secret,
                ),
                chatbot_handler=handler,
            )

            # Start the WebSocket client in the background
            asyncio.create_task(self._client.start_forever())

            self._running = True
            logger.info("DingTalk channel started")

        except ImportError:
            raise ImportError(
                "dingtalk-stream not installed. "
                "Install with: pip install clawscope[dingtalk]"
            )

    async def stop(self) -> None:
        """Stop DingTalk bot."""
        self._running = False
        if self._client and hasattr(self._client, "stop"):
            try:
                await self._client.stop()
            except Exception as exc:
                logger.warning(f"DingTalk client stop error: {exc}")
        self._reply_callbacks.clear()
        logger.info("DingTalk channel stopped")

    async def send(self, message: "OutboundMessage") -> None:
        """
        Send a message to a DingTalk conversation.

        Uses the stored reply callback for the target chat_id when available
        (reactive pattern).  Logs a warning when no prior interaction exists
        for that chat_id (proactive pattern not yet supported).
        """
        if not self._running:
            logger.warning("DingTalk: channel is not running, message dropped")
            return

        callback = self._reply_callbacks.get(message.chat_id)
        if callback is None:
            logger.warning(
                f"DingTalk: no reply callback for chat_id={message.chat_id!r}; "
                "proactive messaging requires the DingTalk Open API and is not "
                "implemented.  Message dropped."
            )
            return

        try:
            from dingtalk_stream.chatbot import TextMessage  # type: ignore[import]

            text_msg = TextMessage(content=message.content)
            await callback.reply([text_msg])
            logger.debug(
                f"DingTalk sent to {message.chat_id}: {message.content[:60]!r}"
            )

        except Exception as exc:
            logger.error(f"DingTalk send error for chat_id={message.chat_id!r}: {exc}")


__all__ = ["DingTalkChannel"]
