"""DingTalk channel implementation for ClawScope."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import httpx
from loguru import logger

from clawscope.bus import InboundMessage
from clawscope.channels.base import BaseChannel

if TYPE_CHECKING:
    from clawscope.bus import MessageBus, OutboundMessage
    from clawscope.config import DingTalkConfig


_MAX_CALLBACKS = 256


class DingTalkChannel(BaseChannel):
    """
    DingTalk bot channel implementation.

    Uses dingtalk-stream library for WebSocket communication.
    """

    def __init__(
        self,
        bus: MessageBus,
        config: DingTalkConfig,
    ):
        """
        Initialize DingTalk channel.

        Args:
            bus: Message bus instance
            config: DingTalk configuration
        """
        super().__init__(name="dingtalk", bus=bus, config=config)
        self._client = None
        self._handler = None
        self._task: asyncio.Task[Any] | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._session_webhooks: dict[str, str] = {}
        self._reply_callbacks: OrderedDict[str, Any] = OrderedDict()

    def _store_callback(self, chat_id: str, callback: Any) -> None:
        """Store the latest reactive reply callback for a conversation."""
        if chat_id in self._reply_callbacks:
            self._reply_callbacks.move_to_end(chat_id)
        self._reply_callbacks[chat_id] = callback
        while len(self._reply_callbacks) > _MAX_CALLBACKS:
            self._reply_callbacks.popitem(last=False)

    async def start(self) -> None:
        """Start DingTalk bot."""
        if not self.config.app_key or not self.config.app_secret:
            raise ValueError("DingTalk app_key and app_secret required")

        try:
            from dingtalk_stream import AIODingTalkStreamClient, ChatbotHandler

            ack_message_cls = None
            chatbot_message_cls = None
            try:
                from dingtalk_stream import AckMessage as ImportedAckMessage

                ack_message_cls = ImportedAckMessage
            except ImportError:
                ack_message_cls = None

            try:
                from dingtalk_stream import ChatbotMessage as ImportedChatbotMessage

                chatbot_message_cls = ImportedChatbotMessage
            except ImportError:
                chatbot_message_cls = None

            class MessageHandler(ChatbotHandler):
                def __init__(self, channel: DingTalkChannel):
                    super().__init__()
                    self.channel = channel

                async def process(self, callback: Any) -> Any:
                    await self.channel._handle_callback(callback)
                    return self.channel._build_ack_message(ack_message_cls)

            self._client = AIODingTalkStreamClient(
                credential={
                    "client_id": self.config.app_key,
                    "client_secret": self.config.app_secret,
                },
            )
            self._handler = MessageHandler(self)

            if hasattr(self._client, "register_callback_handler") and self._handler:
                topic = (
                    getattr(chatbot_message_cls, "TOPIC", None)
                    or getattr(self._handler, "TOPIC", None)
                )
                if topic:
                    self._client.register_callback_handler(topic, self._handler)

            if hasattr(self._client, "start_forever"):
                self._task = asyncio.create_task(self._client.start_forever())

            self._running = True
            logger.info("DingTalk channel started")

        except ImportError as exc:
            raise ImportError(
                "dingtalk-stream not installed. Install with: pip install dingtalk-stream"
            ) from exc

    async def stop(self) -> None:
        """Stop DingTalk bot."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        self._reply_callbacks.clear()

        logger.info("DingTalk channel stopped")

    async def send(self, message: OutboundMessage) -> None:
        """Send message to DingTalk chat."""
        if not self._running:
            logger.warning("DingTalk channel not running")
            return

        try:
            webhook = (
                message.metadata.get("session_webhook")
                or self._session_webhooks.get(message.chat_id)
            )
            if not webhook:
                callback = self._reply_callbacks.get(message.chat_id)
                if callback is not None:
                    await self._reply_with_callback(callback, message.content)
                    return
                logger.warning(f"DingTalk session webhook not found for chat: {message.chat_id}")
                return

            client = self._http_client
            close_after_send = False
            if client is None:
                client = httpx.AsyncClient(timeout=10.0)
                close_after_send = True

            try:
                await self._post_text_message(client, webhook, message.content)
                for media_url in message.media:
                    await self._post_text_message(client, webhook, media_url)
            finally:
                if close_after_send:
                    await client.aclose()
        except Exception as e:
            logger.error(f"DingTalk send error: {e}")

    async def _handle_callback(self, callback: Any) -> None:
        """Translate DingTalk callback payloads into inbound bus messages."""
        payload = getattr(callback, "data", None)
        if not isinstance(payload, dict):
            logger.debug("Ignoring DingTalk callback without structured payload")
            return

        sender_id = self._string_value(
            payload.get("senderStaffId"),
            payload.get("staffId"),
            payload.get("senderId"),
            payload.get("conversationStaffId"),
        )
        chat_id = self._string_value(
            payload.get("conversationId"),
            payload.get("chatId"),
            payload.get("openConversationId"),
            payload.get("sessionWebhook"),
        )
        content = self._extract_text(payload)

        if not sender_id or not chat_id or not content:
            return

        if not self.is_allowed(sender_id):
            return

        session_webhook = payload.get("sessionWebhook")
        if isinstance(session_webhook, str) and session_webhook:
            self._session_webhooks[chat_id] = session_webhook
        self._store_callback(chat_id, callback)

        inbound = InboundMessage(
            channel="dingtalk",
            sender_id=sender_id,
            chat_id=chat_id,
            content=content,
            media=self._extract_media(payload),
            metadata={
                "conversation_type": payload.get("conversationType"),
                "msg_id": payload.get("msgId") or payload.get("messageId"),
                "msgtype": payload.get("msgtype"),
                "sender_nick": payload.get("senderNick"),
                "session_webhook": session_webhook,
            },
        )
        await self.bus.publish_inbound(inbound)

    def _build_ack_message(self, ack_message_cls: type[Any] | None) -> Any:
        """Create a successful DingTalk ack response when supported."""
        if ack_message_cls is None:
            return None

        status_ok = getattr(ack_message_cls, "STATUS_OK", None)
        if status_ok is None:
            return None

        try:
            return ack_message_cls(status_ok, "OK")
        except TypeError:
            return None

    async def _post_text_message(
        self,
        client: httpx.AsyncClient,
        webhook: str,
        content: str,
    ) -> None:
        """Send a text payload through a DingTalk session webhook."""
        response = await client.post(
            webhook,
            json={
                "msgtype": "text",
                "text": {"content": content},
            },
        )
        response.raise_for_status()

        body = response.json()
        if isinstance(body, dict) and body.get("errcode", 0) not in (0, "0", None):
            raise ValueError(f"DingTalk send failed: {body}")

    async def _reply_with_callback(self, callback: Any, content: str) -> None:
        """Send a reactive reply through the SDK callback when available."""
        try:
            from dingtalk_stream.chatbot import TextMessage

            text_message = TextMessage(content=content)
        except Exception:
            text_message = content

        reply = getattr(callback, "reply", None)
        if not callable(reply):
            raise ValueError("DingTalk callback does not support replies")

        result = reply([text_message])
        if hasattr(result, "__await__"):
            await result

    def _extract_text(self, payload: dict[str, Any]) -> str:
        """Extract text content from DingTalk callback payload."""
        text_payload = payload.get("text")
        if isinstance(text_payload, dict):
            content = text_payload.get("content")
            if isinstance(content, str):
                return content.strip()

        content_payload = payload.get("content")
        if isinstance(content_payload, str):
            return content_payload.strip()
        if isinstance(content_payload, dict):
            text_content = content_payload.get("text") or content_payload.get("content")
            if isinstance(text_content, str):
                return text_content.strip()

        return ""

    def _extract_media(self, payload: dict[str, Any]) -> list[str]:
        """Extract media URLs or download codes when available."""
        media: list[str] = []

        for key in ("downloadCode", "mediaId", "picUrl"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                media.append(value)

        return media

    def _string_value(self, *values: Any) -> str:
        """Return the first non-empty string-like value."""
        for value in values:
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""


__all__ = ["DingTalkChannel"]
