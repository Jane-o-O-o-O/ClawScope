"""Feishu (Lark) channel implementation for ClawScope."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from loguru import logger

from clawscope.channels.base import BaseChannel
from clawscope.bus import InboundMessage

if TYPE_CHECKING:
    from clawscope.bus import MessageBus, OutboundMessage
    from clawscope.config import FeishuConfig


class FeishuChannel(BaseChannel):
    """
    Feishu (Lark) bot channel implementation.

    Uses lark-oapi for both sending (REST) and receiving (WebSocket long-polling).

    Inbound flow (WebSocket mode):
      Feishu cloud → WebSocket (lark.ws.Client) → _on_message()
      → InboundMessage → MessageBus

    Outbound flow:
      MessageBus → OutboundMessage → send()
      → lark IM v1 CreateMessage REST API

    Webhook mode (alternative):
      Call ``get_event_handler()`` and mount it on a FastAPI/Flask app so that
      Feishu can POST events to your server.
    """

    def __init__(
        self,
        bus: "MessageBus",
        config: "FeishuConfig",
    ):
        """
        Initialize Feishu channel.

        Args:
            bus: Message bus instance
            config: Feishu configuration
        """
        super().__init__(name="feishu", bus=bus, config=config)
        self._client = None
        self._ws_client = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start Feishu bot using WebSocket long-polling."""
        if not self.config.app_id or not self.config.app_secret:
            raise ValueError("Feishu app_id and app_secret required")

        try:
            import lark_oapi as lark

            # REST client for sending messages
            self._client = (
                lark.Client.builder()
                .app_id(self.config.app_id)
                .app_secret(self.config.app_secret)
                .build()
            )

            # WebSocket client for receiving events
            event_handler = self._build_event_handler()
            self._ws_client = lark.ws.Client(
                self.config.app_id,
                self.config.app_secret,
                event_handler=event_handler,
                log_level=lark.LogLevel.WARNING,
            )

            # Run WebSocket client in background task
            asyncio.create_task(self._run_ws_client())

            self._running = True
            logger.info("Feishu channel started (WebSocket mode)")

        except ImportError:
            raise ImportError(
                "lark-oapi not installed. "
                "Install with: pip install clawscope[feishu]"
            )

    async def _run_ws_client(self) -> None:
        """Run the WebSocket client; reconnect on transient errors."""
        while self._running:
            try:
                await asyncio.to_thread(self._ws_client.start)
            except Exception as exc:
                if not self._running:
                    break
                logger.warning(
                    f"Feishu WebSocket disconnected ({exc}), reconnecting in 5 s"
                )
                await asyncio.sleep(5)

    async def stop(self) -> None:
        """Stop Feishu bot."""
        self._running = False
        logger.info("Feishu channel stopped")

    # ------------------------------------------------------------------
    # Event handler
    # ------------------------------------------------------------------

    def _build_event_handler(self) -> Any:
        """Build a lark-oapi EventDispatcherHandler that forwards messages to the bus."""
        try:
            import lark_oapi as lark
            from lark_oapi.api.im.v1 import P2ImMessageReceiveV1  # type: ignore[import]
            from lark_oapi.event.dispatcher import EventDispatcherHandler  # type: ignore[import]
        except ImportError:
            raise ImportError("lark-oapi not installed")

        channel_ref = self

        def on_message(data: "P2ImMessageReceiveV1") -> None:
            """Synchronous callback invoked by the lark SDK on each new message."""
            try:
                event = data.event
                msg = event.message
                sender = event.sender

                # Only handle real user messages (ignore bot echoes)
                if getattr(sender, "sender_type", "") == "app":
                    return

                chat_id: str = getattr(msg, "chat_id", "") or ""
                sender_id: str = (
                    getattr(sender.sender_id, "user_id", "")
                    if hasattr(sender, "sender_id")
                    else ""
                ) or "unknown"

                if not channel_ref.is_allowed(sender_id):
                    logger.debug(
                        f"Feishu: message from {sender_id} rejected by allow-list"
                    )
                    return

                # Decode text content from JSON payload
                content: str = ""
                msg_type: str = getattr(msg, "message_type", "")
                raw_content: str = getattr(msg, "content", "") or ""
                if msg_type == "text":
                    try:
                        content = json.loads(raw_content).get("text", "").strip()
                    except (json.JSONDecodeError, AttributeError):
                        content = raw_content.strip()
                else:
                    # Non-text message types (images, cards, …) – keep raw payload
                    content = raw_content

                if not content:
                    return

                inbound = InboundMessage(
                    channel="feishu",
                    chat_id=chat_id,
                    sender_id=sender_id,
                    content=content,
                    metadata={
                        "message_type": msg_type,
                        "message_id": getattr(msg, "message_id", ""),
                        "chat_type": getattr(msg, "chat_type", ""),
                    },
                )

                # Schedule the coroutine on the running event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        channel_ref.bus.publish_inbound(inbound), loop
                    )
                else:
                    loop.run_until_complete(channel_ref.bus.publish_inbound(inbound))

                logger.debug(
                    f"Feishu inbound from {sender_id} in {chat_id}: {content[:60]!r}"
                )

            except Exception as exc:
                logger.error(f"Feishu on_message error: {exc}")

        handler = (
            EventDispatcherHandler.builder("", "")
            .register(P2ImMessageReceiveV1, on_message)
            .build()
        )
        return handler

    def get_event_handler(self) -> Any:
        """
        Return a lark-oapi EventDispatcherHandler for webhook mode.

        Mount the returned handler on your FastAPI/Flask app so that Feishu
        can POST events to your server.  Example (FastAPI)::

            from fastapi import FastAPI, Request
            from lark_oapi.adapter.fastapi import to_fastapi

            app = FastAPI()
            feishu_channel = FeishuChannel(bus, config)

            @app.post("/feishu/event")
            async def feishu_event(req: Request):
                return await to_fastapi(feishu_channel.get_event_handler(), req)
        """
        return self._build_event_handler()

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    async def send(self, message: "OutboundMessage") -> None:
        """Send a text message to a Feishu chat via the IM v1 REST API."""
        if not self._client or not self._running:
            logger.warning("Feishu channel not running")
            return

        try:
            import lark_oapi as lark
            from lark_oapi.api.im.v1 import (  # type: ignore[import]
                CreateMessageRequest,
                CreateMessageRequestBody,
            )

            body = (
                CreateMessageRequestBody.builder()
                .receive_id(message.chat_id)
                .msg_type("text")
                .content(json.dumps({"text": message.content}))
                .build()
            )

            request = (
                CreateMessageRequest.builder()
                .receive_id_type("chat_id")
                .request_body(body)
                .build()
            )

            # lark-oapi REST calls are synchronous; run them off the event loop
            response = await asyncio.to_thread(
                self._client.im.v1.message.create, request
            )

            if not response.success():
                logger.error(
                    f"Feishu send error (chat_id={message.chat_id!r}): "
                    f"{response.msg} (code={response.code})"
                )
            else:
                logger.debug(
                    f"Feishu sent to {message.chat_id}: {message.content[:60]!r}"
                )

        except Exception as exc:
            logger.error(f"Feishu send error: {exc}")


__all__ = ["FeishuChannel"]
