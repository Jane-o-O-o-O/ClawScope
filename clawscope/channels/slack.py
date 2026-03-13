"""Slack channel implementation for ClawScope."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from loguru import logger

from clawscope.channels.base import BaseChannel
from clawscope.bus import InboundMessage

if TYPE_CHECKING:
    from clawscope.bus import MessageBus, OutboundMessage
    from clawscope.config import SlackConfig


class SlackChannel(BaseChannel):
    """
    Slack bot channel implementation.

    Uses Slack SDK with Socket Mode for real-time communication.
    """

    def __init__(
        self,
        bus: "MessageBus",
        config: "SlackConfig",
    ):
        """
        Initialize Slack channel.

        Args:
            bus: Message bus instance
            config: Slack configuration
        """
        super().__init__(name="slack", bus=bus, config=config)
        self._client = None
        self._socket = None
        self._task = None

    async def start(self) -> None:
        """Start Slack bot."""
        if not self.config.bot_token or not self.config.app_token:
            raise ValueError("Slack bot_token and app_token required")

        try:
            from slack_sdk.web.async_client import AsyncWebClient
            from slack_sdk.socket_mode.aiohttp import SocketModeClient
            from slack_sdk.socket_mode.request import SocketModeRequest
            from slack_sdk.socket_mode.response import SocketModeResponse

            self._client = AsyncWebClient(token=self.config.bot_token)
            self._socket = SocketModeClient(
                app_token=self.config.app_token,
                web_client=self._client,
            )

            async def handle_event(client, req: SocketModeRequest):
                if req.type == "events_api":
                    await client.send_socket_mode_response(
                        SocketModeResponse(envelope_id=req.envelope_id)
                    )

                    event = req.payload.get("event", {})
                    if event.get("type") == "message" and "subtype" not in event:
                        await self._handle_message(event)

            self._socket.socket_mode_request_listeners.append(handle_event)

            # Start socket mode
            self._task = asyncio.create_task(self._socket.connect())
            self._running = True

            logger.info("Slack channel started")

        except ImportError:
            raise ImportError("slack-sdk not installed. Install with: pip install slack-sdk")

    async def stop(self) -> None:
        """Stop Slack bot."""
        self._running = False

        if self._socket:
            await self._socket.close()

        if self._task:
            self._task.cancel()

        logger.info("Slack channel stopped")

    async def send(self, message: "OutboundMessage") -> None:
        """Send message to Slack channel."""
        if not self._client or not self._running:
            logger.warning("Slack channel not running")
            return

        try:
            await self._client.chat_postMessage(
                channel=message.chat_id,
                text=message.content,
                thread_ts=message.reply_to,
            )

            # Send media as attachments
            for media_url in message.media:
                await self._client.chat_postMessage(
                    channel=message.chat_id,
                    text=media_url,
                    thread_ts=message.reply_to,
                )

        except Exception as e:
            logger.error(f"Slack send error: {e}")

    async def _handle_message(self, event: dict) -> None:
        """Handle incoming Slack message."""
        try:
            sender_id = event.get("user", "")
            chat_id = event.get("channel", "")
            text = event.get("text", "")

            if not sender_id or not text:
                return

            # Check if sender is allowed
            if not self.is_allowed(sender_id):
                return

            # Extract files
            media = []
            for file in event.get("files", []):
                if "url_private" in file:
                    media.append(file["url_private"])

            # Publish to bus
            inbound = InboundMessage(
                channel="slack",
                sender_id=sender_id,
                chat_id=chat_id,
                content=text,
                media=media,
                metadata={
                    "ts": event.get("ts"),
                    "thread_ts": event.get("thread_ts"),
                },
            )
            await self.bus.publish_inbound(inbound)

        except Exception as e:
            logger.error(f"Slack message handling error: {e}")


__all__ = ["SlackChannel"]
