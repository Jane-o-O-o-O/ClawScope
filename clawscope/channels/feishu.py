"""Feishu (Lark) channel implementation for ClawScope."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from clawscope.channels.base import BaseChannel
from clawscope.bus import InboundMessage

if TYPE_CHECKING:
    from clawscope.bus import MessageBus, OutboundMessage
    from clawscope.config import FeishuConfig


class FeishuChannel(BaseChannel):
    """
    Feishu (Lark) bot channel implementation.

    Uses lark-oapi library for communication.
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

    async def start(self) -> None:
        """Start Feishu bot."""
        if not self.config.app_id or not self.config.app_secret:
            raise ValueError("Feishu app_id and app_secret required")

        try:
            import lark_oapi as lark

            self._client = lark.Client.builder() \
                .app_id(self.config.app_id) \
                .app_secret(self.config.app_secret) \
                .build()

            self._running = True
            logger.info("Feishu channel started (webhook mode)")

        except ImportError:
            raise ImportError("lark-oapi not installed. Install with: pip install lark-oapi")

    async def stop(self) -> None:
        """Stop Feishu bot."""
        self._running = False
        logger.info("Feishu channel stopped")

    async def send(self, message: "OutboundMessage") -> None:
        """Send message to Feishu chat."""
        if not self._client or not self._running:
            logger.warning("Feishu channel not running")
            return

        try:
            import lark_oapi as lark
            from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody

            # Build request
            request = CreateMessageRequest.builder() \
                .receive_id_type("chat_id") \
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(message.chat_id)
                    .msg_type("text")
                    .content(f'{{"text": "{message.content}"}}')
                    .build()
                ) \
                .build()

            # Send
            response = self._client.im.v1.message.create(request)
            if not response.success():
                logger.error(f"Feishu send error: {response.msg}")

        except Exception as e:
            logger.error(f"Feishu send error: {e}")


__all__ = ["FeishuChannel"]
