"""DingTalk channel implementation for ClawScope."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from clawscope.channels.base import BaseChannel
from clawscope.bus import InboundMessage

if TYPE_CHECKING:
    from clawscope.bus import MessageBus, OutboundMessage
    from clawscope.config import DingTalkConfig


class DingTalkChannel(BaseChannel):
    """
    DingTalk bot channel implementation.

    Uses dingtalk-stream library for WebSocket communication.
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

    async def start(self) -> None:
        """Start DingTalk bot."""
        if not self.config.app_key or not self.config.app_secret:
            raise ValueError("DingTalk app_key and app_secret required")

        try:
            from dingtalk_stream import AIODingTalkStreamClient, ChatbotHandler

            class MessageHandler(ChatbotHandler):
                def __init__(self, bus, config):
                    super().__init__()
                    self.bus = bus
                    self.config = config

                async def process(self, callback):
                    # Handle message
                    pass

            self._client = AIODingTalkStreamClient(
                credential=dict(
                    client_id=self.config.app_key,
                    client_secret=self.config.app_secret,
                ),
            )

            self._running = True
            logger.info("DingTalk channel started")

        except ImportError:
            raise ImportError("dingtalk-stream not installed. Install with: pip install dingtalk-stream")

    async def stop(self) -> None:
        """Stop DingTalk bot."""
        self._running = False
        logger.info("DingTalk channel stopped")

    async def send(self, message: "OutboundMessage") -> None:
        """Send message to DingTalk chat."""
        if not self._client or not self._running:
            logger.warning("DingTalk channel not running")
            return

        try:
            # TODO: Implement DingTalk send
            logger.debug(f"DingTalk send: {message.content[:50]}")
        except Exception as e:
            logger.error(f"DingTalk send error: {e}")


__all__ = ["DingTalkChannel"]
