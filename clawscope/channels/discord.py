"""Discord channel implementation for ClawScope."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from loguru import logger

from clawscope.channels.base import BaseChannel
from clawscope.bus import InboundMessage

if TYPE_CHECKING:
    from clawscope.bus import MessageBus, OutboundMessage
    from clawscope.config import DiscordConfig


class DiscordChannel(BaseChannel):
    """
    Discord bot channel implementation.

    Uses discord.py library for communication.
    """

    def __init__(
        self,
        bus: "MessageBus",
        config: "DiscordConfig",
    ):
        """
        Initialize Discord channel.

        Args:
            bus: Message bus instance
            config: Discord configuration
        """
        super().__init__(name="discord", bus=bus, config=config)
        self._client = None
        self._task = None

    async def start(self) -> None:
        """Start Discord bot."""
        if not self.config.bot_token:
            raise ValueError("Discord bot_token not configured")

        try:
            import discord

            intents = discord.Intents.default()
            intents.message_content = True

            self._client = discord.Client(intents=intents)

            @self._client.event
            async def on_ready():
                logger.info(f"Discord bot logged in as {self._client.user}")
                self._running = True

            @self._client.event
            async def on_message(message):
                await self._handle_message(message)

            # Start in background
            self._task = asyncio.create_task(
                self._client.start(self.config.bot_token)
            )

            # Wait for ready
            await asyncio.sleep(2)

        except ImportError:
            raise ImportError("discord.py not installed. Install with: pip install discord.py")

    async def stop(self) -> None:
        """Stop Discord bot."""
        self._running = False

        if self._client:
            await self._client.close()

        if self._task:
            self._task.cancel()

        logger.info("Discord channel stopped")

    async def send(self, message: "OutboundMessage") -> None:
        """Send message to Discord channel."""
        if not self._client or not self._running:
            logger.warning("Discord channel not running")
            return

        try:
            channel = self._client.get_channel(int(message.chat_id))
            if not channel:
                logger.warning(f"Discord channel not found: {message.chat_id}")
                return

            # Send text
            await channel.send(message.content)

            # Send media
            import discord
            for media_url in message.media:
                await channel.send(media_url)

        except Exception as e:
            logger.error(f"Discord send error: {e}")

    async def _handle_message(self, message) -> None:
        """Handle incoming Discord message."""
        try:
            # Ignore bot's own messages
            if message.author == self._client.user:
                return

            # Ignore messages without content
            if not message.content:
                return

            sender_id = str(message.author.id)
            chat_id = str(message.channel.id)

            # Check if sender is allowed
            if not self.is_allowed(sender_id):
                return

            # Extract media
            media = [att.url for att in message.attachments]

            # Publish to bus
            inbound = InboundMessage(
                channel="discord",
                sender_id=sender_id,
                chat_id=chat_id,
                content=message.content,
                media=media,
                metadata={
                    "message_id": message.id,
                    "username": message.author.name,
                    "guild_id": str(message.guild.id) if message.guild else None,
                },
            )
            await self.bus.publish_inbound(inbound)

        except Exception as e:
            logger.error(f"Discord message handling error: {e}")


__all__ = ["DiscordChannel"]
