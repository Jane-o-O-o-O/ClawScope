"""Telegram channel implementation for ClawScope."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from loguru import logger

from clawscope.channels.base import BaseChannel
from clawscope.bus import InboundMessage

if TYPE_CHECKING:
    from clawscope.bus import MessageBus, OutboundMessage
    from clawscope.config import TelegramConfig


class TelegramChannel(BaseChannel):
    """
    Telegram bot channel implementation.

    Uses python-telegram-bot library for communication.
    """

    def __init__(
        self,
        bus: "MessageBus",
        config: "TelegramConfig",
    ):
        """
        Initialize Telegram channel.

        Args:
            bus: Message bus instance
            config: Telegram configuration
        """
        super().__init__(name="telegram", bus=bus, config=config)
        self._app = None
        self._task = None

    async def start(self) -> None:
        """Start Telegram bot."""
        if not self.config.bot_token:
            raise ValueError("Telegram bot_token not configured")

        try:
            from telegram import Update
            from telegram.ext import Application, MessageHandler, filters

            # Create application
            self._app = (
                Application.builder()
                .token(self.config.bot_token)
                .build()
            )

            # Add message handler
            self._app.add_handler(
                MessageHandler(
                    filters.TEXT & ~filters.COMMAND,
                    self._handle_message,
                )
            )

            # Start polling
            await self._app.initialize()
            await self._app.start()
            self._task = asyncio.create_task(self._app.updater.start_polling())

            self._running = True
            logger.info("Telegram channel started")

        except ImportError:
            raise ImportError("python-telegram-bot not installed. Install with: pip install python-telegram-bot")

    async def stop(self) -> None:
        """Stop Telegram bot."""
        self._running = False

        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()

        if self._task:
            self._task.cancel()

        logger.info("Telegram channel stopped")

    async def send(self, message: "OutboundMessage") -> None:
        """Send message to Telegram chat."""
        if not self._app or not self._running:
            logger.warning("Telegram channel not running")
            return

        try:
            # Send text message
            await self._app.bot.send_message(
                chat_id=message.chat_id,
                text=message.content,
                reply_to_message_id=message.reply_to if message.reply_to else None,
            )

            # Send media if any
            for media_url in message.media:
                if any(ext in media_url.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif']):
                    await self._app.bot.send_photo(
                        chat_id=message.chat_id,
                        photo=media_url,
                    )
                elif any(ext in media_url.lower() for ext in ['.mp3', '.wav', '.ogg']):
                    await self._app.bot.send_audio(
                        chat_id=message.chat_id,
                        audio=media_url,
                    )
                else:
                    await self._app.bot.send_document(
                        chat_id=message.chat_id,
                        document=media_url,
                    )

        except Exception as e:
            logger.error(f"Telegram send error: {e}")

    async def _handle_message(self, update, context) -> None:
        """Handle incoming Telegram message."""
        try:
            message = update.message
            if not message or not message.text:
                return

            sender_id = str(message.from_user.id)
            chat_id = str(message.chat_id)

            # Check if sender is allowed
            if not self.is_allowed(sender_id):
                logger.debug(f"Telegram: Ignored message from {sender_id}")
                return

            # Extract media
            media = []
            if message.photo:
                photo = message.photo[-1]  # Largest size
                file = await context.bot.get_file(photo.file_id)
                media.append(file.file_path)

            # Publish to bus
            inbound = InboundMessage(
                channel="telegram",
                sender_id=sender_id,
                chat_id=chat_id,
                content=message.text,
                media=media,
                metadata={
                    "message_id": message.message_id,
                    "username": message.from_user.username,
                    "first_name": message.from_user.first_name,
                },
            )
            await self.bus.publish_inbound(inbound)

        except Exception as e:
            logger.error(f"Telegram message handling error: {e}")


__all__ = ["TelegramChannel"]
