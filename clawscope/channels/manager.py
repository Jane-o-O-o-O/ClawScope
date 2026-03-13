"""Channel manager for ClawScope."""

from __future__ import annotations

import asyncio
from typing import Any, TYPE_CHECKING

from loguru import logger

from clawscope.channels.base import BaseChannel

if TYPE_CHECKING:
    from clawscope.bus import MessageBus, OutboundMessage
    from clawscope.config import ChannelsConfig


class ChannelManager:
    """
    Manages multiple communication channels.

    Features:
    - Auto-discovery of channel modules
    - Dynamic start/stop of channels
    - Outbound message routing
    - Health monitoring
    """

    def __init__(
        self,
        bus: "MessageBus",
        config: "ChannelsConfig",
    ):
        """
        Initialize channel manager.

        Args:
            bus: Message bus instance
            config: Channels configuration
        """
        self.bus = bus
        self.config = config
        self._channels: dict[str, BaseChannel] = {}
        self._running = False
        self._outbound_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start all configured channels."""
        self._running = True
        logger.info("Starting ChannelManager")

        # Start enabled channels
        await self._start_enabled_channels()

        # Start outbound message dispatcher
        self._outbound_task = asyncio.create_task(self._dispatch_outbound())

    async def stop(self) -> None:
        """Stop all channels."""
        self._running = False
        logger.info("Stopping ChannelManager")

        # Cancel outbound dispatcher
        if self._outbound_task:
            self._outbound_task.cancel()
            try:
                await self._outbound_task
            except asyncio.CancelledError:
                pass

        # Stop all channels
        for channel in self._channels.values():
            try:
                await channel.stop()
            except Exception as e:
                logger.error(f"Error stopping channel {channel.name}: {e}")

        self._channels.clear()

    async def _start_enabled_channels(self) -> None:
        """Start all enabled channels based on config."""
        channel_configs = [
            ("telegram", self.config.telegram),
            ("discord", self.config.discord),
            ("slack", self.config.slack),
            ("feishu", self.config.feishu),
            ("dingtalk", self.config.dingtalk),
        ]

        for name, channel_config in channel_configs:
            if channel_config.enabled:
                await self._start_channel(name, channel_config)

    async def _start_channel(self, name: str, config: Any) -> None:
        """Start a single channel."""
        try:
            channel = self._create_channel(name, config)
            if channel:
                await channel.start()
                self._channels[name] = channel
                logger.info(f"Channel '{name}' started")
        except Exception as e:
            logger.error(f"Failed to start channel '{name}': {e}")

    def _create_channel(self, name: str, config: Any) -> BaseChannel | None:
        """Create a channel instance."""
        try:
            if name == "telegram":
                from clawscope.channels.telegram import TelegramChannel
                return TelegramChannel(bus=self.bus, config=config)
            elif name == "discord":
                from clawscope.channels.discord import DiscordChannel
                return DiscordChannel(bus=self.bus, config=config)
            elif name == "slack":
                from clawscope.channels.slack import SlackChannel
                return SlackChannel(bus=self.bus, config=config)
            elif name == "feishu":
                from clawscope.channels.feishu import FeishuChannel
                return FeishuChannel(bus=self.bus, config=config)
            elif name == "dingtalk":
                from clawscope.channels.dingtalk import DingTalkChannel
                return DingTalkChannel(bus=self.bus, config=config)
            else:
                logger.warning(f"Unknown channel type: {name}")
                return None
        except ImportError as e:
            logger.warning(f"Channel '{name}' dependencies not installed: {e}")
            return None

    async def _dispatch_outbound(self) -> None:
        """Dispatch outbound messages to appropriate channels."""
        while self._running:
            try:
                message = await self.bus.consume_outbound()
                channel = self._channels.get(message.channel)

                if channel and channel.is_running:
                    await channel.send(message)
                else:
                    logger.warning(f"No active channel for: {message.channel}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Outbound dispatch error: {e}")

    def get_channel(self, name: str) -> BaseChannel | None:
        """Get a channel by name."""
        return self._channels.get(name)

    def list_channels(self) -> list[str]:
        """List active channel names."""
        return list(self._channels.keys())

    def get_status(self) -> dict[str, bool]:
        """Get status of all channels."""
        return {
            name: channel.is_running
            for name, channel in self._channels.items()
        }


__all__ = ["ChannelManager"]
