"""Message bus implementation for ClawScope."""

from __future__ import annotations

import asyncio
from typing import Callable, Awaitable
from collections import defaultdict

from loguru import logger

from clawscope.bus.events import InboundMessage, OutboundMessage, SystemEvent


class MessageBus:
    """
    Async message bus for channel-agent communication.

    Provides decoupled pub/sub communication between:
    - Channels (publishers of inbound, subscribers of outbound)
    - Agent router (subscriber of inbound, publisher of outbound)

    Features:
    - Non-blocking async queues
    - Channel-specific filtering
    - Event listeners for monitoring
    """

    def __init__(self, max_queue_size: int = 1000):
        """
        Initialize message bus.

        Args:
            max_queue_size: Maximum queue size before blocking
        """
        self._inbound: asyncio.Queue[InboundMessage] = asyncio.Queue(maxsize=max_queue_size)
        self._outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue(maxsize=max_queue_size)
        self._system: asyncio.Queue[SystemEvent] = asyncio.Queue(maxsize=100)

        # Channel-specific outbound queues for targeted delivery
        self._channel_queues: dict[str, asyncio.Queue[OutboundMessage]] = defaultdict(
            lambda: asyncio.Queue(maxsize=max_queue_size)
        )

        # Event listeners
        self._inbound_listeners: list[Callable[[InboundMessage], Awaitable[None]]] = []
        self._outbound_listeners: list[Callable[[OutboundMessage], Awaitable[None]]] = []

        # Statistics
        self._stats = {
            "inbound_count": 0,
            "outbound_count": 0,
            "errors": 0,
        }

    # ========== Inbound (Channel → Agent) ==========

    async def publish_inbound(self, message: InboundMessage) -> None:
        """
        Publish an inbound message from a channel.

        Args:
            message: Inbound message to publish
        """
        await self._inbound.put(message)
        self._stats["inbound_count"] += 1
        logger.debug(f"Inbound message from {message.channel}:{message.chat_id}")

        # Notify listeners
        for listener in self._inbound_listeners:
            try:
                await listener(message)
            except Exception as e:
                logger.error(f"Inbound listener error: {e}")
                self._stats["errors"] += 1

    async def consume_inbound(self) -> InboundMessage:
        """
        Consume next inbound message.

        Returns:
            Next inbound message (blocks until available)
        """
        return await self._inbound.get()

    async def consume_inbound_for(
        self,
        channel: str,
        chat_id: str,
        timeout: float | None = None,
    ) -> InboundMessage | None:
        """
        Consume inbound message for specific channel and chat.

        Args:
            channel: Target channel
            chat_id: Target chat ID
            timeout: Optional timeout in seconds

        Returns:
            Matching inbound message or None if timeout
        """
        # This is a simplified implementation
        # In production, we'd want a more efficient filtering mechanism
        try:
            while True:
                if timeout:
                    msg = await asyncio.wait_for(self._inbound.get(), timeout=timeout)
                else:
                    msg = await self._inbound.get()

                if msg.channel == channel and msg.chat_id == chat_id:
                    return msg

                # Put back non-matching messages
                await self._inbound.put(msg)
                await asyncio.sleep(0.01)  # Prevent tight loop

        except asyncio.TimeoutError:
            return None

    def on_inbound(
        self, listener: Callable[[InboundMessage], Awaitable[None]]
    ) -> Callable[[], None]:
        """
        Register inbound message listener.

        Args:
            listener: Async callback for inbound messages

        Returns:
            Unsubscribe function
        """
        self._inbound_listeners.append(listener)
        return lambda: self._inbound_listeners.remove(listener)

    # ========== Outbound (Agent → Channel) ==========

    async def publish_outbound(self, message: OutboundMessage) -> None:
        """
        Publish an outbound message to channels.

        Args:
            message: Outbound message to publish
        """
        # Put in global queue
        await self._outbound.put(message)

        # Put in channel-specific queue
        await self._channel_queues[message.channel].put(message)

        self._stats["outbound_count"] += 1
        logger.debug(f"Outbound message to {message.channel}:{message.chat_id}")

        # Notify listeners
        for listener in self._outbound_listeners:
            try:
                await listener(message)
            except Exception as e:
                logger.error(f"Outbound listener error: {e}")
                self._stats["errors"] += 1

    async def consume_outbound(self) -> OutboundMessage:
        """
        Consume next outbound message (any channel).

        Returns:
            Next outbound message (blocks until available)
        """
        return await self._outbound.get()

    async def consume_outbound_for_channel(
        self,
        channel: str,
        timeout: float | None = None,
    ) -> OutboundMessage | None:
        """
        Consume outbound message for specific channel.

        Args:
            channel: Target channel name
            timeout: Optional timeout in seconds

        Returns:
            Outbound message for channel or None if timeout
        """
        try:
            if timeout:
                return await asyncio.wait_for(
                    self._channel_queues[channel].get(),
                    timeout=timeout,
                )
            return await self._channel_queues[channel].get()
        except asyncio.TimeoutError:
            return None

    def on_outbound(
        self, listener: Callable[[OutboundMessage], Awaitable[None]]
    ) -> Callable[[], None]:
        """
        Register outbound message listener.

        Args:
            listener: Async callback for outbound messages

        Returns:
            Unsubscribe function
        """
        self._outbound_listeners.append(listener)
        return lambda: self._outbound_listeners.remove(listener)

    # ========== System Events ==========

    async def publish_system_event(self, event: SystemEvent) -> None:
        """Publish a system event."""
        await self._system.put(event)

    async def consume_system_event(self) -> SystemEvent:
        """Consume next system event."""
        return await self._system.get()

    # ========== Queue Status ==========

    @property
    def inbound_size(self) -> int:
        """Get current inbound queue size."""
        return self._inbound.qsize()

    @property
    def outbound_size(self) -> int:
        """Get current outbound queue size."""
        return self._outbound.qsize()

    @property
    def stats(self) -> dict[str, int]:
        """Get message bus statistics."""
        return dict(self._stats)

    def clear(self) -> None:
        """Clear all queues."""
        while not self._inbound.empty():
            try:
                self._inbound.get_nowait()
            except asyncio.QueueEmpty:
                break

        while not self._outbound.empty():
            try:
                self._outbound.get_nowait()
            except asyncio.QueueEmpty:
                break

        for queue in self._channel_queues.values():
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break


__all__ = ["MessageBus"]
