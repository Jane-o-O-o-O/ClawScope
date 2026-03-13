"""User agent implementations for ClawScope."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from loguru import logger

from clawscope.agent.base import AgentBase
from clawscope.message import Msg

if TYPE_CHECKING:
    from clawscope.bus import MessageBus, InboundMessage
    from clawscope.message.adapters import MessageAdapter


class UserAgent(AgentBase):
    """
    User agent for interactive input.

    Collects input from user via configured input source
    (CLI, Studio, Channel, etc.)
    """

    def __init__(
        self,
        name: str = "User",
        input_prompt: str = ">>> ",
        **kwargs: Any,
    ):
        """
        Initialize user agent.

        Args:
            name: User name
            input_prompt: Prompt string for input
            **kwargs: Additional options
        """
        super().__init__(name=name, **kwargs)
        self.input_prompt = input_prompt

    async def reply(self, message: Msg | None = None, **kwargs: Any) -> Msg:
        """
        Get reply from user via CLI input.

        Args:
            message: Optional context message
            **kwargs: Additional options

        Returns:
            User's input message
        """
        # Show context if provided
        if message:
            print(f"\n{message.name}: {message.get_text_content()}\n")

        # Get input
        try:
            user_input = input(self.input_prompt)
        except EOFError:
            user_input = "exit"
        except KeyboardInterrupt:
            user_input = "exit"

        return Msg(
            name=self.name,
            content=user_input,
            role="user",
        )


class ChannelUserAgent(AgentBase):
    """
    User agent that receives input from a message channel.

    Bridges the MessageBus with the agent system, allowing
    multi-platform chat integration.
    """

    def __init__(
        self,
        name: str,
        bus: "MessageBus",
        channel: str,
        chat_id: str,
        **kwargs: Any,
    ):
        """
        Initialize channel user agent.

        Args:
            name: User name
            bus: Message bus instance
            channel: Channel name (telegram, discord, etc.)
            chat_id: Chat/conversation ID
            **kwargs: Additional options
        """
        super().__init__(name=name, **kwargs)
        self.bus = bus
        self.channel = channel
        self.chat_id = chat_id

    async def reply(self, message: Msg | None = None, **kwargs: Any) -> Msg:
        """
        Get reply from channel.

        Waits for next inbound message from the configured channel/chat.

        Args:
            message: Optional context message (sent to channel)
            **kwargs: Additional options

        Returns:
            User's message from channel
        """
        from clawscope.message.adapters import MessageAdapter

        # Send context message to channel if provided
        if message:
            outbound = MessageAdapter.msg_to_outbound(
                message, self.channel, self.chat_id
            )
            await self.bus.publish_outbound(outbound)

        # Wait for inbound message
        timeout = kwargs.get("timeout")
        inbound = await self.bus.consume_inbound_for(
            self.channel, self.chat_id, timeout=timeout
        )

        if inbound is None:
            return Msg(
                name=self.name,
                content="",
                role="user",
                metadata={"timeout": True},
            )

        # Convert to Msg
        return MessageAdapter.inbound_to_msg(inbound)

    async def observe(self, message: Msg | list[Msg] | None) -> None:
        """
        Observe message and forward to channel.

        Args:
            message: Message(s) to observe and forward
        """
        if message is None:
            return

        from clawscope.message.adapters import MessageAdapter

        messages = [message] if isinstance(message, Msg) else message
        for msg in messages:
            # Only forward assistant messages
            if msg.role == "assistant":
                outbound = MessageAdapter.msg_to_outbound(
                    msg, self.channel, self.chat_id
                )
                await self.bus.publish_outbound(outbound)

        # Also add to memory
        await super().observe(message)

    @property
    def session_key(self) -> str:
        """Get session key for this user agent."""
        return f"{self.channel}:{self.chat_id}"


__all__ = ["UserAgent", "ChannelUserAgent"]
