"""Message adapters for ClawScope."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from clawscope.conversation_context import attach_runtime_context
from clawscope.message.base import Msg, TextBlock, ImageBlock
from clawscope.message.unified import UnifiedMessage

if TYPE_CHECKING:
    from clawscope.bus import InboundMessage, OutboundMessage


class MessageAdapter:
    """
    Adapter for converting between message formats.

    Supports conversions between:
    - AgentScope Msg
    - Nanobot InboundMessage/OutboundMessage
    - ClawScope UnifiedMessage
    """

    @staticmethod
    def inbound_to_msg(inbound: "InboundMessage") -> Msg:
        """
        Convert InboundMessage to AgentScope Msg.

        Args:
            inbound: Nanobot-style inbound message

        Returns:
            AgentScope-compatible Msg
        """
        # Build content with media if present
        if inbound.media:
            content = [TextBlock(text=inbound.content)]
            for url in inbound.media:
                # Determine media type from URL
                if any(ext in url.lower() for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]):
                    content.append(ImageBlock(source_type="url", source=url))
        else:
            content = inbound.content

        msg = Msg(
            name=inbound.sender_id,
            content=content,
            role="user",
            timestamp=inbound.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            metadata={
                "channel": inbound.channel,
                "chat_id": inbound.chat_id,
                **inbound.metadata,
            },
        )
        return attach_runtime_context(
            msg,
            channel=inbound.channel,
            chat_id=inbound.chat_id,
            session_key=inbound.session_key,
            sender_id=inbound.sender_id,
        )

    @staticmethod
    def msg_to_outbound(
        msg: Msg,
        channel: str,
        chat_id: str,
        reply_to: str | None = None,
    ) -> "OutboundMessage":
        """
        Convert AgentScope Msg to OutboundMessage.

        Args:
            msg: AgentScope Msg
            channel: Target channel
            chat_id: Target chat ID
            reply_to: Optional message ID to reply to

        Returns:
            Nanobot-style outbound message
        """
        from clawscope.bus import OutboundMessage

        # Extract text content
        text_content = msg.get_text_content()

        # Extract media URLs
        media = []
        if isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, ImageBlock):
                    media.append(block.source)
                elif isinstance(block, dict):
                    if block.get("type") == "image":
                        source = block.get("source", {})
                        if source.get("type") == "url":
                            media.append(source.get("url", ""))

        return OutboundMessage(
            channel=channel,
            chat_id=chat_id,
            content=text_content,
            media=media,
            reply_to=reply_to,
            metadata=msg.metadata,
        )

    @staticmethod
    def unified_to_msg(unified: UnifiedMessage) -> Msg:
        """Convert UnifiedMessage to AgentScope Msg."""
        return unified.to_agentscope_msg()

    @staticmethod
    def msg_to_unified(
        msg: Msg,
        channel: str | None = None,
        chat_id: str | None = None,
    ) -> UnifiedMessage:
        """Convert AgentScope Msg to UnifiedMessage."""
        return UnifiedMessage.from_agentscope_msg(msg, channel, chat_id)

    @staticmethod
    def inbound_to_unified(inbound: "InboundMessage") -> UnifiedMessage:
        """
        Convert InboundMessage to UnifiedMessage.

        Args:
            inbound: Nanobot-style inbound message

        Returns:
            ClawScope UnifiedMessage
        """
        # Build content with media if present
        if inbound.media:
            content = [TextBlock(text=inbound.content)]
            for url in inbound.media:
                if any(ext in url.lower() for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]):
                    content.append(ImageBlock(source_type="url", source=url))
        else:
            content = inbound.content

        return UnifiedMessage(
            name=inbound.sender_id,
            content=content,
            role="user",
            channel=inbound.channel,
            sender_id=inbound.sender_id,
            chat_id=inbound.chat_id,
            timestamp=inbound.timestamp,
            metadata=inbound.metadata,
            media=inbound.media,
        )

    @staticmethod
    def unified_to_outbound(unified: UnifiedMessage) -> "OutboundMessage":
        """
        Convert UnifiedMessage to OutboundMessage.

        Args:
            unified: ClawScope UnifiedMessage

        Returns:
            Nanobot-style outbound message
        """
        from clawscope.bus import OutboundMessage

        return OutboundMessage(
            channel=unified.channel or "cli",
            chat_id=unified.chat_id or "default",
            content=unified.get_text_content(),
            media=unified.media,
            reply_to=unified.reply_to,
            metadata=unified.metadata,
        )


__all__ = ["MessageAdapter"]
