"""Session router for ClawScope."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, TYPE_CHECKING

from loguru import logger

from clawscope.message import Msg
from clawscope.message.adapters import MessageAdapter

if TYPE_CHECKING:
    from clawscope.bus import MessageBus, InboundMessage
    from clawscope.kernel import AgentKernel
    from clawscope.memory import SessionManager
    from clawscope.config import AgentConfig
    from clawscope.agent import AgentBase


class SessionRouter:
    """
    Routes messages to appropriate agent sessions.

    Manages the lifecycle of agent sessions and coordinates
    message flow between channels and agents.
    """

    def __init__(
        self,
        bus: "MessageBus",
        sessions: "SessionManager",
        kernel: "AgentKernel",
        config: "AgentConfig",
    ):
        """
        Initialize session router.

        Args:
            bus: Message bus instance
            sessions: Session manager
            kernel: Agent kernel
            config: Agent configuration
        """
        self.bus = bus
        self.sessions = sessions
        self.kernel = kernel
        self.config = config

        self._agents: dict[str, "AgentBase"] = {}
        self._running = False
        self._lock = asyncio.Lock()

    async def run(self) -> None:
        """Run the router main loop."""
        self._running = True
        logger.info("SessionRouter started")

        while self._running:
            try:
                # Get next inbound message
                message = await self.bus.consume_inbound()

                # Process in background
                asyncio.create_task(self._process_message(message))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Router error: {e}")

    def stop(self) -> None:
        """Stop the router."""
        self._running = False
        logger.info("SessionRouter stopping")

    async def _process_message(self, inbound: "InboundMessage") -> None:
        """Process an inbound message."""
        session_key = inbound.session_key

        try:
            # Get or create agent for this session
            agent = await self._get_or_create_agent(session_key, inbound.channel, inbound.chat_id)

            # Convert to Msg
            msg = MessageAdapter.inbound_to_msg(inbound)

            # Get agent response
            response = await agent(msg)

            # Send response back to channel
            if response:
                outbound = MessageAdapter.msg_to_outbound(
                    response,
                    inbound.channel,
                    inbound.chat_id,
                )
                await self.bus.publish_outbound(outbound)

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            # Send error message back
            from clawscope.bus import OutboundMessage
            await self.bus.publish_outbound(OutboundMessage(
                channel=inbound.channel,
                chat_id=inbound.chat_id,
                content=f"Sorry, an error occurred: {str(e)}",
            ))

    async def _get_or_create_agent(
        self,
        session_key: str,
        channel: str,
        chat_id: str,
    ) -> "AgentBase":
        """Get existing agent or create new one for session."""
        async with self._lock:
            if session_key not in self._agents:
                logger.info(f"Creating new agent for session: {session_key}")
                agent = await self._create_agent(session_key, channel, chat_id)
                self._agents[session_key] = agent

            return self._agents[session_key]

    async def _create_agent(
        self,
        session_key: str,
        channel: str,
        chat_id: str,
    ) -> "AgentBase":
        """Create a new agent for a session."""
        # Get session
        session = await self.sessions.get_or_create(session_key)

        # Create memory from session
        from clawscope.memory import SessionMemory
        memory = SessionMemory(session)

        return self.kernel.create_agent(
            name=self.config.name,
            sys_prompt=self.config.sys_prompt,
            memory=memory,
            max_iterations=self.config.max_iterations,
            session_key=session_key,
            channel=channel,
            chat_id=chat_id,
        )

    def get_active_sessions(self) -> list[str]:
        """Get list of active session keys."""
        return list(self._agents.keys())

    async def remove_session(self, session_key: str) -> None:
        """Remove an agent session."""
        async with self._lock:
            if session_key in self._agents:
                del self._agents[session_key]
                logger.info(f"Removed session: {session_key}")


__all__ = ["SessionRouter"]
