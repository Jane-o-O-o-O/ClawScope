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

    When sub-agents are registered (via :meth:`register_sub_agent`), each
    new session receives an :class:`~clawscope.agent.OrchestratorAgent` that
    automatically delegates to those sub-agents instead of a plain ReActAgent.
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

        # Sub-agents shared across all sessions (stateless per-call)
        self._sub_agents: dict[str, "AgentBase"] = {}

    # ------------------------------------------------------------------
    # Sub-agent management
    # ------------------------------------------------------------------

    def register_sub_agent(self, name: str, agent: "AgentBase") -> None:
        """
        Register a sub-agent that the orchestrator can delegate to.

        Calling this at least once switches all *future* sessions from a
        plain ReActAgent to an :class:`~clawscope.agent.OrchestratorAgent`.
        Existing sessions are not affected.

        Args:
            name: Logical name used as the tool key (``ask_<name>``).
            agent: Agent instance to delegate to.
        """
        self._sub_agents[name] = agent
        logger.info(f"SessionRouter: registered sub-agent '{name}'")

    def unregister_sub_agent(self, name: str) -> bool:
        """Remove a sub-agent. Returns True if it existed."""
        existed = name in self._sub_agents
        self._sub_agents.pop(name, None)
        return existed

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

            # Inject a fresh ProgressReporter so progress goes to this exact chat
            from clawscope.agent.orchestrator import OrchestratorAgent, ProgressReporter
            if isinstance(agent, OrchestratorAgent):
                agent.set_progress_reporter(
                    ProgressReporter(self.bus, inbound.channel, inbound.chat_id)
                )

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
        """Create a new agent for a session.

        If sub-agents are registered, creates an OrchestratorAgent that
        wraps them as tools. Otherwise falls back to the kernel's default agent.
        """
        session = await self.sessions.get_or_create(session_key)

        from clawscope.memory import SessionMemory
        memory = SessionMemory(session)

        if self._sub_agents:
            # Orchestrator mode: build an OrchestratorAgent with a model
            # sourced from the kernel (NativeKernel exposes model_registry).
            from clawscope.agent.orchestrator import OrchestratorAgent

            model = getattr(self.kernel, "model_registry", None)
            if model is not None:
                model = model.get_model()
            else:
                model = getattr(self.kernel, "model_config", None)

            orchestrator = OrchestratorAgent(
                name=self.config.name or "Orchestrator",
                sys_prompt=self.config.sys_prompt or "",
                model=model,
                memory=memory,
                sub_agents=dict(self._sub_agents),  # snapshot
                max_iterations=self.config.max_iterations,
            )
            logger.info(
                f"SessionRouter: created OrchestratorAgent for session '{session_key}' "
                f"with {len(self._sub_agents)} sub-agent(s)"
            )
            return orchestrator

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
