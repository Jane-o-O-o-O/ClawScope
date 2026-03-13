"""MsgHub for multi-agent message broadcasting."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Awaitable

from loguru import logger

from clawscope.message import Msg

if TYPE_CHECKING:
    from clawscope.agent import AgentBase


@dataclass
class Participant:
    """Participant in a MsgHub."""

    agent: "AgentBase"
    active: bool = True
    message_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class MsgHub:
    """
    Message Hub for multi-agent communication.

    Provides a shared message space where agents can broadcast
    messages to all participants. Supports both synchronous
    round-robin and asynchronous message passing.
    """

    def __init__(
        self,
        participants: list["AgentBase"] | None = None,
        announcement: Msg | None = None,
        max_rounds: int = 10,
        termination_condition: Callable[[list[Msg]], bool] | None = None,
    ):
        """
        Initialize MsgHub.

        Args:
            participants: Initial list of participating agents
            announcement: Initial announcement message
            max_rounds: Maximum number of broadcast rounds
            termination_condition: Function to check if hub should stop
        """
        self._participants: dict[str, Participant] = {}
        self._messages: list[Msg] = []
        self.max_rounds = max_rounds
        self.termination_condition = termination_condition

        # Add initial participants
        if participants:
            for agent in participants:
                self.add(agent)

        # Add announcement
        if announcement:
            self._messages.append(announcement)

    def add(self, agent: "AgentBase", **metadata: Any) -> None:
        """
        Add an agent to the hub.

        Args:
            agent: Agent to add
            **metadata: Additional metadata
        """
        self._participants[agent.name] = Participant(
            agent=agent,
            metadata=metadata,
        )
        logger.debug(f"Added {agent.name} to MsgHub")

    def remove(self, agent_name: str) -> bool:
        """
        Remove an agent from the hub.

        Args:
            agent_name: Name of agent to remove

        Returns:
            True if removed
        """
        if agent_name in self._participants:
            del self._participants[agent_name]
            logger.debug(f"Removed {agent_name} from MsgHub")
            return True
        return False

    def deactivate(self, agent_name: str) -> None:
        """Temporarily deactivate an agent."""
        if agent_name in self._participants:
            self._participants[agent_name].active = False

    def activate(self, agent_name: str) -> None:
        """Reactivate an agent."""
        if agent_name in self._participants:
            self._participants[agent_name].active = True

    @property
    def active_participants(self) -> list[Participant]:
        """Get list of active participants."""
        return [p for p in self._participants.values() if p.active]

    @property
    def messages(self) -> list[Msg]:
        """Get all messages in the hub."""
        return self._messages.copy()

    def broadcast(self, msg: Msg) -> None:
        """
        Broadcast a message to all participants.

        Args:
            msg: Message to broadcast
        """
        self._messages.append(msg)
        logger.debug(f"Broadcast message from {msg.name}")

    async def run(
        self,
        initial_message: Msg | None = None,
        order: list[str] | None = None,
    ) -> list[Msg]:
        """
        Run the hub in round-robin mode.

        Each agent takes turns responding to the accumulated messages.

        Args:
            initial_message: Starting message
            order: Custom order of agent names

        Returns:
            All messages exchanged
        """
        if initial_message:
            self._messages.append(initial_message)

        # Determine order
        if order:
            participants = [
                self._participants[name]
                for name in order
                if name in self._participants
            ]
        else:
            participants = self.active_participants

        if not participants:
            logger.warning("No active participants in MsgHub")
            return self._messages

        for round_num in range(self.max_rounds):
            logger.debug(f"MsgHub round {round_num + 1}/{self.max_rounds}")

            for participant in participants:
                if not participant.active:
                    continue

                # Build context from all messages
                context = self._build_context(participant.agent.name)

                # Get agent response
                try:
                    response = await participant.agent(context)

                    if response:
                        self._messages.append(response)
                        participant.message_count += 1
                        logger.debug(f"{participant.agent.name} responded")

                except Exception as e:
                    logger.error(f"Agent {participant.agent.name} error: {e}")

            # Check termination
            if self.termination_condition and self.termination_condition(self._messages):
                logger.info("MsgHub termination condition met")
                break

        return self._messages

    async def run_async(
        self,
        initial_message: Msg | None = None,
        timeout: float = 60.0,
    ) -> list[Msg]:
        """
        Run the hub in async mode.

        All agents respond concurrently to each message.

        Args:
            initial_message: Starting message
            timeout: Timeout per round

        Returns:
            All messages exchanged
        """
        if initial_message:
            self._messages.append(initial_message)

        participants = self.active_participants

        for round_num in range(self.max_rounds):
            logger.debug(f"MsgHub async round {round_num + 1}")

            # Build context for all agents
            tasks = []
            for participant in participants:
                context = self._build_context(participant.agent.name)
                tasks.append(self._get_response(participant, context))

            # Wait for all responses
            try:
                responses = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=timeout,
                )

                for response in responses:
                    if isinstance(response, Msg):
                        self._messages.append(response)

            except asyncio.TimeoutError:
                logger.warning(f"MsgHub round {round_num + 1} timed out")

            # Check termination
            if self.termination_condition and self.termination_condition(self._messages):
                break

        return self._messages

    async def _get_response(
        self,
        participant: Participant,
        context: Msg,
    ) -> Msg | None:
        """Get response from a participant."""
        try:
            response = await participant.agent(context)
            if response:
                participant.message_count += 1
            return response
        except Exception as e:
            logger.error(f"Agent {participant.agent.name} error: {e}")
            return None

    def _build_context(self, exclude_name: str | None = None) -> Msg:
        """
        Build context message from history.

        Args:
            exclude_name: Agent name to exclude from context

        Returns:
            Context message
        """
        # Combine all messages into context
        context_parts = []

        for msg in self._messages:
            if exclude_name and msg.name == exclude_name:
                continue
            context_parts.append(f"{msg.name}: {msg.get_text_content()}")

        context_text = "\n".join(context_parts)

        return Msg(
            name="hub",
            content=f"Conversation history:\n{context_text}\n\nPlease respond:",
            role="user",
        )

    def __enter__(self) -> "MsgHub":
        """Context manager entry."""
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit."""
        pass


class MsgHubBuilder:
    """Builder for creating MsgHub instances."""

    def __init__(self):
        self._participants: list["AgentBase"] = []
        self._announcement: Msg | None = None
        self._max_rounds: int = 10
        self._termination: Callable[[list[Msg]], bool] | None = None

    def add_participant(self, agent: "AgentBase") -> "MsgHubBuilder":
        """Add a participant."""
        self._participants.append(agent)
        return self

    def add_participants(self, agents: list["AgentBase"]) -> "MsgHubBuilder":
        """Add multiple participants."""
        self._participants.extend(agents)
        return self

    def with_announcement(self, msg: Msg | str) -> "MsgHubBuilder":
        """Set announcement message."""
        if isinstance(msg, str):
            msg = Msg(name="system", content=msg, role="system")
        self._announcement = msg
        return self

    def max_rounds(self, rounds: int) -> "MsgHubBuilder":
        """Set maximum rounds."""
        self._max_rounds = rounds
        return self

    def terminate_when(
        self,
        condition: Callable[[list[Msg]], bool],
    ) -> "MsgHubBuilder":
        """Set termination condition."""
        self._termination = condition
        return self

    def build(self) -> MsgHub:
        """Build the MsgHub."""
        return MsgHub(
            participants=self._participants,
            announcement=self._announcement,
            max_rounds=self._max_rounds,
            termination_condition=self._termination,
        )


__all__ = ["MsgHub", "MsgHubBuilder", "Participant"]
