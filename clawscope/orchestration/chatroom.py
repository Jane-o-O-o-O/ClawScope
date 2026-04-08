"""ChatRoom for multi-agent conversations."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from clawscope.message import Msg

if TYPE_CHECKING:
    from clawscope.agent import AgentBase


@dataclass
class ChatParticipant:
    """Participant in a ChatRoom."""

    agent: "AgentBase"
    role: str = "participant"  # host, participant, observer
    joined_at: datetime = field(default_factory=datetime.now)
    message_count: int = 0
    speaking_probability: float = 1.0  # For random speaking mode
    is_muted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class SpeakingPolicy:
    """Policy for determining who speaks next."""

    @staticmethod
    def round_robin(participants: list[ChatParticipant], history: list[Msg]) -> ChatParticipant | None:
        """Each participant speaks in order."""
        if not participants:
            return None

        if not history:
            return participants[0]

        # Find last speaker
        last_speaker = history[-1].name
        for i, p in enumerate(participants):
            if p.agent.name == last_speaker:
                next_idx = (i + 1) % len(participants)
                return participants[next_idx]

        return participants[0]

    @staticmethod
    def random(participants: list[ChatParticipant], history: list[Msg]) -> ChatParticipant | None:
        """Random participant speaks based on probability."""
        if not participants:
            return None

        eligible = [p for p in participants if not p.is_muted and random.random() < p.speaking_probability]

        if not eligible:
            # Fall back to any non-muted participant
            eligible = [p for p in participants if not p.is_muted]

        return random.choice(eligible) if eligible else None

    @staticmethod
    def host_moderated(participants: list[ChatParticipant], history: list[Msg]) -> ChatParticipant | None:
        """Host decides who speaks (alternates with host)."""
        hosts = [p for p in participants if p.role == "host"]
        if not hosts:
            return SpeakingPolicy.round_robin(participants, history)

        host = hosts[0]

        if not history or history[-1].name != host.agent.name:
            return host

        # Host just spoke, pick someone else
        others = [p for p in participants if p.role != "host" and not p.is_muted]
        return random.choice(others) if others else host

    @staticmethod
    def llm_decided(
        model: Any,
        participants: list[ChatParticipant],
        history: list[Msg],
    ) -> ChatParticipant | None:
        """
        Ask the LLM to choose the next speaker.

        The model receives a brief summary of the conversation and the list of
        available participants, then returns the name of who should speak next.
        Falls back to round-robin if the model is unavailable or returns an
        unrecognised name.

        Args:
            model: A ``ChatModelBase`` instance (or any object with a synchronous
                   ``chat()`` or ``__call__`` method that accepts a list of dicts).
            participants: Active, non-muted participants to choose from.
            history: Full conversation history so far.

        Returns:
            The chosen ``ChatParticipant``, or ``None`` if no participants exist.
        """
        if not participants:
            return None

        if model is None:
            return SpeakingPolicy.round_robin(participants, history)

        names = [p.agent.name for p in participants]
        names_str = ", ".join(names)

        # Build a short conversation snippet (last 6 messages max)
        recent = history[-6:]
        snippet = "\n".join(
            f"{m.name}: {m.get_text_content()[:120]}" for m in recent
        )

        prompt = (
            f"You are a conversation moderator.\n\n"
            f"Participants: {names_str}\n\n"
            f"Recent conversation:\n{snippet}\n\n"
            f"Who should speak next? Reply with ONLY one of the participant names "
            f"listed above and nothing else."
        )

        try:
            # Support both coroutine-based (async) and plain synchronous models.
            import asyncio
            import inspect

            messages = [{"role": "user", "content": prompt}]

            if hasattr(model, "chat"):
                result = model.chat(messages)
                if inspect.isawaitable(result):
                    # Synchronous context – run in a new loop or existing one
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # Can't block a running loop; fall back
                            return SpeakingPolicy.round_robin(participants, history)
                        response = loop.run_until_complete(result)
                    except RuntimeError:
                        response = asyncio.run(result)
                else:
                    response = result

                # Extract text from response
                chosen_name: str = ""
                if hasattr(response, "get_text_content"):
                    chosen_name = response.get_text_content().strip()
                elif hasattr(response, "content"):
                    chosen_name = str(response.content).strip()
                else:
                    chosen_name = str(response).strip()
            else:
                # Callable fallback (e.g. a plain function)
                chosen_name = str(model(prompt)).strip()

            # Match to a known participant (case-insensitive)
            chosen_name_lower = chosen_name.lower()
            for p in participants:
                if p.agent.name.lower() == chosen_name_lower:
                    logger.debug(f"LLM selected next speaker: {p.agent.name}")
                    return p

            logger.warning(
                f"LLM returned unknown speaker name {chosen_name!r}; "
                "falling back to round-robin"
            )

        except Exception as exc:
            logger.error(f"SpeakingPolicy.llm_decided error: {exc}; falling back")

        return SpeakingPolicy.round_robin(participants, history)


class ChatRoom:
    """
    ChatRoom for natural multi-agent conversations.

    Unlike MsgHub which is more structured, ChatRoom provides
    a more natural conversation flow with features like:
    - Speaking policies (who speaks next)
    - Roles (host, participant, observer)
    - Natural conversation dynamics
    """

    def __init__(
        self,
        name: str = "chatroom",
        max_messages: int = 100,
        speaking_policy: str | Callable = "round_robin",
        idle_timeout: float = 30.0,
    ):
        """
        Initialize ChatRoom.

        Args:
            name: Room name
            max_messages: Maximum messages before stopping
            speaking_policy: Policy for speaker selection
            idle_timeout: Timeout for inactive conversation
        """
        self.name = name
        self.max_messages = max_messages
        self.idle_timeout = idle_timeout
        self._participants: dict[str, ChatParticipant] = {}
        self._history: list[Msg] = []
        self._running = False

        # Set speaking policy
        if isinstance(speaking_policy, str):
            policies = {
                "round_robin": SpeakingPolicy.round_robin,
                "random": SpeakingPolicy.random,
                "host_moderated": SpeakingPolicy.host_moderated,
            }
            self.speaking_policy = policies.get(speaking_policy, SpeakingPolicy.round_robin)
        else:
            self.speaking_policy = speaking_policy

    def join(
        self,
        agent: "AgentBase",
        role: str = "participant",
        speaking_probability: float = 1.0,
    ) -> None:
        """
        Have an agent join the room.

        Args:
            agent: Agent to join
            role: Role in the room
            speaking_probability: Probability of speaking when selected
        """
        self._participants[agent.name] = ChatParticipant(
            agent=agent,
            role=role,
            speaking_probability=speaking_probability,
        )
        logger.info(f"{agent.name} joined {self.name} as {role}")

    def leave(self, agent_name: str) -> bool:
        """Have an agent leave the room."""
        if agent_name in self._participants:
            del self._participants[agent_name]
            logger.info(f"{agent_name} left {self.name}")
            return True
        return False

    def mute(self, agent_name: str) -> None:
        """Mute a participant."""
        if agent_name in self._participants:
            self._participants[agent_name].is_muted = True

    def unmute(self, agent_name: str) -> None:
        """Unmute a participant."""
        if agent_name in self._participants:
            self._participants[agent_name].is_muted = False

    @property
    def active_participants(self) -> list[ChatParticipant]:
        """Get active (non-muted, non-observer) participants."""
        return [
            p for p in self._participants.values()
            if not p.is_muted and p.role != "observer"
        ]

    @property
    def history(self) -> list[Msg]:
        """Get conversation history."""
        return self._history.copy()

    def say(self, msg: Msg) -> None:
        """Add a message to the room."""
        self._history.append(msg)

        # Update participant stats
        if msg.name in self._participants:
            self._participants[msg.name].message_count += 1

    async def run(
        self,
        topic: str | None = None,
        starter: str | None = None,
        termination_phrases: list[str] | None = None,
    ) -> list[Msg]:
        """
        Run the chatroom conversation.

        Args:
            topic: Conversation topic
            starter: Name of agent to start
            termination_phrases: Phrases that end conversation

        Returns:
            Conversation history
        """
        self._running = True
        termination_phrases = termination_phrases or ["goodbye", "bye", "end conversation"]

        # Start with topic if provided
        if topic:
            intro = Msg(
                name="system",
                content=f"Topic for discussion: {topic}",
                role="system",
            )
            self._history.append(intro)

        # Determine first speaker
        participants = self.active_participants

        if starter and starter in self._participants:
            current_speaker = self._participants[starter]
        elif participants:
            hosts = [p for p in participants if p.role == "host"]
            current_speaker = hosts[0] if hosts else participants[0]
        else:
            logger.warning("No active participants in chatroom")
            return self._history

        while self._running and len(self._history) < self.max_messages:
            # Get current speaker
            if current_speaker is None:
                current_speaker = self.speaking_policy(participants, self._history)

            if current_speaker is None:
                break

            # Build context for speaker
            context = self._build_context(current_speaker.agent.name)

            # Get response
            try:
                response = await asyncio.wait_for(
                    current_speaker.agent(context),
                    timeout=self.idle_timeout,
                )

                if response:
                    self._history.append(response)
                    current_speaker.message_count += 1

                    # Check for termination phrases
                    response_text = response.get_text_content().lower()
                    if any(phrase in response_text for phrase in termination_phrases):
                        logger.info("Termination phrase detected")
                        break

            except asyncio.TimeoutError:
                logger.warning(f"{current_speaker.agent.name} timed out")
            except Exception as e:
                logger.error(f"Error from {current_speaker.agent.name}: {e}")

            # Get next speaker
            current_speaker = self.speaking_policy(participants, self._history)

        self._running = False
        return self._history

    async def run_until(
        self,
        condition: Callable[[list[Msg]], bool],
        topic: str | None = None,
    ) -> list[Msg]:
        """
        Run until a condition is met.

        Args:
            condition: Function that returns True to stop
            topic: Conversation topic

        Returns:
            Conversation history
        """
        self._running = True

        if topic:
            intro = Msg(
                name="system",
                content=f"Topic: {topic}",
                role="system",
            )
            self._history.append(intro)

        participants = self.active_participants

        while self._running and not condition(self._history):
            if len(self._history) >= self.max_messages:
                break

            speaker = self.speaking_policy(participants, self._history)
            if not speaker:
                break

            context = self._build_context(speaker.agent.name)

            try:
                response = await speaker.agent(context)
                if response:
                    self._history.append(response)
                    speaker.message_count += 1
            except Exception as e:
                logger.error(f"Error: {e}")

        self._running = False
        return self._history

    def stop(self) -> None:
        """Stop the conversation."""
        self._running = False

    def clear(self) -> None:
        """Clear conversation history."""
        self._history.clear()
        for p in self._participants.values():
            p.message_count = 0

    def _build_context(self, speaker_name: str) -> Msg:
        """Build context message for a speaker."""
        # Get recent messages (last 20)
        recent = self._history[-20:]

        context_parts = []
        for msg in recent:
            context_parts.append(f"{msg.name}: {msg.get_text_content()}")

        context_text = "\n".join(context_parts)

        # Get speaker's role
        role = "participant"
        if speaker_name in self._participants:
            role = self._participants[speaker_name].role

        prompt = f"""You are {speaker_name}, a {role} in this conversation.

Recent conversation:
{context_text}

Please respond naturally to continue the conversation."""

        return Msg(name="system", content=prompt, role="user")

    def get_stats(self) -> dict[str, Any]:
        """Get chatroom statistics."""
        return {
            "name": self.name,
            "participants": len(self._participants),
            "active_participants": len(self.active_participants),
            "message_count": len(self._history),
            "participant_stats": {
                name: {
                    "role": p.role,
                    "messages": p.message_count,
                    "muted": p.is_muted,
                }
                for name, p in self._participants.items()
            },
        }


class Debate(ChatRoom):
    """
    Specialized ChatRoom for debates.

    Features:
    - Proposer and Opposer roles
    - Structured turn-taking
    - Judge for verdict
    """

    def __init__(
        self,
        topic: str,
        proposer: "AgentBase",
        opposer: "AgentBase",
        judge: "AgentBase | None" = None,
        rounds: int = 3,
    ):
        """
        Initialize debate.

        Args:
            topic: Debate topic
            proposer: Agent arguing for
            opposer: Agent arguing against
            judge: Optional judge agent
            rounds: Number of debate rounds
        """
        super().__init__(
            name=f"debate:{topic[:20]}",
            max_messages=rounds * 4 + 2,  # Opening + rounds + closing
            speaking_policy="round_robin",
        )

        self.topic = topic
        self.rounds = rounds

        self.join(proposer, role="proposer")
        self.join(opposer, role="opposer")
        if judge:
            self.join(judge, role="judge")

    async def run(self, **kwargs) -> list[Msg]:
        """Run the debate."""
        # Opening statement
        opening = Msg(
            name="moderator",
            content=f"Welcome to the debate on: {self.topic}\n"
                    f"Proposer will argue FOR, Opposer will argue AGAINST.\n"
                    f"We will have {self.rounds} rounds.",
            role="system",
        )
        self._history.append(opening)

        proposer = next(p for p in self._participants.values() if p.role == "proposer")
        opposer = next(p for p in self._participants.values() if p.role == "opposer")
        judge = next((p for p in self._participants.values() if p.role == "judge"), None)

        for round_num in range(self.rounds):
            # Proposer
            self._history.append(Msg(
                name="moderator",
                content=f"Round {round_num + 1}: Proposer, please present your argument.",
                role="system",
            ))

            context = self._build_debate_context(proposer.agent.name, "proposer")
            response = await proposer.agent(context)
            if response:
                self._history.append(response)

            # Opposer
            self._history.append(Msg(
                name="moderator",
                content=f"Round {round_num + 1}: Opposer, please respond.",
                role="system",
            ))

            context = self._build_debate_context(opposer.agent.name, "opposer")
            response = await opposer.agent(context)
            if response:
                self._history.append(response)

        # Judge verdict
        if judge:
            self._history.append(Msg(
                name="moderator",
                content="The debate has concluded. Judge, please give your verdict.",
                role="system",
            ))

            context = self._build_debate_context(judge.agent.name, "judge")
            verdict = await judge.agent(context)
            if verdict:
                self._history.append(verdict)

        return self._history

    def _build_debate_context(self, speaker_name: str, role: str) -> Msg:
        """Build debate-specific context."""
        context_parts = []
        for msg in self._history:
            context_parts.append(f"{msg.name}: {msg.get_text_content()}")

        context_text = "\n".join(context_parts)

        role_instructions = {
            "proposer": f"You are arguing FOR the topic: {self.topic}. Make compelling arguments.",
            "opposer": f"You are arguing AGAINST the topic: {self.topic}. Counter the proposer's points.",
            "judge": f"You are judging this debate on: {self.topic}. Evaluate both sides fairly.",
        }

        prompt = f"""{role_instructions.get(role, "")}

Debate so far:
{context_text}

Your response:"""

        return Msg(name="system", content=prompt, role="user")


__all__ = ["ChatRoom", "ChatParticipant", "SpeakingPolicy", "Debate"]
