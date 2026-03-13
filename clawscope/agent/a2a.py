"""A2A (Agent-to-Agent) communication protocol."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Awaitable
from uuid import uuid4

from loguru import logger

from clawscope.message import Msg

if TYPE_CHECKING:
    from clawscope.agent import AgentBase


class A2AMessageType(str, Enum):
    """Types of A2A messages."""

    # Task delegation
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    TASK_PROGRESS = "task_progress"
    TASK_CANCEL = "task_cancel"

    # Discovery
    DISCOVER = "discover"
    ANNOUNCE = "announce"
    CAPABILITIES = "capabilities"

    # Communication
    MESSAGE = "message"
    QUERY = "query"
    NOTIFY = "notify"

    # Coordination
    SYNC = "sync"
    HEARTBEAT = "heartbeat"


@dataclass
class A2AMessage:
    """Message for A2A communication."""

    type: A2AMessageType
    sender: str
    recipient: str
    payload: dict[str, Any]
    message_id: str = field(default_factory=lambda: str(uuid4()))
    correlation_id: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    ttl: int = 60  # Time-to-live in seconds
    priority: int = 5  # 1-10, higher is more important

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type.value,
            "sender": self.sender,
            "recipient": self.recipient,
            "payload": self.payload,
            "message_id": self.message_id,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat(),
            "ttl": self.ttl,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "A2AMessage":
        """Create from dictionary."""
        return cls(
            type=A2AMessageType(data["type"]),
            sender=data["sender"],
            recipient=data["recipient"],
            payload=data.get("payload", {}),
            message_id=data.get("message_id", str(uuid4())),
            correlation_id=data.get("correlation_id"),
            timestamp=datetime.fromisoformat(data["timestamp"])
            if "timestamp" in data
            else datetime.now(),
            ttl=data.get("ttl", 60),
            priority=data.get("priority", 5),
        )

    def create_response(self, payload: dict[str, Any]) -> "A2AMessage":
        """Create a response to this message."""
        return A2AMessage(
            type=A2AMessageType.TASK_RESPONSE,
            sender=self.recipient,
            recipient=self.sender,
            payload=payload,
            correlation_id=self.message_id,
        )


@dataclass
class AgentCapability:
    """Describes an agent's capability."""

    name: str
    description: str
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class AgentCard:
    """Agent's identity card for discovery."""

    name: str
    description: str
    capabilities: list[AgentCapability] = field(default_factory=list)
    status: str = "available"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": [
                {
                    "name": c.name,
                    "description": c.description,
                    "tags": c.tags,
                }
                for c in self.capabilities
            ],
            "status": self.status,
            "metadata": self.metadata,
        }


class A2ARouter:
    """
    Router for A2A message delivery.

    Handles message routing between agents, with support for:
    - Direct messaging
    - Broadcast
    - Capability-based routing
    """

    def __init__(self):
        self._agents: dict[str, "A2AAgent"] = {}
        self._message_queue: asyncio.Queue[A2AMessage] = asyncio.Queue()
        self._running = False

    def register(self, agent: "A2AAgent") -> None:
        """Register an agent with the router."""
        self._agents[agent.name] = agent
        logger.debug(f"Registered A2A agent: {agent.name}")

    def unregister(self, agent_name: str) -> None:
        """Unregister an agent."""
        self._agents.pop(agent_name, None)

    async def send(self, message: A2AMessage) -> None:
        """Send a message through the router."""
        await self._message_queue.put(message)

    async def send_direct(self, message: A2AMessage) -> A2AMessage | None:
        """Send a message and wait for response."""
        if message.recipient not in self._agents:
            logger.warning(f"Recipient not found: {message.recipient}")
            return None

        recipient = self._agents[message.recipient]
        return await recipient.receive(message)

    async def broadcast(self, message: A2AMessage) -> list[A2AMessage]:
        """Broadcast a message to all agents."""
        responses = []

        for name, agent in self._agents.items():
            if name == message.sender:
                continue

            msg_copy = A2AMessage(
                type=message.type,
                sender=message.sender,
                recipient=name,
                payload=message.payload,
                correlation_id=message.message_id,
            )

            response = await agent.receive(msg_copy)
            if response:
                responses.append(response)

        return responses

    async def discover(
        self,
        capability: str | None = None,
        tag: str | None = None,
    ) -> list[AgentCard]:
        """Discover agents with specific capabilities."""
        results = []

        for agent in self._agents.values():
            card = agent.get_card()

            if capability:
                has_cap = any(c.name == capability for c in card.capabilities)
                if not has_cap:
                    continue

            if tag:
                has_tag = any(
                    tag in c.tags
                    for c in card.capabilities
                )
                if not has_tag:
                    continue

            results.append(card)

        return results

    async def run(self) -> None:
        """Run the message routing loop."""
        self._running = True

        while self._running:
            try:
                message = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=1.0,
                )

                # Route message
                if message.recipient == "*":
                    await self.broadcast(message)
                elif message.recipient in self._agents:
                    await self._agents[message.recipient].receive(message)
                else:
                    logger.warning(f"Unknown recipient: {message.recipient}")

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Router error: {e}")

    def stop(self) -> None:
        """Stop the router."""
        self._running = False


class A2AAgent:
    """
    Agent capable of A2A (Agent-to-Agent) communication.

    Wraps a base agent and adds A2A protocol support.
    """

    def __init__(
        self,
        agent: "AgentBase",
        router: A2ARouter | None = None,
        capabilities: list[AgentCapability] | None = None,
        description: str | None = None,
    ):
        """
        Initialize A2A agent.

        Args:
            agent: Base agent
            router: A2A router for message delivery
            capabilities: Agent capabilities
            description: Agent description
        """
        self.agent = agent
        self.router = router
        self.capabilities = capabilities or []
        self.description = description or f"A2A-enabled {agent.name}"

        self._handlers: dict[A2AMessageType, Callable] = {}
        self._pending_tasks: dict[str, asyncio.Future] = {}

        # Register default handlers
        self._register_default_handlers()

        # Register with router
        if router:
            router.register(self)

    @property
    def name(self) -> str:
        """Get agent name."""
        return self.agent.name

    def _register_default_handlers(self) -> None:
        """Register default message handlers."""
        self._handlers[A2AMessageType.TASK_REQUEST] = self._handle_task_request
        self._handlers[A2AMessageType.QUERY] = self._handle_query
        self._handlers[A2AMessageType.DISCOVER] = self._handle_discover
        self._handlers[A2AMessageType.CAPABILITIES] = self._handle_capabilities

    def on(
        self,
        message_type: A2AMessageType,
    ) -> Callable:
        """Decorator to register a message handler."""
        def decorator(func: Callable) -> Callable:
            self._handlers[message_type] = func
            return func
        return decorator

    def get_card(self) -> AgentCard:
        """Get agent's identity card."""
        return AgentCard(
            name=self.name,
            description=self.description,
            capabilities=self.capabilities,
        )

    async def receive(self, message: A2AMessage) -> A2AMessage | None:
        """
        Receive and process a message.

        Args:
            message: Incoming message

        Returns:
            Response message if applicable
        """
        logger.debug(f"{self.name} received {message.type} from {message.sender}")

        # Check for handler
        handler = self._handlers.get(message.type)
        if handler:
            return await handler(message)

        # Default: pass to base agent
        return await self._handle_default(message)

    async def send(
        self,
        recipient: str,
        message_type: A2AMessageType,
        payload: dict[str, Any],
        wait_response: bool = True,
    ) -> A2AMessage | None:
        """
        Send a message to another agent.

        Args:
            recipient: Target agent name
            message_type: Message type
            payload: Message payload
            wait_response: Wait for response

        Returns:
            Response message if wait_response is True
        """
        if not self.router:
            logger.warning("No router configured")
            return None

        message = A2AMessage(
            type=message_type,
            sender=self.name,
            recipient=recipient,
            payload=payload,
        )

        if wait_response:
            return await self.router.send_direct(message)
        else:
            await self.router.send(message)
            return None

    async def delegate_task(
        self,
        recipient: str,
        task: str,
        context: dict[str, Any] | None = None,
        timeout: float = 60.0,
    ) -> Any:
        """
        Delegate a task to another agent.

        Args:
            recipient: Target agent
            task: Task description
            context: Additional context
            timeout: Timeout in seconds

        Returns:
            Task result
        """
        payload = {
            "task": task,
            "context": context or {},
        }

        response = await asyncio.wait_for(
            self.send(
                recipient=recipient,
                message_type=A2AMessageType.TASK_REQUEST,
                payload=payload,
                wait_response=True,
            ),
            timeout=timeout,
        )

        if response and response.payload:
            return response.payload.get("result")

        return None

    async def query(
        self,
        recipient: str,
        question: str,
        timeout: float = 30.0,
    ) -> str | None:
        """
        Query another agent.

        Args:
            recipient: Target agent
            question: Question to ask
            timeout: Timeout

        Returns:
            Answer string
        """
        response = await asyncio.wait_for(
            self.send(
                recipient=recipient,
                message_type=A2AMessageType.QUERY,
                payload={"question": question},
                wait_response=True,
            ),
            timeout=timeout,
        )

        if response and response.payload:
            return response.payload.get("answer")

        return None

    async def discover_agents(
        self,
        capability: str | None = None,
        tag: str | None = None,
    ) -> list[AgentCard]:
        """Discover other agents."""
        if not self.router:
            return []

        return await self.router.discover(capability=capability, tag=tag)

    async def _handle_task_request(self, message: A2AMessage) -> A2AMessage:
        """Handle incoming task request."""
        task = message.payload.get("task", "")
        context = message.payload.get("context", {})

        # Create message for base agent
        msg = Msg(
            name=message.sender,
            content=task,
            role="user",
            metadata=context,
        )

        # Process with base agent
        try:
            response = await self.agent(msg)
            result = response.get_text_content() if response else ""

            return message.create_response({
                "status": "completed",
                "result": result,
            })

        except Exception as e:
            return message.create_response({
                "status": "failed",
                "error": str(e),
            })

    async def _handle_query(self, message: A2AMessage) -> A2AMessage:
        """Handle query message."""
        question = message.payload.get("question", "")

        msg = Msg(
            name=message.sender,
            content=question,
            role="user",
        )

        try:
            response = await self.agent(msg)
            answer = response.get_text_content() if response else ""

            return message.create_response({
                "answer": answer,
            })

        except Exception as e:
            return message.create_response({
                "error": str(e),
            })

    async def _handle_discover(self, message: A2AMessage) -> A2AMessage:
        """Handle discovery request."""
        return message.create_response({
            "card": self.get_card().to_dict(),
        })

    async def _handle_capabilities(self, message: A2AMessage) -> A2AMessage:
        """Handle capabilities request."""
        return message.create_response({
            "capabilities": [
                {
                    "name": c.name,
                    "description": c.description,
                    "tags": c.tags,
                }
                for c in self.capabilities
            ],
        })

    async def _handle_default(self, message: A2AMessage) -> A2AMessage | None:
        """Default message handler."""
        logger.warning(f"No handler for message type: {message.type}")
        return None


# Global router instance
_global_router: A2ARouter | None = None


def get_router() -> A2ARouter:
    """Get the global A2A router."""
    global _global_router
    if _global_router is None:
        _global_router = A2ARouter()
    return _global_router


__all__ = [
    "A2AMessage",
    "A2AMessageType",
    "A2AAgent",
    "A2ARouter",
    "AgentCard",
    "AgentCapability",
    "get_router",
]
