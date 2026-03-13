"""Pipeline implementations for ClawScope."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

from loguru import logger

from clawscope.message import Msg

if TYPE_CHECKING:
    from clawscope.agent import AgentBase


class Pipeline(ABC):
    """Abstract base class for agent pipelines."""

    @abstractmethod
    async def run(self, input_msg: Msg | None = None) -> Msg | None:
        """
        Run the pipeline.

        Args:
            input_msg: Initial input message

        Returns:
            Final output message
        """
        pass


class SequentialPipeline(Pipeline):
    """
    Sequential pipeline that runs agents one after another.

    Each agent's output becomes the next agent's input.
    """

    def __init__(self, agents: list["AgentBase"]):
        """
        Initialize sequential pipeline.

        Args:
            agents: List of agents to run in sequence
        """
        self.agents = agents

    async def run(self, input_msg: Msg | None = None) -> Msg | None:
        """
        Run agents sequentially.

        Args:
            input_msg: Initial input message

        Returns:
            Output from the last agent
        """
        current_msg = input_msg

        for i, agent in enumerate(self.agents):
            logger.debug(f"Sequential pipeline step {i+1}/{len(self.agents)}: {agent.name}")
            current_msg = await agent(current_msg)

            if current_msg is None:
                logger.warning(f"Agent {agent.name} returned None, stopping pipeline")
                break

        return current_msg


class FanOutPipeline(Pipeline):
    """
    Fan-out pipeline that runs agents in parallel.

    All agents receive the same input and run concurrently.
    """

    def __init__(self, agents: list["AgentBase"]):
        """
        Initialize fan-out pipeline.

        Args:
            agents: List of agents to run in parallel
        """
        self.agents = agents

    async def run(self, input_msg: Msg | None = None) -> list[Msg]:
        """
        Run agents in parallel.

        Args:
            input_msg: Input message for all agents

        Returns:
            List of outputs from all agents
        """
        import asyncio

        logger.debug(f"Fan-out pipeline running {len(self.agents)} agents in parallel")

        tasks = [agent(input_msg) for agent in self.agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        outputs = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Agent {self.agents[i].name} failed: {result}")
            elif result is not None:
                outputs.append(result)

        return outputs


class ConditionalPipeline(Pipeline):
    """
    Conditional pipeline that routes to different agents.

    Uses a selector function to choose which agent handles the message.
    """

    def __init__(
        self,
        agents: dict[str, "AgentBase"],
        selector: callable,
        default: str | None = None,
    ):
        """
        Initialize conditional pipeline.

        Args:
            agents: Dict mapping names to agents
            selector: Function that takes a Msg and returns agent name
            default: Default agent name if selector returns None
        """
        self.agents = agents
        self.selector = selector
        self.default = default

    async def run(self, input_msg: Msg | None = None) -> Msg | None:
        """
        Route message to appropriate agent.

        Args:
            input_msg: Input message to route

        Returns:
            Output from selected agent
        """
        if input_msg is None:
            return None

        # Select agent
        agent_name = self.selector(input_msg)
        if agent_name is None:
            agent_name = self.default

        if agent_name is None or agent_name not in self.agents:
            logger.warning(f"No agent found for selection: {agent_name}")
            return None

        agent = self.agents[agent_name]
        logger.debug(f"Conditional pipeline selected: {agent_name}")

        return await agent(input_msg)


__all__ = [
    "Pipeline",
    "SequentialPipeline",
    "FanOutPipeline",
    "ConditionalPipeline",
]
