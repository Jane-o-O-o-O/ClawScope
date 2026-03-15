"""Kernel-owned reasoning loops for agent execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncIterator

from loguru import logger

from clawscope.message import Msg, ToolResultBlock, ToolUseBlock

if TYPE_CHECKING:
    from clawscope.agent import AgentBase
    from clawscope.model import ToolCall


@dataclass(slots=True)
class LoopConfig:
    """Configuration for a kernel-managed reasoning loop."""

    max_iterations: int = 40
    max_tokens: int = 4096


class AgentLoop(ABC):
    """Abstract reasoning loop owned by a kernel."""

    def __init__(self, config: LoopConfig):
        self.config = config

    @abstractmethod
    async def run(
        self,
        agent: AgentBase,
        message: Msg | None = None,
        **kwargs: Any,
    ) -> Msg:
        """Run the agent loop until a final assistant message is produced."""

    async def stream(
        self,
        agent: AgentBase,
        message: Msg | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream the agent loop when supported by the underlying model."""
        raise NotImplementedError("Streaming is not implemented for this agent loop")


class NativeAgentLoop(AgentLoop):
    """Native ReAct-style reasoning loop managed by the kernel."""

    async def run(
        self,
        agent: AgentBase,
        message: Msg | None = None,
        **kwargs: Any,
    ) -> Msg:
        if message is not None:
            await agent.observe(message)

        messages = await self._build_conversation(agent)
        tools = agent.get_tool_definitions() if agent.tools else None
        final_response: Msg | None = None

        for iteration in range(1, self.config.max_iterations + 1):
            logger.debug(
                "NativeAgentLoop iteration {}/{}",
                iteration,
                self.config.max_iterations,
            )

            model = agent.model
            if model is None:
                raise RuntimeError("No model configured for agent")

            response = await model.chat(
                messages=messages,
                tools=tools,
                max_tokens=self.config.max_tokens,
                **kwargs,
            )
            response_msg = response.to_msg(name=agent.name)

            if response.has_tool_calls():
                messages.append(response_msg)
                await agent.observe(response_msg)

                tool_results = await self._execute_tools(agent, response.tool_calls)
                tool_result_msg = self._create_tool_result_msg(tool_results)
                messages.append(tool_result_msg)
                await agent.observe(tool_result_msg)
                continue

            final_response = response_msg
            break

        if final_response is None:
            final_response = self._build_max_iterations_message(agent.name)

        await agent.observe(final_response)
        return final_response

    async def stream(
        self,
        agent: AgentBase,
        message: Msg | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        if message is not None:
            await agent.observe(message)

        messages = await self._build_conversation(agent)
        tools = agent.get_tool_definitions() if agent.tools else None

        for _ in range(self.config.max_iterations):
            model = agent.model
            if model is None:
                raise RuntimeError("No model configured for agent")

            content_buffer = ""
            tool_calls: list[ToolCall] = []

            async for chunk in model.stream_chat(
                messages=messages,
                tools=tools,
                max_tokens=self.config.max_tokens,
                **kwargs,
            ):
                if chunk.content:
                    content_buffer += chunk.content
                    yield {"type": "content", "content": chunk.content}

                if chunk.thinking_content:
                    yield {"type": "thinking", "content": chunk.thinking_content}

                if chunk.tool_calls:
                    tool_calls = chunk.tool_calls

                if chunk.finish_reason:
                    break

            if tool_calls:
                assistant_msg = Msg(
                    name=agent.name,
                    content=[
                        ToolUseBlock(
                            id=tool_call.id,
                            name=tool_call.name,
                            input=tool_call.arguments,
                        )
                        for tool_call in tool_calls
                    ],
                    role="assistant",
                )
                messages.append(assistant_msg)
                await agent.observe(assistant_msg)

                tool_results: list[dict[str, Any]] = []
                for tool_call in tool_calls:
                    yield {
                        "type": "tool_start",
                        "tool_name": tool_call.name,
                        "tool_id": tool_call.id,
                    }
                    tool_result = await self._execute_single_tool(agent, tool_call)
                    tool_results.append(tool_result)
                    yield {
                        "type": "tool_result",
                        "tool_id": tool_call.id,
                        "content": tool_result["content"],
                        "is_error": tool_result["is_error"],
                    }

                tool_result_msg = self._create_tool_result_msg(tool_results)
                messages.append(tool_result_msg)
                await agent.observe(tool_result_msg)
                continue

            final_msg = Msg(
                name=agent.name,
                content=content_buffer,
                role="assistant",
            )
            await agent.observe(final_msg)
            yield {"type": "done", "message": final_msg}
            return

        final_msg = self._build_max_iterations_message(agent.name)
        await agent.observe(final_msg)
        yield {"type": "done", "message": final_msg}

    async def _build_conversation(self, agent: AgentBase) -> list[Msg]:
        """Assemble system prompt and memory into the next model input."""
        messages: list[Msg] = []
        sys_msg = agent._build_system_message()
        if sys_msg is not None:
            messages.append(sys_msg)
        messages.extend(await agent.get_memory_messages())
        return messages

    async def _execute_tools(
        self,
        agent: AgentBase,
        tool_calls: list[ToolCall],
    ) -> list[dict[str, Any]]:
        """Execute tool calls and collect normalized results."""
        results: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            results.append(await self._execute_single_tool(agent, tool_call))
        return results

    async def _execute_single_tool(
        self,
        agent: AgentBase,
        tool_call: ToolCall,
    ) -> dict[str, Any]:
        """Execute one tool call and normalize success and failure payloads."""
        try:
            logger.info("Executing tool: {}", tool_call.name)
            result = await agent.execute_tool(tool_call.name, tool_call.arguments)
            return {
                "tool_use_id": tool_call.id,
                "content": result,
                "is_error": False,
            }
        except Exception as exc:
            logger.error("Tool execution error: {}", exc)
            return {
                "tool_use_id": tool_call.id,
                "content": f"Error: {exc}",
                "is_error": True,
            }

    def _create_tool_result_msg(self, results: list[dict[str, Any]]) -> Msg:
        """Pack tool execution results into a single tool-role message."""
        return Msg(
            name="tool",
            content=[
                ToolResultBlock(
                    tool_use_id=result["tool_use_id"],
                    content=result["content"],
                    is_error=result["is_error"],
                )
                for result in results
            ],
            role="tool",
        )

    def _build_max_iterations_message(self, agent_name: str) -> Msg:
        """Return the fallback message when the loop exhausts its budget."""
        return Msg(
            name=agent_name,
            content=(
                "I've reached the maximum number of reasoning steps. "
                "Here's what I've accomplished so far."
            ),
            role="assistant",
        )


__all__ = ["AgentLoop", "LoopConfig", "NativeAgentLoop"]
