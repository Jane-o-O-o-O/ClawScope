"""ReAct agent implementation for ClawScope."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING
from dataclasses import dataclass, field

from loguru import logger

from clawscope.agent.base import AgentBase
from clawscope.message import Msg, TextBlock, ToolUseBlock, ToolResultBlock

if TYPE_CHECKING:
    from clawscope.memory import UnifiedMemory
    from clawscope.model import ChatModelBase
    from clawscope.tool import ToolRegistry


@dataclass
class CompressionConfig:
    """Configuration for memory compression."""

    enabled: bool = True
    trigger_tokens: int = 50000  # Trigger compression at this token count
    target_tokens: int = 30000  # Target token count after compression
    preserve_recent: int = 10  # Always preserve this many recent messages


class ReActAgent(AgentBase):
    """
    ReAct (Reasoning + Acting) agent implementation.

    Features:
    - Iterative reasoning and tool use
    - Memory compression
    - Streaming support
    - Extended thinking (when supported by model)
    """

    def __init__(
        self,
        name: str,
        sys_prompt: str = "You are a helpful AI assistant.",
        model: "ChatModelBase | None" = None,
        memory: "UnifiedMemory | None" = None,
        tools: "ToolRegistry | None" = None,
        max_iterations: int = 40,
        max_tokens: int = 4096,
        compression: CompressionConfig | None = None,
        **kwargs: Any,
    ):
        """
        Initialize ReAct agent.

        Args:
            name: Agent name
            sys_prompt: System prompt
            model: Chat model
            memory: Memory system
            tools: Tool registry
            max_iterations: Maximum reasoning iterations
            max_tokens: Maximum tokens per response
            compression: Memory compression config
            **kwargs: Additional options
        """
        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            model=model,
            memory=memory,
            tools=tools,
            **kwargs,
        )
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.compression = compression or CompressionConfig()

    async def reply(self, message: Msg | None = None, **kwargs: Any) -> Msg:
        """
        Generate a reply using ReAct loop.

        Args:
            message: Input message
            **kwargs: Additional options

        Returns:
            Agent's final response
        """
        # Run pre-reply hooks
        kwargs = await self._run_pre_reply_hooks(message=message, **kwargs)
        message = kwargs.pop("message", message)

        # Add input message to memory
        if message:
            await self.observe(message)

        # Build conversation
        messages = await self._build_conversation()

        # Get tool definitions
        tools = self.get_tool_definitions() if self.tools else None

        # ReAct loop
        iteration = 0
        final_response: Msg | None = None

        while iteration < self.max_iterations:
            iteration += 1
            logger.debug(f"ReAct iteration {iteration}/{self.max_iterations}")

            # Call model
            if not self.model:
                raise RuntimeError("No model configured for agent")

            response = await self.model.chat(
                messages=messages,
                tools=tools,
                max_tokens=self.max_tokens,
                **kwargs,
            )

            # Convert response to Msg
            response_msg = response.to_msg(name=self.name)

            # Check for tool calls
            if response.has_tool_calls():
                # Add assistant message to conversation
                messages.append(response_msg)
                await self.observe(response_msg)

                # Execute tools
                tool_results = await self._execute_tools(response.tool_calls)

                # Add tool results to conversation
                tool_result_msg = self._create_tool_result_msg(tool_results)
                messages.append(tool_result_msg)
                await self.observe(tool_result_msg)

            else:
                # No tool calls - final response
                final_response = response_msg
                break

        # If we hit max iterations without final response
        if final_response is None:
            final_response = Msg(
                name=self.name,
                content="I've reached the maximum number of reasoning steps. Here's what I've accomplished so far.",
                role="assistant",
            )

        # Add final response to memory
        await self.observe(final_response)

        # Run post-reply hooks
        result = await self._run_post_reply_hooks(final_response)
        return result or final_response

    async def _build_conversation(self) -> list[Msg]:
        """Build conversation from memory and system prompt."""
        messages = []

        # Add system message
        sys_msg = self._build_system_message()
        if sys_msg:
            messages.append(sys_msg)

        # Add memory messages
        memory_messages = await self.get_memory_messages()
        messages.extend(memory_messages)

        return messages

    async def _execute_tools(
        self, tool_calls: list["ToolCall"]
    ) -> list[dict[str, Any]]:
        """Execute tool calls and collect results."""
        from clawscope.model import ToolCall

        results = []
        for tc in tool_calls:
            try:
                logger.info(f"Executing tool: {tc.name}")
                result = await self.execute_tool(tc.name, tc.arguments)
                results.append({
                    "tool_use_id": tc.id,
                    "content": result,
                    "is_error": False,
                })
            except Exception as e:
                logger.error(f"Tool execution error: {e}")
                results.append({
                    "tool_use_id": tc.id,
                    "content": f"Error: {str(e)}",
                    "is_error": True,
                })

        return results

    def _create_tool_result_msg(self, results: list[dict[str, Any]]) -> Msg:
        """Create a message containing tool results."""
        content_blocks = []
        for result in results:
            content_blocks.append(ToolResultBlock(
                tool_use_id=result["tool_use_id"],
                content=result["content"],
                is_error=result["is_error"],
            ))

        return Msg(
            name="tool",
            content=content_blocks,
            role="tool",
        )

    async def stream_reply(self, message: Msg | None = None, **kwargs: Any):
        """
        Stream a reply using ReAct loop.

        Yields partial responses as they come in.
        """
        # Run pre-reply hooks
        kwargs = await self._run_pre_reply_hooks(message=message, **kwargs)
        message = kwargs.pop("message", message)

        # Add input message to memory
        if message:
            await self.observe(message)

        # Build conversation
        messages = await self._build_conversation()

        # Get tool definitions
        tools = self.get_tool_definitions() if self.tools else None

        # ReAct loop with streaming
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1

            if not self.model:
                raise RuntimeError("No model configured for agent")

            # Stream from model
            content_buffer = ""
            tool_calls = []
            thinking_buffer = ""

            async for chunk in self.model.stream_chat(
                messages=messages,
                tools=tools,
                max_tokens=self.max_tokens,
                **kwargs,
            ):
                # Yield content chunks
                if chunk.content:
                    content_buffer += chunk.content
                    yield {
                        "type": "content",
                        "content": chunk.content,
                    }

                # Yield thinking chunks
                if chunk.thinking_content:
                    thinking_buffer += chunk.thinking_content
                    yield {
                        "type": "thinking",
                        "content": chunk.thinking_content,
                    }

                # Collect tool calls
                if chunk.tool_calls:
                    tool_calls = chunk.tool_calls

                # Check for finish
                if chunk.finish_reason:
                    break

            # Handle tool calls
            if tool_calls:
                # Create assistant message
                assistant_msg = Msg(
                    name=self.name,
                    content=[ToolUseBlock(
                        id=tc.id,
                        name=tc.name,
                        input=tc.arguments,
                    ) for tc in tool_calls],
                    role="assistant",
                )
                messages.append(assistant_msg)
                await self.observe(assistant_msg)

                # Execute tools once, collecting results and yielding events
                tool_results = []
                for tc in tool_calls:
                    yield {
                        "type": "tool_start",
                        "tool_name": tc.name,
                        "tool_id": tc.id,
                    }

                    try:
                        result = await self.execute_tool(tc.name, tc.arguments)
                        tool_results.append({
                            "tool_use_id": tc.id,
                            "content": result,
                            "is_error": False,
                        })
                        yield {
                            "type": "tool_result",
                            "tool_id": tc.id,
                            "content": result,
                            "is_error": False,
                        }
                    except Exception as e:
                        error_msg = str(e)
                        tool_results.append({
                            "tool_use_id": tc.id,
                            "content": error_msg,
                            "is_error": True,
                        })
                        yield {
                            "type": "tool_result",
                            "tool_id": tc.id,
                            "content": error_msg,
                            "is_error": True,
                        }

                # Add tool results to conversation
                tool_result_msg = self._create_tool_result_msg(tool_results)
                messages.append(tool_result_msg)
                await self.observe(tool_result_msg)

            else:
                # Final response
                final_msg = Msg(
                    name=self.name,
                    content=content_buffer,
                    role="assistant",
                )
                await self.observe(final_msg)

                yield {
                    "type": "done",
                    "message": final_msg,
                }
                break


__all__ = ["ReActAgent", "CompressionConfig"]
