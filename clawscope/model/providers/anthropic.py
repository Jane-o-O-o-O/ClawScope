"""Anthropic/Claude chat model implementation."""

from __future__ import annotations

from typing import Any, AsyncIterator

from loguru import logger

from clawscope.model.base import ChatModelBase, ChatResponse, ToolCall, UsageInfo
from clawscope.message import Msg


class AnthropicChatModel(ChatModelBase):
    """
    Anthropic Claude chat model implementation.

    Supports Claude 3, Claude 3.5, and Claude 4 models with
    extended thinking capabilities.
    """

    def __init__(
        self,
        model_name: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        api_base: str | None = None,
        max_tokens: int = 4096,
        **kwargs: Any,
    ):
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            **kwargs,
        )
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        """Get or create Anthropic client."""
        if self._client is None:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(
                api_key=self.api_key,
                base_url=self.api_base,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return self._client

    async def chat(
        self,
        messages: list[Msg],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send chat request to Anthropic."""
        client = self._get_client()

        # Extract system message and format others
        system_prompt, formatted_messages = self._format_messages(messages)

        # Build request parameters
        params: dict[str, Any] = {
            "model": self.model_name,
            "messages": formatted_messages,
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
            **kwargs,
        }

        if system_prompt:
            params["system"] = system_prompt

        # Convert tools to Anthropic format
        if tools:
            params["tools"] = self._convert_tools(tools)
            if tool_choice:
                params["tool_choice"] = self._convert_tool_choice(tool_choice)

        try:
            response = await client.messages.create(**params)

            # Parse response
            content = ""
            tool_calls = []
            thinking_content = ""
            thinking_blocks = []

            for block in response.content:
                if block.type == "text":
                    content += block.text
                elif block.type == "thinking":
                    thinking_content += block.thinking
                    thinking_blocks.append({
                        "type": "thinking",
                        "thinking": block.thinking,
                    })
                elif block.type == "tool_use":
                    tool_calls.append(ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input,
                    ))

            # Parse usage
            usage = UsageInfo()
            if response.usage:
                usage = UsageInfo(
                    prompt_tokens=response.usage.input_tokens,
                    completion_tokens=response.usage.output_tokens,
                    total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                    cached_tokens=getattr(response.usage, "cache_read_input_tokens", 0),
                )

            # Determine finish reason
            finish_reason = "stop"
            if response.stop_reason == "tool_use":
                finish_reason = "tool_calls"
            elif response.stop_reason == "max_tokens":
                finish_reason = "length"

            return ChatResponse(
                content=content or None,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                usage=usage,
                thinking_content=thinking_content or None,
                thinking_blocks=thinking_blocks,
                raw_response=response,
            )

        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    async def stream_chat(
        self,
        messages: list[Msg],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatResponse]:
        """Stream chat response from Anthropic."""
        client = self._get_client()

        # Extract system message and format others
        system_prompt, formatted_messages = self._format_messages(messages)

        # Build request parameters
        params: dict[str, Any] = {
            "model": self.model_name,
            "messages": formatted_messages,
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
            **kwargs,
        }

        if system_prompt:
            params["system"] = system_prompt

        if tools:
            params["tools"] = self._convert_tools(tools)
            if tool_choice:
                params["tool_choice"] = self._convert_tool_choice(tool_choice)

        try:
            async with client.messages.stream(**params) as stream:
                content_buffer = ""
                thinking_buffer = ""
                current_tool_id = None
                current_tool_name = None
                tool_input_buffer = ""
                tool_calls = []

                async for event in stream:
                    if event.type == "content_block_start":
                        block = event.content_block
                        if block.type == "tool_use":
                            current_tool_id = block.id
                            current_tool_name = block.name
                            tool_input_buffer = ""

                    elif event.type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            content_buffer += delta.text
                            yield ChatResponse(
                                content=delta.text,
                                finish_reason=None,
                            )
                        elif delta.type == "thinking_delta":
                            thinking_buffer += delta.thinking
                            yield ChatResponse(
                                thinking_content=delta.thinking,
                                finish_reason=None,
                            )
                        elif delta.type == "input_json_delta":
                            tool_input_buffer += delta.partial_json

                    elif event.type == "content_block_stop":
                        if current_tool_id:
                            # Parse tool input
                            import json
                            try:
                                tool_input = json.loads(tool_input_buffer) if tool_input_buffer else {}
                            except json.JSONDecodeError:
                                tool_input = {"raw": tool_input_buffer}

                            tool_calls.append(ToolCall(
                                id=current_tool_id,
                                name=current_tool_name or "",
                                arguments=tool_input,
                            ))
                            current_tool_id = None
                            current_tool_name = None

                    elif event.type == "message_stop":
                        yield ChatResponse(
                            content=content_buffer if not tool_calls else None,
                            tool_calls=tool_calls,
                            thinking_content=thinking_buffer or None,
                            finish_reason="tool_calls" if tool_calls else "stop",
                        )

        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            raise

    def _format_messages(self, messages: list[Msg]) -> tuple[str, list[dict[str, Any]]]:
        """
        Format messages for Anthropic API.

        Returns:
            Tuple of (system_prompt, formatted_messages)
        """
        system_prompt = ""
        formatted = []

        for msg in messages:
            if msg.role == "system":
                system_prompt += msg.get_text_content() + "\n"
                continue

            content = msg.content

            # Handle content blocks
            if isinstance(content, list):
                anthropic_content = []
                for block in content:
                    if hasattr(block, "to_dict"):
                        block_dict = block.to_dict()
                    else:
                        block_dict = block

                    block_type = block_dict.get("type", "text")

                    if block_type == "text":
                        anthropic_content.append({
                            "type": "text",
                            "text": block_dict.get("text", ""),
                        })
                    elif block_type == "image":
                        source = block_dict.get("source", {})
                        anthropic_content.append({
                            "type": "image",
                            "source": source,
                        })
                    elif block_type == "tool_use":
                        anthropic_content.append({
                            "type": "tool_use",
                            "id": block_dict.get("id", ""),
                            "name": block_dict.get("name", ""),
                            "input": block_dict.get("input", {}),
                        })
                    elif block_type == "tool_result":
                        anthropic_content.append({
                            "type": "tool_result",
                            "tool_use_id": block_dict.get("tool_use_id", ""),
                            "content": block_dict.get("content", ""),
                            "is_error": block_dict.get("is_error", False),
                        })

                content = anthropic_content if anthropic_content else [{"type": "text", "text": msg.get_text_content()}]
            else:
                content = [{"type": "text", "text": content}]

            # Handle tool role -> user with tool_result
            role = msg.role
            if role == "tool":
                role = "user"

            formatted.append({
                "role": role,
                "content": content,
            })

        return system_prompt.strip(), formatted

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI tool format to Anthropic format."""
        anthropic_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                anthropic_tools.append({
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {}),
                })
            else:
                anthropic_tools.append(tool)
        return anthropic_tools

    def _convert_tool_choice(self, tool_choice: str | dict) -> dict[str, Any]:
        """Convert tool choice to Anthropic format."""
        if isinstance(tool_choice, str):
            if tool_choice == "auto":
                return {"type": "auto"}
            elif tool_choice == "none":
                return {"type": "none"}
            elif tool_choice == "required":
                return {"type": "any"}
            else:
                return {"type": "tool", "name": tool_choice}
        return tool_choice


__all__ = ["AnthropicChatModel"]
