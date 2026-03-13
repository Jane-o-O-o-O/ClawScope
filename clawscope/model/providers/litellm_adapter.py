"""LiteLLM adapter for additional model providers."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from loguru import logger

from clawscope.model.base import ChatModelBase, ChatResponse, ToolCall, UsageInfo
from clawscope.message import Msg


class LiteLLMChatModel(ChatModelBase):
    """
    LiteLLM adapter for 30+ model providers.

    Supports providers like OpenRouter, DeepSeek, Groq,
    SiliconFlow, Moonshot, Zhipu, and many more.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            **kwargs,
        )

    async def chat(
        self,
        messages: list[Msg],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send chat request via LiteLLM."""
        try:
            import litellm
        except ImportError:
            raise ImportError("LiteLLM is required. Install with: pip install litellm")

        # Format messages
        formatted_messages = self._format_messages(messages)

        # Build request parameters
        params: dict[str, Any] = {
            "model": self.model_name,
            "messages": formatted_messages,
            "api_key": self.api_key,
            "timeout": self.timeout,
            **kwargs,
        }

        if self.api_base:
            params["api_base"] = self.api_base

        if tools:
            params["tools"] = tools
            if tool_choice:
                params["tool_choice"] = self._validate_tool_choice(tool_choice, tools)

        try:
            response = await litellm.acompletion(**params)

            # Parse response
            choice = response.choices[0]
            message = choice.message

            # Parse tool calls
            tool_calls = []
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tc in message.tool_calls:
                    args = tc.function.arguments
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {"raw": args}

                    tool_calls.append(ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=args,
                    ))

            # Parse usage
            usage = UsageInfo()
            if hasattr(response, "usage") and response.usage:
                usage = UsageInfo(
                    prompt_tokens=getattr(response.usage, "prompt_tokens", 0),
                    completion_tokens=getattr(response.usage, "completion_tokens", 0),
                    total_tokens=getattr(response.usage, "total_tokens", 0),
                )

            # Check for extended thinking (DeepSeek-R1, Kimi, etc.)
            thinking_content = None
            if hasattr(message, "reasoning_content"):
                thinking_content = message.reasoning_content

            return ChatResponse(
                content=message.content,
                tool_calls=tool_calls,
                finish_reason=choice.finish_reason,
                usage=usage,
                thinking_content=thinking_content,
                raw_response=response,
            )

        except Exception as e:
            logger.error(f"LiteLLM API error: {e}")
            raise

    async def stream_chat(
        self,
        messages: list[Msg],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatResponse]:
        """Stream chat response via LiteLLM."""
        try:
            import litellm
        except ImportError:
            raise ImportError("LiteLLM is required. Install with: pip install litellm")

        # Format messages
        formatted_messages = self._format_messages(messages)

        # Build request parameters
        params: dict[str, Any] = {
            "model": self.model_name,
            "messages": formatted_messages,
            "api_key": self.api_key,
            "timeout": self.timeout,
            "stream": True,
            **kwargs,
        }

        if self.api_base:
            params["api_base"] = self.api_base

        if tools:
            params["tools"] = tools
            if tool_choice:
                params["tool_choice"] = self._validate_tool_choice(tool_choice, tools)

        try:
            response = await litellm.acompletion(**params)

            content_buffer = ""
            tool_calls_buffer: dict[int, dict] = {}

            async for chunk in response:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # Accumulate content
                if hasattr(delta, "content") and delta.content:
                    content_buffer += delta.content
                    yield ChatResponse(
                        content=delta.content,
                        finish_reason=None,
                    )

                # Accumulate tool calls
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_buffer:
                            tool_calls_buffer[idx] = {
                                "id": tc.id or "",
                                "name": tc.function.name if tc.function else "",
                                "arguments": "",
                            }
                        if tc.function and tc.function.arguments:
                            tool_calls_buffer[idx]["arguments"] += tc.function.arguments

                # Check for finish
                finish_reason = chunk.choices[0].finish_reason
                if finish_reason:
                    # Parse accumulated tool calls
                    tool_calls = []
                    for tc_data in tool_calls_buffer.values():
                        args = tc_data["arguments"]
                        try:
                            args = json.loads(args) if args else {}
                        except json.JSONDecodeError:
                            args = {"raw": args}

                        tool_calls.append(ToolCall(
                            id=tc_data["id"],
                            name=tc_data["name"],
                            arguments=args,
                        ))

                    yield ChatResponse(
                        content=content_buffer if not tool_calls else None,
                        tool_calls=tool_calls,
                        finish_reason=finish_reason,
                    )

        except Exception as e:
            logger.error(f"LiteLLM streaming error: {e}")
            raise

    def _format_messages(self, messages: list[Msg]) -> list[dict[str, Any]]:
        """Format messages for LiteLLM (OpenAI-compatible format)."""
        formatted = []

        for msg in messages:
            content = msg.content

            # Handle content blocks
            if isinstance(content, list):
                openai_content = []
                for block in content:
                    if hasattr(block, "to_dict"):
                        block_dict = block.to_dict()
                    else:
                        block_dict = block

                    block_type = block_dict.get("type", "text")

                    if block_type == "text":
                        openai_content.append({
                            "type": "text",
                            "text": block_dict.get("text", ""),
                        })
                    elif block_type == "image":
                        source = block_dict.get("source", {})
                        if source.get("type") == "url":
                            openai_content.append({
                                "type": "image_url",
                                "image_url": {"url": source.get("url", "")},
                            })
                        elif source.get("type") == "base64":
                            media_type = source.get("media_type", "image/png")
                            data = source.get("data", "")
                            openai_content.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:{media_type};base64,{data}"},
                            })

                content = openai_content if openai_content else msg.get_text_content()

            message_dict: dict[str, Any] = {
                "role": msg.role,
                "content": content,
            }

            # Add name for user messages
            if msg.role == "user" and msg.name:
                message_dict["name"] = msg.name

            # Handle tool role
            if msg.role == "tool":
                if isinstance(msg.content, list):
                    for block in msg.content:
                        if hasattr(block, "tool_use_id"):
                            message_dict["tool_call_id"] = block.tool_use_id
                            message_dict["content"] = getattr(block, "content", "")
                            break

            formatted.append(message_dict)

        return formatted


__all__ = ["LiteLLMChatModel"]
