from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, AsyncIterator

from clawscope.config import AgentConfig, ToolsConfig
from clawscope.kernel import NativeAgentLoop, NativeKernel
from clawscope.memory import InMemoryMemory
from clawscope.message import Msg
from clawscope.model import ChatModelBase, ChatResponse, ToolCall
from clawscope.tool import ToolRegistry


class StubModel(ChatModelBase):
    def __init__(self, responses: list[ChatResponse]):
        super().__init__(model_name="stub", stream=True)
        self._responses = responses
        self.chat_calls: list[list[Msg]] = []

    async def chat(
        self,
        messages: list[Msg],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        self.chat_calls.append(list(messages))
        return self._responses.pop(0)

    async def stream_chat(
        self,
        messages: list[Msg],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatResponse]:
        if False:
            yield ChatResponse()
        return


class StubStreamingModel(ChatModelBase):
    def __init__(self, streams: list[list[ChatResponse]]):
        super().__init__(model_name="stub-stream", stream=True)
        self._streams = streams
        self.stream_calls: list[list[Msg]] = []

    async def chat(
        self,
        messages: list[Msg],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        raise NotImplementedError

    async def stream_chat(
        self,
        messages: list[Msg],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatResponse]:
        self.stream_calls.append(list(messages))
        for chunk in self._streams.pop(0):
            yield chunk


class StubModelRegistry:
    def __init__(self, model: ChatModelBase):
        self._model = model

    def get_model(self) -> ChatModelBase:
        return self._model


def test_native_kernel_owns_reasoning_loop() -> None:
    async def run_test() -> None:
        model = StubModel(
            responses=[
                ChatResponse(
                    tool_calls=[
                        ToolCall(
                            id="tool-1",
                            name="lookup_weather",
                            arguments={"city": "Shanghai"},
                        )
                    ],
                    finish_reason="tool_calls",
                ),
                ChatResponse(content="It is sunny.", finish_reason="stop"),
            ]
        )
        tool_registry = ToolRegistry(ToolsConfig())

        async def lookup_weather(city: str) -> str:
            return f"{city}: sunny"

        tool_registry.register_function(lookup_weather)

        kernel = NativeKernel(
            agent_config=AgentConfig(name="KernelAgent", max_iterations=3, max_tokens=512),
            tool_registry=tool_registry,
            model_registry=StubModelRegistry(model),
            workspace=Path.cwd(),
        )

        agent = kernel.create_agent(memory=InMemoryMemory(), max_iterations=2)

        assert isinstance(agent.loop, NativeAgentLoop)
        assert agent.max_iterations == 2
        assert agent.max_tokens == 512
        assert agent.loop.config.max_iterations == 2
        assert agent.loop.config.max_tokens == 512

        response = await agent.reply(Msg(name="user", content="weather?", role="user"))

        assert response.get_text_content() == "It is sunny."
        assert len(model.chat_calls) == 2
        assert any(msg.role == "tool" for msg in model.chat_calls[1])

        memory_messages = await agent.get_memory_messages()
        assert [msg.role for msg in memory_messages] == [
            "user",
            "assistant",
            "tool",
            "assistant",
        ]

    asyncio.run(run_test())


def test_native_kernel_stream_executes_each_tool_once() -> None:
    async def run_test() -> None:
        model = StubStreamingModel(
            streams=[
                [
                    ChatResponse(
                        tool_calls=[
                            ToolCall(
                                id="tool-1",
                                name="lookup_weather",
                                arguments={"city": "Shanghai"},
                            )
                        ],
                        finish_reason="tool_calls",
                    )
                ],
                [
                    ChatResponse(content="It is sunny.", finish_reason="stop"),
                ],
            ]
        )
        tool_registry = ToolRegistry(ToolsConfig())
        executions: list[str] = []

        async def lookup_weather(city: str) -> str:
            executions.append(city)
            return f"{city}: sunny"

        tool_registry.register_function(lookup_weather)

        kernel = NativeKernel(
            agent_config=AgentConfig(name="KernelAgent", max_iterations=3),
            tool_registry=tool_registry,
            model_registry=StubModelRegistry(model),
            workspace=Path.cwd(),
        )
        agent = kernel.create_agent(memory=InMemoryMemory())

        events = [
            event
            async for event in agent.stream_reply(
                Msg(name="user", content="weather?", role="user")
            )
        ]

        assert executions == ["Shanghai"]
        assert [event["type"] for event in events] == [
            "tool_start",
            "tool_result",
            "content",
            "done",
        ]
        assert events[-1]["message"].get_text_content() == "It is sunny."

    asyncio.run(run_test())
