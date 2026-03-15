import asyncio
from typing import Any

from clawscope.app import ClawScope
from clawscope.config import Config
from clawscope.message import Msg


class StubStreamingAgent:
    def __init__(self) -> None:
        self.name = "stub"
        self.seen_messages: list[Msg] = []

    async def stream_reply(self, message: Msg, **kwargs: Any):
        self.seen_messages.append(message)
        yield {"type": "content", "content": "hel"}
        yield {"type": "content", "content": "lo"}
        yield {"type": "done", "message": Msg(name="stub", content="hello", role="assistant")}


def test_clawscope_stream_chat_uses_agent_stream_reply() -> None:
    async def run_test() -> None:
        app = ClawScope(Config())
        agent = StubStreamingAgent()
        app.register_agent("default", agent)  # type: ignore[arg-type]

        events = [
            event
            async for event in app.stream_chat(
                message="hi",
                agent_name="default",
                session_id="stream-session",
            )
        ]

        assert [event["type"] for event in events] == ["content", "content", "done"]
        assert agent.seen_messages[0].metadata["_runtime_context"] is True
        assert "hi" in agent.seen_messages[0].get_text_content()

    asyncio.run(run_test())
