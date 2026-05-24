import asyncio
import json
from typing import Any, AsyncIterator

import httpx

from clawscope.app import ClawScope
from clawscope.config import Config
from clawscope.message import Msg
from clawscope.server import create_api


class StubStreamingAgent:
    def __init__(self) -> None:
        self.name = "stub-stream"

    async def stream_reply(
        self,
        message: Msg,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        yield {"type": "content", "content": "hel"}
        yield {"type": "content", "content": "lo"}
        yield {"type": "done", "message": Msg(name="stub", content="hello", role="assistant")}


def _parse_sse(text: str) -> list[tuple[str, str]]:
    events: list[tuple[str, str]] = []
    for chunk in text.strip().split("\n\n"):
        event_type = "message"
        data = ""
        for line in chunk.splitlines():
            if line.startswith("event: "):
                event_type = line[len("event: ") :]
            elif line.startswith("data: "):
                data = line[len("data: ") :]
        events.append((event_type, data))
    return events


def test_chat_stream_endpoint_returns_structured_sse_events() -> None:
    async def run_test() -> None:
        clawscope = ClawScope(Config())
        clawscope.register_agent("default", StubStreamingAgent())  # type: ignore[arg-type]
        api = create_api(clawscope_app=clawscope)

        transport = httpx.ASGITransport(app=api)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            async with client.stream(
                "POST",
                "/chat/stream",
                json={"message": "hi", "agent": "default", "session_id": "stream-session"},
            ) as response:
                body = ""
                async for chunk in response.aiter_text():
                    body += chunk

        assert response.status_code == 200

        events = _parse_sse(body)
        assert [event_type for event_type, _ in events] == ["content", "content", "done", "end"]
        assert json.loads(events[0][1]) == {"type": "content", "content": "hel"}
        assert json.loads(events[1][1]) == {"type": "content", "content": "lo"}
        assert json.loads(events[2][1]) == {"type": "done", "content": "hello"}
        assert events[3][1] == "[DONE]"

    asyncio.run(run_test())


def test_chat_stream_endpoint_returns_structured_error_event() -> None:
    async def run_test() -> None:
        api = create_api(clawscope_app=ClawScope(Config()))

        transport = httpx.ASGITransport(app=api)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            async with client.stream(
                "POST",
                "/chat/stream",
                json={"message": "hi", "agent": "missing"},
            ) as response:
                body = ""
                async for chunk in response.aiter_text():
                    body += chunk

        assert response.status_code == 200

        events = _parse_sse(body)
        assert len(events) == 1
        assert events[0][0] == "error"
        assert json.loads(events[0][1]) == {
            "type": "error",
            "error": "Agent not found: missing",
        }

    asyncio.run(run_test())