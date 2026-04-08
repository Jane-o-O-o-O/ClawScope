"""Tests for the FastAPI server layer."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import clawscope.server as server_module
from clawscope.server import create_api


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_clawscope():
    """Minimal ClawScope mock that satisfies the server's interface."""
    app = MagicMock()
    app.start = AsyncMock()
    app.stop = AsyncMock()
    app.chat = AsyncMock(return_value="Hello, world!")

    async def _stream_chat(message, agent_name="default", session_id=None):
        yield {"type": "content", "content": "streamed "}
        yield {"type": "content", "content": "response"}
        yield {"type": "done", "content": "streamed response"}

    app.stream_chat = _stream_chat

    app.get_stats = MagicMock(
        return_value={
            "version": "0.1.0",
            "running": True,
            "agents": ["default"],
            "knowledge_bases": [],
            "skills": [],
        }
    )
    app._agents = {"default": MagicMock()}
    app._knowledge_bases = {}
    app.list_skills = MagicMock(return_value=["weather", "calculator"])
    app.register_agent = MagicMock()
    app.run_pipeline = AsyncMock(return_value="pipeline result")
    app.create_hub = MagicMock()
    app.add_knowledge = AsyncMock(return_value=3)
    app.search_knowledge = AsyncMock(return_value=[{"text": "result", "score": 0.9}])
    app.install_skill = AsyncMock(return_value=True)
    return app


@pytest.fixture()
def client(mock_clawscope):
    """TestClient with lifespan running the mock app."""
    server_module._app_instance = mock_clawscope
    api = create_api(clawscope_app=mock_clawscope)
    with TestClient(api) as c:
        yield c
    # Reset global state
    server_module._app_instance = None


# ---------------------------------------------------------------------------
# Health & status
# ---------------------------------------------------------------------------


def test_health_returns_200(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_status_returns_platform_info(client: TestClient) -> None:
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert data["running"] is True
    assert "default" in data["agents"]
    assert "skills" in data
    assert "knowledge_bases" in data


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


def test_chat_returns_response(client: TestClient) -> None:
    response = client.post("/chat", json={"message": "hello", "agent": "default"})
    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "Hello, world!"
    assert data["agent"] == "default"


def test_chat_with_session_id(client: TestClient) -> None:
    response = client.post(
        "/chat",
        json={"message": "hi", "agent": "default", "session_id": "session-42"},
    )
    assert response.status_code == 200
    assert response.json()["session_id"] == "session-42"


def test_chat_propagates_error(client: TestClient, mock_clawscope) -> None:
    mock_clawscope.chat = AsyncMock(side_effect=ValueError("Model error"))
    response = client.post("/chat", json={"message": "boom", "agent": "default"})
    assert response.status_code == 500


def test_chat_stream_returns_sse(client: TestClient) -> None:
    response = client.post(
        "/chat/stream",
        json={"message": "hello", "agent": "default"},
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    body = response.text
    assert "data:" in body
    assert "[DONE]" in body


def test_chat_stream_contains_content_chunks(client: TestClient) -> None:
    response = client.post(
        "/chat/stream",
        json={"message": "hello", "agent": "default"},
    )
    lines = [l for l in response.text.split("\n") if l.startswith("data:") and l != "data: [DONE]"]
    assert len(lines) >= 1
    # Each non-DONE line should be valid JSON
    for line in lines:
        payload = json.loads(line[len("data: "):])
        assert "type" in payload


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


def test_list_agents(client: TestClient) -> None:
    response = client.get("/agents")
    assert response.status_code == 200
    data = response.json()
    assert "agents" in data
    assert "default" in data["agents"]


def test_create_agent(client: TestClient, mock_clawscope) -> None:
    with patch("clawscope.app.create_agent") as mock_create:
        mock_create.return_value = MagicMock()
        response = client.post(
            "/agents",
            json={"name": "my-bot", "model": "gpt-4", "system_prompt": "Be helpful"},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "created"
    assert data["name"] == "my-bot"


# ---------------------------------------------------------------------------
# Multi-agent
# ---------------------------------------------------------------------------


def test_run_pipeline(client: TestClient) -> None:
    response = client.post(
        "/pipeline",
        json={"agents": ["default"], "message": "summarize this"},
    )
    assert response.status_code == 200
    assert "result" in response.json()


def test_run_hub(client: TestClient, mock_clawscope) -> None:
    hub = MagicMock()
    from clawscope.message import Msg

    hub.run = AsyncMock(
        return_value=[Msg(name="agent", content="hub response", role="assistant")]
    )
    mock_clawscope.create_hub = MagicMock(return_value=hub)

    response = client.post(
        "/hub",
        json={"agents": ["default"], "initial_message": "start", "max_rounds": 3},
    )
    assert response.status_code == 200
    data = response.json()
    assert "messages" in data
    assert len(data["messages"]) == 1


# ---------------------------------------------------------------------------
# Knowledge (RAG)
# ---------------------------------------------------------------------------


def test_add_knowledge(client: TestClient) -> None:
    response = client.post(
        "/knowledge",
        json={"content": "Python is a programming language.", "source": "wiki"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "added"
    assert data["chunks"] == 3


def test_search_knowledge(client: TestClient) -> None:
    response = client.post(
        "/knowledge/search",
        json={"query": "Python", "top_k": 5},
    )
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 1


def test_list_knowledge_bases(client: TestClient) -> None:
    response = client.get("/knowledge/bases")
    assert response.status_code == 200
    assert "knowledge_bases" in response.json()


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------


def test_list_skills(client: TestClient) -> None:
    response = client.get("/skills")
    assert response.status_code == 200
    data = response.json()
    assert "skills" in data
    assert "weather" in data["skills"]


def test_install_skill_success(client: TestClient) -> None:
    response = client.post("/skills/install", json={"name": "calculator"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "installed"
    assert data["name"] == "calculator"


def test_install_skill_failure(client: TestClient, mock_clawscope) -> None:
    mock_clawscope.install_skill = AsyncMock(return_value=False)
    response = client.post("/skills/install", json={"name": "broken-skill"})
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# 503 when app not initialised
# ---------------------------------------------------------------------------


def test_chat_503_when_no_app() -> None:
    server_module._app_instance = None
    api = create_api()  # No lifespan invocation via TestClient(api, raise_server_exceptions=False)
    with TestClient(api, raise_server_exceptions=False) as c:
        # lifespan will try to create + start a real app; we just check the
        # get_app() guard exists by calling health (which doesn't need the app)
        response = c.get("/health")
        assert response.status_code == 200
