"""FastAPI server for ClawScope."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger

from clawscope.app import ClawScope
from clawscope.config import Config
from clawscope._version import __version__


# ==================== Request/Response Models ====================

class ChatRequest(BaseModel):
    """Chat request model."""

    message: str = Field(..., description="User message")
    agent: str = Field(default="default", description="Agent to use")
    session_id: str | None = Field(default=None, description="Session ID")


class ChatResponse(BaseModel):
    """Chat response model."""

    response: str
    agent: str
    session_id: str | None = None


class AgentCreateRequest(BaseModel):
    """Agent creation request."""

    name: str
    system_prompt: str | None = None
    model: str = "gpt-4"


class KnowledgeAddRequest(BaseModel):
    """Add knowledge request."""

    content: str
    source: str | None = None
    kb_name: str = "default"


class KnowledgeSearchRequest(BaseModel):
    """Search knowledge request."""

    query: str
    top_k: int = 5
    kb_name: str = "default"


class PipelineRequest(BaseModel):
    """Pipeline execution request."""

    agents: list[str]
    message: str


class HubRequest(BaseModel):
    """MsgHub execution request."""

    agents: list[str]
    initial_message: str
    max_rounds: int = 10


class SkillInstallRequest(BaseModel):
    """Skill installation request."""

    name: str


class StatusResponse(BaseModel):
    """Platform status response."""

    version: str
    running: bool
    agents: list[str]
    knowledge_bases: list[str]
    skills: list[str]


# ==================== Global State ====================

_app_instance: ClawScope | None = None


def get_app() -> ClawScope:
    """Get ClawScope instance."""
    if _app_instance is None:
        raise HTTPException(status_code=503, detail="ClawScope not initialized")
    return _app_instance


# ==================== FastAPI App ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global _app_instance

    # Startup – create a default instance only when none was pre-configured
    if _app_instance is None:
        logger.info("Starting ClawScope API server...")
        _app_instance = ClawScope.create()
    await _app_instance.start()

    yield

    # Shutdown
    logger.info("Stopping ClawScope API server...")
    if _app_instance:
        await _app_instance.stop()


def create_api(
    clawscope_app: ClawScope | None = None,
    title: str = "ClawScope API",
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """
    Create FastAPI application.

    Args:
        clawscope_app: Optional pre-configured ClawScope instance
        title: API title
        cors_origins: CORS allowed origins

    Returns:
        FastAPI application
    """
    global _app_instance

    if clawscope_app:
        _app_instance = clawscope_app

    app = FastAPI(
        title=title,
        description="ClawScope - Unified AI Agent Platform API",
        version=__version__,
        lifespan=lifespan,
    )

    # CORS
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Register routes
    _register_routes(app)

    return app


def _register_routes(app: FastAPI) -> None:
    """Register API routes."""

    # ==================== Health & Status ====================

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy", "version": __version__}

    @app.get("/status", response_model=StatusResponse)
    async def status():
        """Get platform status."""
        clawscope = get_app()
        stats = clawscope.get_stats()
        return StatusResponse(**stats)

    # ==================== Chat ====================

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        """
        Send a chat message.

        Returns the agent's response.
        """
        clawscope = get_app()

        try:
            response = await clawscope.chat(
                message=request.message,
                agent_name=request.agent,
                session_id=request.session_id,
            )
            return ChatResponse(
                response=response,
                agent=request.agent,
                session_id=request.session_id,
            )
        except Exception as e:
            logger.error(f"Chat error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/chat/stream")
    async def chat_stream(request: ChatRequest):
        """
        Stream chat response (SSE).

        Each event is a JSON-encoded chunk.  Possible ``type`` values:
        ``content``, ``thinking``, ``tool_start``, ``tool_result``, ``done``, ``error``.
        The stream ends with the sentinel line ``data: [DONE]``.
        """
        import json
        from fastapi.responses import StreamingResponse

        clawscope = get_app()

        async def generate():
            try:
                async for chunk in clawscope.stream_chat(
                    message=request.message,
                    agent_name=request.agent,
                    session_id=request.session_id,
                ):
                    # ``done`` chunk contains a Msg object – serialise it
                    if chunk.get("type") == "done" and "message" in chunk:
                        msg = chunk["message"]
                        serialisable = {
                            "type": "done",
                            "content": msg.get_text_content() if msg else "",
                        }
                        yield f"data: {json.dumps(serialisable)}\n\n"
                    else:
                        yield f"data: {json.dumps(chunk)}\n\n"
            except Exception as e:
                logger.error(f"Stream chat error: {e}")
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            finally:
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ==================== Agents ====================

    @app.get("/agents")
    async def list_agents():
        """List registered agents."""
        clawscope = get_app()
        return {"agents": list(clawscope._agents.keys())}

    @app.post("/agents")
    async def create_agent(request: AgentCreateRequest):
        """Create and register a new agent."""
        from clawscope.app import create_agent as make_agent

        clawscope = get_app()

        agent = make_agent(
            name=request.name,
            model=request.model,
            system_prompt=request.system_prompt,
        )
        clawscope.register_agent(request.name, agent)

        return {"status": "created", "name": request.name}

    # ==================== Multi-Agent ====================

    @app.post("/pipeline")
    async def run_pipeline(request: PipelineRequest):
        """Run agents in a sequential pipeline."""
        clawscope = get_app()

        try:
            result = await clawscope.run_pipeline(
                agent_names=request.agents,
                input_message=request.message,
            )
            return {"result": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/hub")
    async def run_hub(request: HubRequest):
        """Run multi-agent hub conversation."""
        clawscope = get_app()

        try:
            hub = clawscope.create_hub(
                agent_names=request.agents,
                max_rounds=request.max_rounds,
            )

            from clawscope.message import Msg
            initial = Msg(name="user", content=request.initial_message, role="user")
            messages = await hub.run(initial_message=initial)

            return {
                "messages": [
                    {"name": m.name, "content": m.get_text_content()}
                    for m in messages
                ]
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ==================== Knowledge (RAG) ====================

    @app.post("/knowledge")
    async def add_knowledge(request: KnowledgeAddRequest):
        """Add knowledge to the knowledge base."""
        clawscope = get_app()

        try:
            chunks = await clawscope.add_knowledge(
                content=request.content,
                source=request.source,
                kb_name=request.kb_name,
            )
            return {"status": "added", "chunks": chunks}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/knowledge/search")
    async def search_knowledge(request: KnowledgeSearchRequest):
        """Search the knowledge base."""
        clawscope = get_app()

        results = await clawscope.search_knowledge(
            query=request.query,
            top_k=request.top_k,
            kb_name=request.kb_name,
        )
        return {"results": results}

    @app.get("/knowledge/bases")
    async def list_knowledge_bases():
        """List knowledge bases."""
        clawscope = get_app()
        return {"knowledge_bases": list(clawscope._knowledge_bases.keys())}

    # ==================== Skills ====================

    @app.get("/skills")
    async def list_skills():
        """List installed skills."""
        clawscope = get_app()
        return {"skills": clawscope.list_skills()}

    @app.post("/skills/install")
    async def install_skill(request: SkillInstallRequest):
        """Install a skill from the marketplace."""
        clawscope = get_app()

        success = await clawscope.install_skill(request.name)
        if success:
            return {"status": "installed", "name": request.name}
        else:
            raise HTTPException(status_code=400, detail="Installation failed")


# ==================== Server Runner ====================

def run_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    config_path: str | None = None,
    reload: bool = False,
):
    """
    Run the API server.

    Args:
        host: Host to bind
        port: Port to listen on
        config_path: Path to config file
        reload: Enable auto-reload
    """
    import uvicorn

    global _app_instance

    # Create ClawScope instance
    if config_path:
        _app_instance = ClawScope.from_config(config_path)
    else:
        _app_instance = ClawScope.create()

    # Create FastAPI app
    api = create_api(clawscope_app=_app_instance)

    # Run server
    uvicorn.run(
        api,
        host=host,
        port=port,
        reload=reload,
    )


# Create default app instance
api = create_api()

__all__ = [
    "create_api",
    "run_server",
    "api",
    "ChatRequest",
    "ChatResponse",
]
