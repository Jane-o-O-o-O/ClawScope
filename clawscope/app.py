"""ClawScope main application class."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from clawscope.conversation_context import attach_runtime_context
from clawscope.config import Config
from clawscope._version import __version__

if TYPE_CHECKING:
    from clawscope.agent import AgentBase, ReActAgent, A2ARouter
    from clawscope.bus import MessageBus
    from clawscope.channels import ChannelManager
    from clawscope.kernel import AgentKernel
    from clawscope.memory import SessionManager, UnifiedMemory
    from clawscope.model import ModelRegistry
    from clawscope.orchestration import SessionRouter, MsgHub, ChatRoom
    from clawscope.rag import KnowledgeBase
    from clawscope.sandbox import SandboxManager
    from clawscope.services import SchedulerService
    from clawscope.skills import SkillRegistry, SkillMarketplace
    from clawscope.tool import ToolRegistry
    from clawscope.tracing import Tracer


class ClawScope:
    """
    ClawScope - Unified AI Agent Platform.

    Main application class that orchestrates all components:
    - MessageBus for channel-agent communication
    - ChannelManager for multi-platform support
    - SessionRouter for message routing
    - Agent instances for conversation handling
    - RAG for knowledge retrieval
    - Multi-agent orchestration
    - Skills marketplace
    - Background services for scheduling
    - OpenTelemetry tracing
    """

    def __init__(self, config: Config):
        """
        Initialize ClawScope application.

        Args:
            config: Application configuration
        """
        self.config = config
        self.version = __version__

        # Core components
        self._bus: "MessageBus | None" = None
        self._model_registry: "ModelRegistry | None" = None
        self._tool_registry: "ToolRegistry | None" = None
        self._session_manager: "SessionManager | None" = None

        # Channel & routing
        self._channels: "ChannelManager | None" = None
        self._router: "SessionRouter | None" = None

        # Agent system
        self._default_agent: "AgentBase | None" = None
        self._agents: dict[str, "AgentBase"] = {}
        self._a2a_router: "A2ARouter | None" = None
        self._kernel: "AgentKernel | None" = None

        # RAG
        self._knowledge_bases: dict[str, "KnowledgeBase"] = {}

        # Skills
        self._skill_registry: "SkillRegistry | None" = None
        self._skill_marketplace: "SkillMarketplace | None" = None

        # Services
        self._scheduler: "SchedulerService | None" = None
        self._sandbox_manager: "SandboxManager | None" = None

        # Tracing
        self._tracer: "Tracer | None" = None

        # Runtime
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._hooks: dict[str, list[Callable]] = {}

    # ==================== Factory Methods ====================

    @classmethod
    def from_config(cls, path: str | Path) -> "ClawScope":
        """
        Create ClawScope instance from config file.

        Args:
            path: Path to YAML or JSON config file

        Returns:
            Configured ClawScope instance
        """
        config = Config.from_file(path)
        return cls(config)

    @classmethod
    def create(
        cls,
        model_provider: str = "openai",
        default_model: str = "gpt-4",
        api_key: str | None = None,
        workspace: Path | None = None,
        **kwargs,
    ) -> "ClawScope":
        """
        Create ClawScope with simple configuration.

        Args:
            model_provider: Model provider name
            default_model: Default model to use
            api_key: API key for the provider
            workspace: Workspace directory
            **kwargs: Additional config options

        Returns:
            Configured ClawScope instance
        """
        from clawscope.config import ModelConfig, Config

        model_config = ModelConfig(
            provider=model_provider,
            default_model=default_model,
            api_key=api_key,
        )

        config = Config(
            model=model_config,
            workspace=workspace or Path.home() / ".clawscope" / "workspace",
            **kwargs,
        )

        return cls(config)

    # ==================== Lifecycle ====================

    async def start(self) -> None:
        """Start the ClawScope platform."""
        if self._running:
            logger.warning("ClawScope is already running")
            return

        logger.info(f"Starting ClawScope v{self.version}...")

        # Ensure workspace exists
        self.config.ensure_workspace()

        # Initialize tracing first
        await self._init_tracing()

        # Initialize all components
        await self._init_core_components()
        await self._init_agent_system()
        await self._init_rag_system()
        await self._init_skills_system()
        await self._init_sandbox()
        await self._init_services()

        # Start all components
        await self._start_components()

        self._running = True
        await self._trigger_hook("on_start")
        logger.info("ClawScope platform started successfully")

    async def stop(self) -> None:
        """Stop the ClawScope platform."""
        if not self._running:
            return

        logger.info("Stopping ClawScope platform...")
        await self._trigger_hook("on_stop")

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks.clear()

        # Stop components
        await self._stop_components()

        # Shutdown tracing
        if self._tracer:
            self._tracer.shutdown()

        self._running = False
        logger.info("ClawScope platform stopped")

    async def run_forever(self) -> None:
        """Run the platform until interrupted."""
        await self.start()
        try:
            if self._tasks:
                await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    # ==================== Initialization ====================

    async def _init_tracing(self) -> None:
        """Initialize OpenTelemetry tracing."""
        if not self.config.tracing.enabled:
            return

        try:
            from clawscope.tracing import Tracer, TracingConfig

            tracing_config = TracingConfig(
                service_name=self.config.project,
                enabled=self.config.tracing.enabled,
                endpoint=self.config.tracing.endpoint,
            )

            self._tracer = Tracer(tracing_config)
            self._tracer.initialize()
            logger.info("OpenTelemetry tracing initialized")

        except ImportError:
            logger.debug("OpenTelemetry not available, tracing disabled")

    async def _init_core_components(self) -> None:
        """Initialize core components."""
        from clawscope.bus import MessageBus
        from clawscope.memory import SessionManager
        from clawscope.model import ModelRegistry
        from clawscope.tool import ToolRegistry

        # Message bus
        self._bus = MessageBus()

        # Model registry
        self._model_registry = ModelRegistry(self.config.model)

        # Tool registry
        self._tool_registry = ToolRegistry(self.config.tools)
        await self._tool_registry.load_builtin_tools()

        # Session manager
        self._session_manager = SessionManager(
            workspace=self.config.workspace,
            backend=self.config.memory.session,
        )

        logger.debug("Core components initialized")

    async def _init_agent_system(self) -> None:
        """Initialize agent system."""
        from clawscope.agent import A2ARouter, get_router
        from clawscope.channels import ChannelManager
        from clawscope.memory import InMemoryMemory
        from clawscope.kernel import build_kernel
        from clawscope.orchestration import SessionRouter

        self._kernel = build_kernel(
            agent_config=self.config.agent,
            model_config=self.config.model,
            model_registry=self._model_registry,
            tool_registry=self._tool_registry,
            workspace=self.config.workspace,
        )

        # Create default agent
        self._default_agent = self._kernel.create_agent(
            memory=InMemoryMemory(),
        )
        self._agents["default"] = self._default_agent

        # A2A router
        self._a2a_router = get_router()

        # Channel manager
        self._channels = ChannelManager(
            bus=self._bus,
            config=self.config.channels,
        )

        # Session router
        self._router = SessionRouter(
            bus=self._bus,
            sessions=self._session_manager,
            kernel=self._kernel,
            config=self.config.agent,
        )

        # Flush any sub-agents registered before start()
        pending = getattr(self, "_pending_sub_agents", {})
        for sub_name, sub_agent in pending.items():
            self._router.register_sub_agent(sub_name, sub_agent)
        if pending:
            logger.info(f"Flushed {len(pending)} pending sub-agent(s) to SessionRouter")

        logger.debug("Agent system initialized")

    async def _init_rag_system(self) -> None:
        """Initialize RAG system."""
        try:
            from clawscope.rag import KnowledgeBase, OpenAIEmbedding

            # Create default knowledge base
            embedding = OpenAIEmbedding(
                api_key=self.config.model.api_key,
            )

            kb_path = self.config.workspace / "knowledge"
            kb_path.mkdir(parents=True, exist_ok=True)

            self._knowledge_bases["default"] = KnowledgeBase(
                name="default",
                embedding=embedding,
                persist_path=kb_path,
            )

            logger.debug("RAG system initialized")

        except Exception as e:
            logger.debug(f"RAG initialization skipped: {e}")

    async def _init_skills_system(self) -> None:
        """Initialize skills system."""
        try:
            from clawscope.skills import SkillRegistry, SkillLoader, SkillMarketplace

            # Skill registry
            self._skill_registry = SkillRegistry()

            # Skill loader
            loader = SkillLoader(self._skill_registry)

            # Skill marketplace
            skills_dir = self.config.workspace / "skills"
            self._skill_marketplace = SkillMarketplace(
                registry=self._skill_registry,
                loader=loader,
                skills_dir=skills_dir,
            )

            # Load installed skills
            await self._skill_marketplace.load_installed()

            # Load local skills if exists
            local_skills = self.config.workspace / "skills" / "local"
            if local_skills.exists():
                await loader.load_directory(local_skills)

            logger.debug("Skills system initialized")

        except Exception as e:
            logger.debug(f"Skills initialization skipped: {e}")

    async def _init_sandbox(self) -> None:
        """Initialize sandbox system."""
        if not self.config.tools.sandbox_enabled:
            return

        try:
            from clawscope.sandbox import SandboxManager, SandboxConfig, configure_sandbox

            sandbox_config = SandboxConfig(
                enabled=True,
                image=self.config.tools.sandbox.image,
                memory_limit=self.config.tools.sandbox.memory_limit,
                cpu_limit=self.config.tools.sandbox.cpu_limit,
                network_enabled=self.config.tools.sandbox.network_enabled,
                workspace_path=self.config.workspace,
            )

            self._sandbox_manager = configure_sandbox(
                config=sandbox_config,
                workspace=self.config.workspace,
            )

            logger.debug("Sandbox system initialized")

        except Exception as e:
            logger.debug(f"Sandbox initialization skipped: {e}")

    async def _init_services(self) -> None:
        """Initialize background services."""
        from clawscope.services import SchedulerService

        if self.config.services.cron_enabled or self.config.services.heartbeat_enabled:
            self._scheduler = SchedulerService(
                workspace=self.config.workspace,
                bus=self._bus,
                config=self.config.services,
            )

        logger.debug("Services initialized")

    async def _start_components(self) -> None:
        """Start all components."""
        # Start channel manager
        if self._channels:
            task = asyncio.create_task(self._channels.start())
            self._tasks.append(task)

        # Start session router
        if self._router:
            task = asyncio.create_task(self._router.run())
            self._tasks.append(task)

        # Start scheduler
        if self._scheduler:
            task = asyncio.create_task(self._scheduler.start())
            self._tasks.append(task)

        # Start A2A router
        if self._a2a_router:
            task = asyncio.create_task(self._a2a_router.run())
            self._tasks.append(task)

    async def _stop_components(self) -> None:
        """Stop all components."""
        if self._channels:
            await self._channels.stop()

        if self._router:
            self._router.stop()

        if self._scheduler:
            await self._scheduler.stop()

        if self._a2a_router:
            self._a2a_router.stop()

        if self._sandbox_manager:
            await self._sandbox_manager.cleanup_all()

    # ==================== Agent Management ====================

    def register_agent(self, name: str, agent: "AgentBase") -> None:
        """
        Register an agent.

        Args:
            name: Agent name
            agent: Agent instance
        """
        self._agents[name] = agent

        # Also register with A2A router
        if self._a2a_router:
            from clawscope.agent import A2AAgent
            if not isinstance(agent, A2AAgent):
                a2a_agent = A2AAgent(agent, router=self._a2a_router)
            else:
                self._a2a_router.register(agent)

        logger.info(f"Registered agent: {name}")

    def register_sub_agent(
        self,
        agent: "AgentBase",
        name: str | None = None,
    ) -> None:
        """
        Register a sub-agent that the orchestrator will automatically delegate to.

        Once at least one sub-agent is registered, every new channel session
        will receive an :class:`~clawscope.agent.OrchestratorAgent` (instead of
        a plain ReActAgent) whose LLM autonomously decides which sub-agents to
        call based on the incoming message.

        Existing sessions are not affected; only sessions created *after* this
        call will use the orchestrator.

        Args:
            agent: The sub-agent to register.
            name: Logical key used as the tool name (``ask_<name>``).
                  Defaults to ``agent.name``.

        Example::

            researcher = ReActAgent(name="researcher", sys_prompt="...", ...)
            writer     = ReActAgent(name="writer",     sys_prompt="...", ...)

            app.register_sub_agent(researcher)
            app.register_sub_agent(writer)
            # Future channel messages are handled by OrchestratorAgent
        """
        key = name or agent.name
        if self._router is not None:
            self._router.register_sub_agent(key, agent)
        else:
            # Platform not started yet – store for later pickup
            if not hasattr(self, "_pending_sub_agents"):
                self._pending_sub_agents: dict[str, "AgentBase"] = {}
            self._pending_sub_agents[key] = agent  # type: ignore[attr-defined]

        logger.info(f"Registered sub-agent: '{key}'")

    def get_agent(self, name: str = "default") -> "AgentBase | None":
        """Get an agent by name."""
        return self._agents.get(name)

    async def chat(
        self,
        message: str,
        agent_name: str = "default",
        session_id: str | None = None,
    ) -> str:
        """
        Send a chat message and get response.

        Args:
            message: User message
            agent_name: Agent to use
            session_id: Session identifier

        Returns:
            Agent response text
        """
        from clawscope.message import Msg

        agent = self.get_agent(agent_name)
        if not agent:
            raise ValueError(f"Agent not found: {agent_name}")

        msg = attach_runtime_context(
            Msg(name="user", content=message, role="user"),
            channel="cli",
            chat_id=session_id or "direct",
            session_key=session_id or "cli:direct",
            sender_id="user",
        )

        # Check skills first
        if self._skill_registry:
            skill_response = await self._skill_registry.execute(msg)
            if skill_response:
                return skill_response.get_text_content()

        # Use agent
        response = await agent(msg)
        return response.get_text_content() if response else ""

    async def stream_chat(
        self,
        message: str,
        agent_name: str = "default",
        session_id: str | None = None,
    ):
        """
        Stream a chat response as an async generator.

        Yields dicts with ``type`` key:
        - ``{"type": "content", "content": str}`` – partial text
        - ``{"type": "thinking", "content": str}`` – reasoning token (if supported)
        - ``{"type": "tool_start", "tool_name": str, "tool_id": str}``
        - ``{"type": "tool_result", "tool_id": str, "content": str, "is_error": bool}``
        - ``{"type": "done", "message": Msg}`` – final assembled message

        Falls back to a single non-streamed chunk when the agent doesn't support
        ``stream_reply``.
        """
        from clawscope.message import Msg

        agent = self.get_agent(agent_name)
        if not agent:
            raise ValueError(f"Agent not found: {agent_name}")

        msg = attach_runtime_context(
            Msg(name="user", content=message, role="user"),
            channel="cli",
            chat_id=session_id or "direct",
            session_key=session_id or "cli:direct",
            sender_id="user",
        )

        if hasattr(agent, "stream_reply"):
            async for chunk in agent.stream_reply(msg):
                yield chunk
        else:
            response = await agent(msg)
            text = response.get_text_content() if response else ""
            yield {"type": "content", "content": text}
            yield {"type": "done", "message": response}

    # ==================== Multi-Agent Orchestration ====================

    def create_hub(
        self,
        agent_names: list[str],
        max_rounds: int = 10,
    ) -> "MsgHub":
        """
        Create a MsgHub for multi-agent collaboration.

        Args:
            agent_names: Names of agents to include
            max_rounds: Maximum conversation rounds

        Returns:
            MsgHub instance
        """
        from clawscope.orchestration import MsgHub

        agents = [self._agents[name] for name in agent_names if name in self._agents]
        return MsgHub(participants=agents, max_rounds=max_rounds)

    def create_chatroom(
        self,
        agent_names: list[str],
        name: str = "chatroom",
        speaking_policy: str = "round_robin",
    ) -> "ChatRoom":
        """
        Create a ChatRoom for natural multi-agent conversation.

        Args:
            agent_names: Names of agents to include
            name: Room name
            speaking_policy: Speaking policy

        Returns:
            ChatRoom instance
        """
        from clawscope.orchestration import ChatRoom

        room = ChatRoom(name=name, speaking_policy=speaking_policy)
        for agent_name in agent_names:
            if agent_name in self._agents:
                room.join(self._agents[agent_name])
        return room

    async def run_pipeline(
        self,
        agent_names: list[str],
        input_message: str,
    ) -> str:
        """
        Run agents in a sequential pipeline.

        Args:
            agent_names: Agent names in order
            input_message: Initial input

        Returns:
            Final output
        """
        from clawscope.orchestration import SequentialPipeline
        from clawscope.message import Msg

        agents = [self._agents[name] for name in agent_names if name in self._agents]
        pipeline = SequentialPipeline(agents)

        msg = Msg(name="user", content=input_message, role="user")
        result = await pipeline.run(msg)

        return result.get_text_content() if result else ""

    # ==================== RAG ====================

    def get_knowledge_base(self, name: str = "default") -> "KnowledgeBase | None":
        """Get a knowledge base by name."""
        return self._knowledge_bases.get(name)

    async def add_knowledge(
        self,
        content: str,
        source: str | None = None,
        kb_name: str = "default",
    ) -> int:
        """
        Add knowledge to the knowledge base.

        Args:
            content: Text content
            source: Source identifier
            kb_name: Knowledge base name

        Returns:
            Number of chunks created
        """
        kb = self.get_knowledge_base(kb_name)
        if not kb:
            raise ValueError(f"Knowledge base not found: {kb_name}")

        return await kb.add_text(content, source=source)

    async def search_knowledge(
        self,
        query: str,
        top_k: int = 5,
        kb_name: str = "default",
    ) -> list[dict]:
        """
        Search the knowledge base.

        Args:
            query: Search query
            top_k: Number of results
            kb_name: Knowledge base name

        Returns:
            List of search results
        """
        kb = self.get_knowledge_base(kb_name)
        if not kb:
            return []

        results = await kb.search(query, top_k=top_k)
        return [r.to_dict() for r in results]

    # ==================== Skills ====================

    async def install_skill(self, name: str) -> bool:
        """
        Install a skill from the marketplace.

        Args:
            name: Skill name

        Returns:
            True if installed
        """
        if not self._skill_marketplace:
            return False

        return await self._skill_marketplace.install(name)

    def list_skills(self) -> list[str]:
        """List installed skills."""
        if not self._skill_registry:
            return []

        return [s.name for s in self._skill_registry.list_skills()]

    # ==================== Hooks ====================

    def on(self, event: str, callback: Callable) -> None:
        """
        Register an event hook.

        Events: on_start, on_stop, on_message, on_response, on_error

        Args:
            event: Event name
            callback: Callback function
        """
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(callback)

    async def _trigger_hook(self, event: str, *args, **kwargs) -> None:
        """Trigger hooks for an event."""
        for callback in self._hooks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(*args, **kwargs)
                else:
                    callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Hook error ({event}): {e}")

    # ==================== Properties ====================

    @property
    def is_running(self) -> bool:
        """Check if platform is running."""
        return self._running

    @property
    def bus(self) -> "MessageBus":
        """Get message bus instance."""
        if self._bus is None:
            raise RuntimeError("ClawScope not initialized")
        return self._bus

    @property
    def model_registry(self) -> "ModelRegistry":
        """Get model registry."""
        if self._model_registry is None:
            raise RuntimeError("ClawScope not initialized")
        return self._model_registry

    @property
    def tool_registry(self) -> "ToolRegistry":
        """Get tool registry."""
        if self._tool_registry is None:
            raise RuntimeError("ClawScope not initialized")
        return self._tool_registry

    # ==================== Stats ====================

    def get_stats(self) -> dict[str, Any]:
        """Get platform statistics."""
        stats = {
            "version": self.version,
            "running": self._running,
            "agents": list(self._agents.keys()),
            "knowledge_bases": list(self._knowledge_bases.keys()),
            "skills": self.list_skills(),
            "tasks": len(self._tasks),
        }

        if self._session_manager:
            stats["active_sessions"] = len(self._session_manager._sessions)

        return stats


# ==================== Quick Start Helpers ====================

async def quick_chat(
    message: str,
    model: str = "gpt-4",
    system_prompt: str | None = None,
) -> str:
    """
    Quick one-off chat without full platform setup.

    Args:
        message: User message
        model: Model to use
        system_prompt: System prompt

    Returns:
        Response text
    """
    from clawscope.agent import ReActAgent
    from clawscope.memory import InMemoryMemory
    from clawscope.message import Msg
    from clawscope.model import ModelRegistry
    from clawscope.config import ModelConfig

    model_config = ModelConfig(default_model=model)
    registry = ModelRegistry(model_config)
    llm = registry.get_model()

    agent = ReActAgent(
        name="Assistant",
        sys_prompt=system_prompt or "You are a helpful assistant.",
        model=llm,
        memory=InMemoryMemory(),
    )

    msg = attach_runtime_context(
        Msg(name="user", content=message, role="user"),
        channel="cli",
        chat_id="quick_chat",
        session_key="cli:quick_chat",
        sender_id="user",
    )
    response = await agent(msg)

    return response.get_text_content() if response else ""


def create_agent(
    name: str = "Assistant",
    model: str = "gpt-4",
    system_prompt: str | None = None,
    tools: list | None = None,
) -> "ReActAgent":
    """
    Create a standalone agent.

    Args:
        name: Agent name
        model: Model to use
        system_prompt: System prompt
        tools: List of tools

    Returns:
        ReActAgent instance
    """
    from clawscope.agent import ReActAgent
    from clawscope.memory import InMemoryMemory
    from clawscope.model import ModelRegistry
    from clawscope.tool import ToolRegistry
    from clawscope.config import ModelConfig, ToolsConfig

    model_config = ModelConfig(default_model=model)
    registry = ModelRegistry(model_config)
    llm = registry.get_model()

    tool_registry = ToolRegistry(ToolsConfig())

    return ReActAgent(
        name=name,
        sys_prompt=system_prompt or "You are a helpful assistant.",
        model=llm,
        memory=InMemoryMemory(),
        tools=tool_registry,
    )


__all__ = [
    "ClawScope",
    "quick_chat",
    "create_agent",
]
