"""OrchestratorAgent – main agent that automatically coordinates sub-agents.

Channel messages arrive at the orchestrator. Its LLM (via the standard ReAct
loop) decides which sub-agents to delegate to, in what order, and how to
combine their answers into a final reply.

Sub-agents are exposed as first-class tools (``ask_<name>(message)``), so
any orchestration strategy that the underlying model can reason about is
supported out of the box: sequential, parallel-then-merge, conditional,
iterative, etc.

Example::

    from clawscope import ClawScope
    from clawscope.agent import ReActAgent

    app = ClawScope.create(model_provider="anthropic", default_model="claude-sonnet-4-6")
    await app.start()

    researcher = ReActAgent(name="researcher", sys_prompt="You research topics thoroughly.", ...)
    writer     = ReActAgent(name="writer",     sys_prompt="You write polished prose.", ...)

    app.register_sub_agent("researcher", researcher)
    app.register_sub_agent("writer",     writer)

    # Every channel message is now routed through the orchestrator,
    # which decides whether to call researcher, writer, both, or neither.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from loguru import logger

from clawscope.agent.base import AgentBase
from clawscope.agent.react import ReActAgent
from clawscope.message import Msg

if TYPE_CHECKING:
    from clawscope.memory import UnifiedMemory
    from clawscope.model import ChatModelBase
    from clawscope.tool import ToolRegistry


# ---------------------------------------------------------------------------
# Default system prompt
# ---------------------------------------------------------------------------

_DEFAULT_ORCHESTRATOR_PROMPT = """\
You are the main orchestrator agent. Your job is to analyse incoming messages
and coordinate specialised sub-agents to produce the best possible response.

Guidelines
----------
1. Read the user's message carefully and decide which sub-agent(s) to call.
2. You may call multiple sub-agents, sequentially or by combining their
   outputs. You may also answer directly without calling any sub-agent when
   the question is simple.
3. After collecting sub-agent results, synthesise them into a coherent,
   helpful reply for the user.
4. Always be transparent: if a sub-agent produced an error, note it and
   try to work around it.

Available sub-agents are listed as tools prefixed with ``ask_``.
"""


# ---------------------------------------------------------------------------
# OrchestratorAgent
# ---------------------------------------------------------------------------


class OrchestratorAgent(ReActAgent):
    """
    Main agent that routes channel messages to a registry of sub-agents.

    Each sub-agent is wrapped as a ReAct tool named ``ask_<agent_name>``.
    The orchestrator's LLM decides autonomously which agents to invoke and
    how to combine their responses.

    Args:
        name: Orchestrator name (appears in responses).
        sys_prompt: System prompt. A sensible default is provided if omitted.
        model: LLM instance driving the ReAct loop.
        memory: Per-session memory.
        tools: Optional extra tool registry (agent tools are added on top).
        sub_agents: Pre-registered sub-agents (list or name→agent dict).
        max_iterations: ReAct loop iteration cap.
        **kwargs: Forwarded to :class:`ReActAgent`.
    """

    def __init__(
        self,
        name: str = "Orchestrator",
        sys_prompt: str = "",
        model: "ChatModelBase | None" = None,
        memory: "UnifiedMemory | None" = None,
        tools: "ToolRegistry | None" = None,
        sub_agents: list[AgentBase] | dict[str, AgentBase] | None = None,
        max_iterations: int = 40,
        **kwargs: Any,
    ):
        # Ensure a sensible default prompt
        effective_prompt = sys_prompt or _DEFAULT_ORCHESTRATOR_PROMPT

        # Build a fresh ToolRegistry to hold the agent-tools.
        # If the caller supplied extra tools, merge them in.
        from clawscope.tool import ToolRegistry
        from clawscope.config import ToolsConfig

        if tools is None:
            agent_tool_registry = ToolRegistry(ToolsConfig())
        else:
            # Copy existing tools so we don't mutate the caller's registry
            agent_tool_registry = tools

        super().__init__(
            name=name,
            sys_prompt=effective_prompt,
            model=model,
            memory=memory,
            tools=agent_tool_registry,
            max_iterations=max_iterations,
            **kwargs,
        )

        self._sub_agents: dict[str, AgentBase] = {}

        # Register provided sub-agents
        if sub_agents is not None:
            if isinstance(sub_agents, dict):
                for agent_name, agent in sub_agents.items():
                    self.register_sub_agent(agent, alias=agent_name)
            else:
                for agent in sub_agents:
                    self.register_sub_agent(agent)

    # ------------------------------------------------------------------
    # Sub-agent registration
    # ------------------------------------------------------------------

    def register_sub_agent(
        self,
        agent: AgentBase,
        alias: str | None = None,
    ) -> None:
        """
        Register a sub-agent and expose it as a tool.

        Args:
            agent: The agent to register.
            alias: Tool name override. Defaults to ``agent.name``.
        """
        key = alias or agent.name
        self._sub_agents[key] = agent
        self._expose_as_tool(key, agent)
        logger.debug(f"OrchestratorAgent '{self.name}': registered sub-agent '{key}'")

    def unregister_sub_agent(self, name: str) -> bool:
        """
        Remove a sub-agent and its corresponding tool.

        Returns:
            True if the agent was found and removed.
        """
        if name not in self._sub_agents:
            return False
        del self._sub_agents[name]
        tool_name = f"ask_{name}"
        self.tools._tools.pop(tool_name, None)
        logger.debug(f"OrchestratorAgent '{self.name}': unregistered sub-agent '{name}'")
        return True

    @property
    def sub_agents(self) -> dict[str, AgentBase]:
        """Read-only view of registered sub-agents."""
        return dict(self._sub_agents)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _expose_as_tool(self, name: str, agent: AgentBase) -> None:
        """Create a tool wrapper for *agent* and insert it into the registry."""
        from clawscope.tool.registry import Tool, ToolParameter

        # Build a short description from the agent's sys_prompt
        raw_desc = getattr(agent, "sys_prompt", "") or f"Sub-agent: {name}"
        short_desc = raw_desc[:300].rstrip()
        if len(raw_desc) > 300:
            short_desc += "…"

        tool_name = f"ask_{name}"
        description = f"Delegate a task to the '{name}' agent.\n\n{short_desc}"

        # Capture by value to avoid late-binding closure issues
        _agent = agent

        async def _delegate(message: str) -> str:  # noqa: ANN202
            """Call the sub-agent."""
            msg = Msg(name=self.name, content=message, role="user")
            try:
                response = await _agent(msg)
                if response is None:
                    return f"[{name}] returned no response."
                text = response.get_text_content()
                logger.debug(f"OrchestratorAgent: '{name}' replied ({len(text)} chars)")
                return text
            except Exception as exc:
                logger.error(f"OrchestratorAgent: sub-agent '{name}' error: {exc}")
                return f"[{name}] error: {exc}"

        _delegate.__name__ = tool_name
        _delegate.__doc__ = description

        self.tools._tools[tool_name] = Tool(
            name=tool_name,
            description=description,
            parameters=[
                ToolParameter(
                    name="message",
                    type="string",
                    description=(
                        "The task or question to send to this agent. "
                        "Be specific and self-contained; include all context the agent needs."
                    ),
                    required=True,
                )
            ],
            func=_delegate,
            enabled=True,
        )

    # ------------------------------------------------------------------
    # Dynamic system-prompt update
    # ------------------------------------------------------------------

    def _build_agent_catalogue(self) -> str:
        """Return a markdown list of registered sub-agents for the sys prompt."""
        if not self._sub_agents:
            return "(no sub-agents registered)"
        lines = []
        for key, agent in self._sub_agents.items():
            raw = getattr(agent, "sys_prompt", "") or ""
            snippet = raw[:120].replace("\n", " ").rstrip()
            if len(raw) > 120:
                snippet += "…"
            lines.append(f"- **{key}** (`ask_{key}`): {snippet}")
        return "\n".join(lines)

    def get_effective_sys_prompt(self) -> str:
        """Return sys_prompt with an auto-generated agent catalogue appended."""
        catalogue = self._build_agent_catalogue()
        return (
            f"{self.sys_prompt}\n\n"
            f"## Registered sub-agents\n\n"
            f"{catalogue}"
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def create_orchestrator(
    name: str = "Orchestrator",
    sys_prompt: str = "",
    model: "ChatModelBase | None" = None,
    sub_agents: list[AgentBase] | dict[str, AgentBase] | None = None,
    **kwargs: Any,
) -> OrchestratorAgent:
    """
    Shorthand factory for creating a standalone :class:`OrchestratorAgent`.

    Example::

        orch = create_orchestrator(
            model=my_model,
            sub_agents={"search": search_agent, "summarise": summarise_agent},
        )
        response = await orch(user_msg)
    """
    from clawscope.memory import InMemoryMemory

    return OrchestratorAgent(
        name=name,
        sys_prompt=sys_prompt,
        model=model,
        memory=InMemoryMemory(),
        sub_agents=sub_agents,
        **kwargs,
    )


__all__ = ["OrchestratorAgent", "create_orchestrator"]
