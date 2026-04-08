"""OrchestratorAgent – main agent that automatically coordinates sub-agents.

Channel messages arrive at the orchestrator. Its LLM (via the standard ReAct
loop) decides which sub-agents to delegate to, in what order, and how to
combine their answers into a final reply.

Sub-agents are exposed as first-class tools (``ask_<name>(message)``), so
any orchestration strategy the LLM can reason about is supported out of the
box: sequential, parallel-then-merge, conditional, iterative, etc.

Progress reporting
------------------
Attach a :class:`ProgressReporter` before each reply so the orchestrator can
send real-time status messages back to the channel while it works::

    reporter = ProgressReporter(bus, channel="feishu", chat_id="xxx")
    orch.set_progress_reporter(reporter)
    response = await orch(msg)

Natural language agent management
----------------------------------
Users can manage agents through conversation.  The orchestrator understands
commands like:

- "给 researcher 换一个更学术的角色"   → calls ``update_agent``
- "删掉 writer 这个 Agent"            → calls ``remove_agent``
- "现在有哪些 Agent？"                → calls ``list_agents``
- "帮我新建一个代码审查 Agent"        → calls ``create_agent``
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, TYPE_CHECKING

from loguru import logger

from clawscope.agent.base import AgentBase
from clawscope.agent.react import ReActAgent
from clawscope.message import Msg

if TYPE_CHECKING:
    from clawscope.bus import MessageBus
    from clawscope.memory import UnifiedMemory
    from clawscope.model import ChatModelBase
    from clawscope.tool import ToolRegistry


# ---------------------------------------------------------------------------
# Progress reporter
# ---------------------------------------------------------------------------


class ProgressReporter:
    """
    Sends real-time orchestration progress messages back to a channel.

    The reporter is injected per-message (not per-session) so that
    concurrent sessions each get their own target::

        reporter = ProgressReporter(bus, "feishu", chat_id="oc_xxx")
        orchestrator.set_progress_reporter(reporter)
        await orchestrator(msg)
    """

    def __init__(
        self,
        bus: "MessageBus",
        channel: str,
        chat_id: str,
        *,
        enabled: bool = True,
    ) -> None:
        self._bus = bus
        self._channel = channel
        self._chat_id = chat_id
        self.enabled = enabled

    async def send(self, text: str) -> None:
        """Publish a progress message to the channel (fire-and-forget)."""
        if not self.enabled:
            return
        try:
            from clawscope.bus import OutboundMessage
            await self._bus.publish_outbound(
                OutboundMessage(
                    channel=self._channel,
                    chat_id=self._chat_id,
                    content=text,
                )
            )
        except Exception as exc:
            logger.warning(f"ProgressReporter: failed to send: {exc}")

    def fire(self, text: str) -> None:
        """Schedule send without awaiting (usable from sync context)."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.send(text))
        except RuntimeError:
            pass


# ---------------------------------------------------------------------------
# Execution log entry
# ---------------------------------------------------------------------------


@dataclass
class _AgentCall:
    name: str
    task_preview: str          # first 60 chars of task
    result_preview: str = ""   # first 60 chars of result
    is_error: bool = False


# ---------------------------------------------------------------------------
# Default system prompt
# ---------------------------------------------------------------------------

_DEFAULT_ORCHESTRATOR_PROMPT = """\
You are the main orchestrator agent. Your job is to analyse incoming messages
and coordinate specialised sub-agents to produce the best possible response.

Orchestration guidelines
-------------------------
1. Read the user's message carefully and decide which sub-agent(s) to call.
2. Call sub-agents via ask_<name>(message=...). You may call multiple agents,
   sequentially or in a logical order, then combine their outputs.
3. You may also answer directly without any sub-agent for simple questions.
4. After collecting results, synthesise them into a coherent reply.

Dynamic agent creation
-----------------------
- spawn_agent(role, task)  — temporary agent for a one-off task, discarded after.
- create_agent(name, role) — register a new persistent agent callable as ask_<name>.
- list_agents()            — list currently registered sub-agents.

Agent management (natural-language users may ask you to do these)
------------------------------------------------------------------
- update_agent(name, new_role)         — change a sub-agent's role/prompt.
- remove_agent(name)                   — unregister and discard a sub-agent.

When a user says something like "给 X Agent 换个角色" or "删掉 Y Agent" or
"现在有哪些 Agent", recognise it as a management command and call the
appropriate tool. Confirm the change clearly in your final reply.

Pre-registered sub-agents are listed as tools prefixed with ask_.
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
        show_progress: Whether to send intermediate progress messages.
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
        show_progress: bool = True,
        **kwargs: Any,
    ):
        effective_prompt = sys_prompt or _DEFAULT_ORCHESTRATOR_PROMPT

        from clawscope.tool import ToolRegistry
        from clawscope.config import ToolsConfig

        agent_tool_registry = tools if tools is not None else ToolRegistry(ToolsConfig())

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
        self._progress_reporter: ProgressReporter | None = None
        self._show_progress = show_progress
        self._call_log: list[_AgentCall] = []   # reset each reply()

        # Register provided sub-agents
        if sub_agents is not None:
            agents_dict = sub_agents if isinstance(sub_agents, dict) else {
                a.name: a for a in sub_agents
            }
            for agent_name, agent in agents_dict.items():
                self.register_sub_agent(agent, alias=agent_name)

        # Built-in meta-tools
        self._register_meta_tools()

    # ------------------------------------------------------------------
    # Progress reporter
    # ------------------------------------------------------------------

    def set_progress_reporter(self, reporter: ProgressReporter | None) -> None:
        """Attach a reporter for the next reply() call."""
        self._progress_reporter = reporter

    async def _progress(self, text: str) -> None:
        """Send a progress message if a reporter is attached."""
        if self._progress_reporter and self._show_progress:
            await self._progress_reporter.send(text)

    # ------------------------------------------------------------------
    # Override reply() to reset log and wrap final response
    # ------------------------------------------------------------------

    async def reply(self, message: Msg | None = None, **kwargs: Any) -> Msg:
        """Run the ReAct loop and return a formatted final message."""
        self._call_log = []
        result = await super().reply(message, **kwargs)

        # Prepend orchestration summary if agents were called
        if self._call_log and result is not None:
            result = self._attach_summary(result)

        return result

    def _attach_summary(self, msg: Msg) -> Msg:
        """Prepend a one-line orchestration summary to the message content."""
        names = [c.name for c in self._call_log]
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_names: list[str] = []
        for n in names:
            if n not in seen:
                seen.add(n)
                unique_names.append(n)

        header = f"[ {' → '.join(unique_names)} ]\n\n"
        text = msg.get_text_content()
        new_content = header + text
        return Msg(
            name=msg.name,
            content=new_content,
            role=msg.role,
            metadata=dict(msg.metadata),
        )

    # ------------------------------------------------------------------
    # Sub-agent registration
    # ------------------------------------------------------------------

    def register_sub_agent(
        self,
        agent: AgentBase,
        alias: str | None = None,
    ) -> None:
        """Register a sub-agent and expose it as a ``ask_<name>`` tool."""
        key = alias or agent.name
        self._sub_agents[key] = agent
        self._expose_as_tool(key, agent)
        logger.debug(f"OrchestratorAgent '{self.name}': registered sub-agent '{key}'")

    def unregister_sub_agent(self, name: str) -> bool:
        """Remove a sub-agent and its tool. Returns True if found."""
        if name not in self._sub_agents:
            return False
        del self._sub_agents[name]
        self.tools._tools.pop(f"ask_{name}", None)
        logger.debug(f"OrchestratorAgent '{self.name}': unregistered sub-agent '{name}'")
        return True

    @property
    def sub_agents(self) -> dict[str, AgentBase]:
        return dict(self._sub_agents)

    # ------------------------------------------------------------------
    # _expose_as_tool  (with progress + log)
    # ------------------------------------------------------------------

    def _expose_as_tool(self, name: str, agent: AgentBase) -> None:
        """Create an ``ask_<name>`` tool wrapper with progress reporting."""
        from clawscope.tool.registry import Tool, ToolParameter

        raw_desc = getattr(agent, "sys_prompt", "") or f"Sub-agent: {name}"
        short_desc = raw_desc[:300].rstrip() + ("…" if len(raw_desc) > 300 else "")
        tool_name = f"ask_{name}"
        description = f"Delegate a task to the '{name}' agent.\n\n{short_desc}"

        _agent = agent
        _name = name

        async def _delegate(message: str) -> str:
            preview = message[:60].replace("\n", " ")
            await self._progress(f"⚙️  [{_name}]  {preview}…")

            entry = _AgentCall(name=_name, task_preview=preview)
            self._call_log.append(entry)

            msg = Msg(name=self.name, content=message, role="user")
            try:
                response = await _agent(msg)
                text = response.get_text_content() if response else ""
                entry.result_preview = text[:60].replace("\n", " ")
                await self._progress(f"✓  [{_name}]  完成")
                return text or f"[{_name}] returned no response."
            except Exception as exc:
                entry.is_error = True
                entry.result_preview = str(exc)[:60]
                await self._progress(f"✗  [{_name}]  出错: {exc}")
                logger.error(f"OrchestratorAgent: sub-agent '{_name}' error: {exc}")
                return f"[{_name}] error: {exc}"

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
    # Meta-tools: dynamic creation + management
    # ------------------------------------------------------------------

    def _register_meta_tools(self) -> None:
        """Register built-in meta-tools for agent creation and management."""
        from clawscope.tool.registry import Tool, ToolParameter

        # ── spawn_agent ──────────────────────────────────────────────
        async def _spawn_agent(role: str, task: str) -> str:
            return await self._do_spawn(role, task, register_as=None)

        self.tools._tools["spawn_agent"] = Tool(
            name="spawn_agent",
            description=(
                "Create a temporary agent with a custom role, run it on one task, "
                "return the result, then discard it. Use when no registered agent fits."
            ),
            parameters=[
                ToolParameter("role", "string",
                              "System prompt / role description for the temporary agent.",
                              required=True),
                ToolParameter("task", "string",
                              "The complete, self-contained task for the agent.",
                              required=True),
            ],
            func=_spawn_agent,
            enabled=True,
        )

        # ── create_agent ─────────────────────────────────────────────
        async def _create_agent(name: str, role: str) -> str:
            return await self._do_create_and_register(name, role)

        self.tools._tools["create_agent"] = Tool(
            name="create_agent",
            description=(
                "Create a new specialised agent and register it permanently under *name*. "
                "It becomes callable as ask_<name> in future turns."
            ),
            parameters=[
                ToolParameter("name", "string",
                              "Short identifier (no spaces). Becomes ask_<name>.",
                              required=True),
                ToolParameter("role", "string",
                              "System prompt / role for the new agent.",
                              required=True),
            ],
            func=_create_agent,
            enabled=True,
        )

        # ── update_agent ─────────────────────────────────────────────
        async def _update_agent(name: str, new_role: str) -> str:
            return self._do_update_agent(name, new_role)

        self.tools._tools["update_agent"] = Tool(
            name="update_agent",
            description=(
                "Change the role/system-prompt of a registered sub-agent. "
                "Use when the user asks to update, adjust, or retrain an agent."
            ),
            parameters=[
                ToolParameter("name", "string",
                              "Name of the sub-agent to update.",
                              required=True),
                ToolParameter("new_role", "string",
                              "New system prompt / role description.",
                              required=True),
            ],
            func=_update_agent,
            enabled=True,
        )

        # ── remove_agent ─────────────────────────────────────────────
        async def _remove_agent(name: str) -> str:
            removed = self.unregister_sub_agent(name)
            if removed:
                return f"Agent '{name}' has been removed."
            return f"No agent named '{name}' found."

        self.tools._tools["remove_agent"] = Tool(
            name="remove_agent",
            description=(
                "Unregister and discard a sub-agent by name. "
                "Use when the user asks to delete or remove an agent."
            ),
            parameters=[
                ToolParameter("name", "string",
                              "Name of the sub-agent to remove.",
                              required=True),
            ],
            func=_remove_agent,
            enabled=True,
        )

        # ── list_agents ──────────────────────────────────────────────
        async def _list_agents() -> str:
            if not self._sub_agents:
                return "没有已注册的 Agent。"
            lines = ["当前已注册的 Agent："]
            for n, agent in self._sub_agents.items():
                role_snippet = (
                    (getattr(agent, "sys_prompt", "") or "")[:80]
                    .replace("\n", " ")
                    .strip()
                )
                lines.append(f"  • {n}：{role_snippet}")
            return "\n".join(lines)

        self.tools._tools["list_agents"] = Tool(
            name="list_agents",
            description="List all currently registered sub-agents and their roles.",
            parameters=[],
            func=_list_agents,
            enabled=True,
        )

    # ------------------------------------------------------------------
    # Dynamic agent helpers
    # ------------------------------------------------------------------

    async def _do_spawn(
        self,
        role: str,
        task: str,
        register_as: str | None,
    ) -> str:
        from clawscope.memory import InMemoryMemory
        from clawscope.tool import ToolRegistry
        from clawscope.config import ToolsConfig

        agent_name = register_as or f"_spawned_{len(self._sub_agents)}"

        spawned = ReActAgent(
            name=agent_name,
            sys_prompt=role,
            model=self.model,
            memory=InMemoryMemory(),
            tools=ToolRegistry(ToolsConfig()),
            max_iterations=self.max_iterations,
        )

        if register_as:
            self.register_sub_agent(spawned, alias=register_as)
            logger.info(f"OrchestratorAgent: created + registered '{register_as}'")

        if not task:
            return f"Agent '{agent_name}' created."

        logger.debug(f"OrchestratorAgent: spawning '{agent_name}' ({len(task)} chars)")
        msg = Msg(name=self.name, content=task, role="user")
        try:
            response = await spawned(msg)
            return response.get_text_content() if response else f"[{agent_name}] no response."
        except Exception as exc:
            logger.error(f"OrchestratorAgent: spawned '{agent_name}' error: {exc}")
            return f"[{agent_name}] error: {exc}"

    async def _do_create_and_register(self, name: str, role: str) -> str:
        safe = name.strip().replace(" ", "_")
        if not safe:
            return "错误：Agent 名称不能为空。"
        if safe in self._sub_agents:
            return (
                f"Agent '{safe}' 已存在，可以直接调用 ask_{safe}。"
                "如需更换角色，请使用 update_agent。"
            )
        await self._do_spawn(role, task="", register_as=safe)
        return (
            f"✅ 已创建并注册 Agent「{safe}」。"
            f"现在可以用 ask_{safe}(message=...) 调用它。"
        )

    def _do_update_agent(self, name: str, new_role: str) -> str:
        agent = self._sub_agents.get(name)
        if agent is None:
            return f"找不到 Agent '{name}'。请先用 list_agents 查看已注册的 Agent。"
        # Update the agent's sys_prompt in-place
        agent.sys_prompt = new_role  # type: ignore[attr-defined]
        # Refresh the tool description so future LLM calls see the new role
        self._expose_as_tool(name, agent)
        logger.info(f"OrchestratorAgent: updated role for '{name}'")
        return f"✅ 已更新 Agent「{name}」的角色。新角色：{new_role[:100]}"

    # ------------------------------------------------------------------
    # Catalogue
    # ------------------------------------------------------------------

    def _build_agent_catalogue(self) -> str:
        if not self._sub_agents:
            return "(暂无已注册的 Agent)"
        lines = []
        for key, agent in self._sub_agents.items():
            raw = getattr(agent, "sys_prompt", "") or ""
            snippet = raw[:120].replace("\n", " ").rstrip()
            if len(raw) > 120:
                snippet += "…"
            lines.append(f"- **{key}** (`ask_{key}`): {snippet}")
        return "\n".join(lines)

    def get_effective_sys_prompt(self) -> str:
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
    Shorthand factory for a standalone :class:`OrchestratorAgent`::

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


__all__ = ["OrchestratorAgent", "ProgressReporter", "create_orchestrator"]
