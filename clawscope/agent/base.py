"""Base agent class for ClawScope."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Awaitable, TYPE_CHECKING

from loguru import logger

from clawscope.message import Msg

if TYPE_CHECKING:
    from clawscope.memory import UnifiedMemory
    from clawscope.model import ChatModelBase
    from clawscope.tool import ToolRegistry


# Hook type definitions
PreReplyHook = Callable[..., Awaitable[dict[str, Any]]]
PostReplyHook = Callable[..., Awaitable[Msg | None]]
PreObserveHook = Callable[..., Awaitable[Msg | list[Msg] | None]]
PostObserveHook = Callable[..., Awaitable[None]]


class AgentBase(ABC):
    """
    Abstract base class for all agents.

    Provides:
    - Hook system for pre/post processing
    - Memory management
    - Model integration
    - Tool registry access
    """

    def __init__(
        self,
        name: str,
        sys_prompt: str = "",
        model: "ChatModelBase | None" = None,
        memory: "UnifiedMemory | None" = None,
        tools: "ToolRegistry | None" = None,
        **kwargs: Any,
    ):
        """
        Initialize agent.

        Args:
            name: Agent name
            sys_prompt: System prompt for the agent
            model: Chat model for LLM calls
            memory: Memory system
            tools: Tool registry
            **kwargs: Additional options
        """
        self.name = name
        self.sys_prompt = sys_prompt
        self.model = model
        self.memory = memory
        self.tools = tools
        self.extra_options = kwargs

        # Hooks
        self._pre_reply_hooks: list[PreReplyHook] = []
        self._post_reply_hooks: list[PostReplyHook] = []
        self._pre_observe_hooks: list[PreObserveHook] = []
        self._post_observe_hooks: list[PostObserveHook] = []

    @abstractmethod
    async def reply(self, message: Msg | None = None, **kwargs: Any) -> Msg:
        """
        Generate a reply to the input message.

        Args:
            message: Input message (optional)
            **kwargs: Additional options

        Returns:
            Agent's response message
        """
        pass

    async def observe(self, message: Msg | list[Msg] | None) -> None:
        """
        Observe a message without generating a reply.

        Args:
            message: Message(s) to observe
        """
        if message is None:
            return

        # Run pre-observe hooks
        for hook in self._pre_observe_hooks:
            message = await hook(message=message)
            if message is None:
                return

        # Add to memory
        if self.memory:
            messages = [message] if isinstance(message, Msg) else message
            await self.memory.add(messages)

        # Run post-observe hooks
        for hook in self._post_observe_hooks:
            await hook(message=message)

    async def __call__(self, message: Msg | None = None, **kwargs: Any) -> Msg:
        """Shorthand for reply method."""
        return await self.reply(message, **kwargs)

    # ========== Hook Registration ==========

    def on_pre_reply(self, hook: PreReplyHook) -> PreReplyHook:
        """
        Register pre-reply hook.

        Hook receives kwargs and returns modified kwargs.
        """
        self._pre_reply_hooks.append(hook)
        return hook

    def on_post_reply(self, hook: PostReplyHook) -> PostReplyHook:
        """
        Register post-reply hook.

        Hook receives output Msg and returns modified Msg or None.
        """
        self._post_reply_hooks.append(hook)
        return hook

    def on_pre_observe(self, hook: PreObserveHook) -> PreObserveHook:
        """
        Register pre-observe hook.

        Hook receives message and returns modified message or None.
        """
        self._pre_observe_hooks.append(hook)
        return hook

    def on_post_observe(self, hook: PostObserveHook) -> PostObserveHook:
        """
        Register post-observe hook.

        Hook receives message after observation.
        """
        self._post_observe_hooks.append(hook)
        return hook

    # ========== Hook Execution ==========

    async def _run_pre_reply_hooks(self, **kwargs: Any) -> dict[str, Any]:
        """Run all pre-reply hooks."""
        for hook in self._pre_reply_hooks:
            try:
                kwargs = await hook(**kwargs)
            except Exception as e:
                logger.error(f"Pre-reply hook error: {e}")
        return kwargs

    async def _run_post_reply_hooks(self, output: Msg) -> Msg | None:
        """Run all post-reply hooks."""
        for hook in self._post_reply_hooks:
            try:
                result = await hook(output=output)
                if result is not None:
                    output = result
            except Exception as e:
                logger.error(f"Post-reply hook error: {e}")
        return output

    # ========== Memory Management ==========

    async def get_memory_messages(self) -> list[Msg]:
        """Get messages from memory."""
        if self.memory:
            return await self.memory.get()
        return []

    async def clear_memory(self) -> None:
        """Clear agent's memory."""
        if self.memory:
            await self.memory.clear()

    # ========== Tool Management ==========

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Get tool definitions for LLM."""
        if self.tools:
            return self.tools.get_definitions()
        return []

    async def execute_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool by name."""
        if self.tools:
            return await self.tools.execute(name, arguments)
        raise RuntimeError("No tool registry configured")

    # ========== Utility Methods ==========

    def _build_system_message(self) -> Msg | None:
        """Build system message from prompt."""
        if self.sys_prompt:
            return Msg(
                name="system",
                content=self.sys_prompt,
                role="system",
            )
        return None

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


__all__ = ["AgentBase"]
