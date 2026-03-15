"""ReAct agent shell backed by a kernel-owned reasoning loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from clawscope.kernel.agent import KernelAgent
from clawscope.kernel.loop import LoopConfig, NativeAgentLoop

if TYPE_CHECKING:
    from clawscope.memory import UnifiedMemory
    from clawscope.model import ChatModelBase
    from clawscope.tool import ToolRegistry


@dataclass
class CompressionConfig:
    """Configuration for memory compression."""

    enabled: bool = True
    trigger_tokens: int = 50000  # Trigger compression at this token count
    target_tokens: int = 30000  # Target token count after compression
    preserve_recent: int = 10  # Always preserve this many recent messages


class ReActAgent(KernelAgent):
    """
    ReAct (Reasoning + Acting) agent implementation.

    Features:
    - Kernel-managed iterative reasoning and tool use
    - Memory compression
    - Streaming support
    - Extended thinking (when supported by model)
    """

    def __init__(
        self,
        name: str,
        sys_prompt: str = "You are a helpful AI assistant.",
        model: "ChatModelBase | None" = None,
        memory: "UnifiedMemory | None" = None,
        tools: "ToolRegistry | None" = None,
        max_iterations: int = 40,
        max_tokens: int = 4096,
        compression: CompressionConfig | None = None,
        loop: NativeAgentLoop | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize ReAct agent.

        Args:
            name: Agent name
            sys_prompt: System prompt
            model: Chat model
            memory: Memory system
            tools: Tool registry
            max_iterations: Maximum reasoning iterations
            max_tokens: Maximum tokens per response
            compression: Memory compression config
            **kwargs: Additional options
        """
        super().__init__(
            loop=loop or NativeAgentLoop(
                LoopConfig(
                    max_iterations=max_iterations,
                    max_tokens=max_tokens,
                )
            ),
            name=name,
            sys_prompt=sys_prompt,
            model=model,
            memory=memory,
            tools=tools,
            **kwargs,
        )
        self.max_iterations = self.loop.config.max_iterations
        self.max_tokens = self.loop.config.max_tokens
        self.compression = compression or CompressionConfig()


__all__ = ["ReActAgent", "CompressionConfig"]
