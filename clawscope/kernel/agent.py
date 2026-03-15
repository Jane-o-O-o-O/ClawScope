"""Agent wrappers that delegate reasoning to kernel-owned loops."""

from __future__ import annotations

from typing import Any

from clawscope.agent.base import AgentBase
from clawscope.kernel.loop import AgentLoop
from clawscope.message import Msg


class KernelAgent(AgentBase):
    """Generic agent shell that delegates reply execution to an AgentLoop."""

    def __init__(
        self,
        *,
        loop: AgentLoop,
        name: str,
        sys_prompt: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, sys_prompt=sys_prompt, **kwargs)
        self.loop = loop

    async def reply(self, message: Msg | None = None, **kwargs: Any) -> Msg:
        """Run hooks and delegate the core reasoning loop to the kernel."""
        kwargs = await self._run_pre_reply_hooks(message=message, **kwargs)
        message = kwargs.pop("message", message)

        response = await self.loop.run(self, message=message, **kwargs)
        result = await self._run_post_reply_hooks(response)
        return result or response

    async def stream_reply(self, message: Msg | None = None, **kwargs: Any):
        """Stream a reply through the configured kernel loop."""
        kwargs = await self._run_pre_reply_hooks(message=message, **kwargs)
        message = kwargs.pop("message", message)

        async for event in self.loop.stream(self, message=message, **kwargs):
            yield event


__all__ = ["KernelAgent"]
