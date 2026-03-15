"""AgentScope kernel integration for ClawScope."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from clawscope.agent.base import AgentBase
from clawscope.config import AgentConfig, ModelConfig
from clawscope.message import Msg
from clawscope.tool import ToolRegistry


class AgentScopeKernelError(RuntimeError):
    """Raised when the AgentScope kernel cannot be initialized."""


def _default_agentscope_src() -> Path:
    """Resolve the local AgentScope source tree."""
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root.parent / "agentscope" / "src"


def ensure_agentscope_importable() -> None:
    """Make the local AgentScope source tree importable."""
    src = Path(
        os.environ.get(
            "CLAWSCOPE_AGENTSCOPE_SRC",
            str(_default_agentscope_src()),
        ),
    )
    if not src.exists():
        raise AgentScopeKernelError(
            f"AgentScope source not found: {src}. "
            "Set CLAWSCOPE_AGENTSCOPE_SRC to the AgentScope src directory.",
        )

    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


def _select_agentscope_classes(provider: str) -> tuple[type[Any], type[Any]]:
    """Select AgentScope model and formatter classes for a provider."""
    ensure_agentscope_importable()

    from agentscope.formatter import (
        AnthropicChatFormatter,
        DashScopeChatFormatter,
        GeminiChatFormatter,
        OllamaChatFormatter,
        OpenAIChatFormatter,
    )
    from agentscope.model import (
        AnthropicChatModel,
        DashScopeChatModel,
        GeminiChatModel,
        OllamaChatModel,
        OpenAIChatModel,
    )

    mapping: dict[str, tuple[type[Any], type[Any]]] = {
        "openai": (OpenAIChatModel, OpenAIChatFormatter),
        "anthropic": (AnthropicChatModel, AnthropicChatFormatter),
        "dashscope": (DashScopeChatModel, DashScopeChatFormatter),
        "gemini": (GeminiChatModel, GeminiChatFormatter),
        "ollama": (OllamaChatModel, OllamaChatFormatter),
    }

    if provider not in mapping:
        raise AgentScopeKernelError(
            f"Provider '{provider}' is not supported by the AgentScope kernel adapter."
        )

    return mapping[provider]


def _build_agentscope_model(model_config: ModelConfig) -> tuple[Any, Any]:
    """Build AgentScope model and formatter instances."""
    model_cls, formatter_cls = _select_agentscope_classes(model_config.provider)

    model_kwargs: dict[str, Any] = {
        "model_name": model_config.default_model,
        "stream": model_config.stream,
    }

    if model_config.api_key:
        model_kwargs["api_key"] = model_config.api_key

    if model_config.provider == "openai":
        client_kwargs: dict[str, Any] = {}
        if model_config.api_base:
            client_kwargs["base_url"] = model_config.api_base
        if client_kwargs:
            model_kwargs["client_kwargs"] = client_kwargs
    elif model_config.provider == "anthropic":
        if model_config.timeout:
            model_kwargs["timeout"] = model_config.timeout
    elif model_config.provider == "dashscope":
        if model_config.api_base:
            model_kwargs["base_http_api_url"] = model_config.api_base
    elif model_config.provider == "ollama":
        if model_config.api_base:
            model_kwargs["host"] = model_config.api_base
    elif model_config.provider == "gemini":
        client_kwargs = {}
        if model_config.api_base:
            client_kwargs["http_options"] = {"base_url": model_config.api_base}
        if client_kwargs:
            model_kwargs["client_kwargs"] = client_kwargs

    model = model_cls(**model_kwargs)
    formatter = formatter_cls()
    return model, formatter


def _to_agentscope_msg(msg: Msg | None) -> Any:
    """Convert a ClawScope Msg into an AgentScope Msg."""
    if msg is None:
        return None

    ensure_agentscope_importable()
    from agentscope.message import Msg as AgentScopeMsg

    content: Any
    if isinstance(msg.content, str):
        content = msg.content
    else:
        content = [
            block.to_dict() if hasattr(block, "to_dict") else block
            for block in msg.content
        ]

    converted = AgentScopeMsg(
        name=msg.name,
        content=content,
        role=msg.role,
        metadata=dict(msg.metadata),
        timestamp=msg.timestamp,
        invocation_id=msg.invocation_id,
    )
    converted.id = msg.id
    return converted


def _from_agentscope_msg(msg: Any) -> Msg:
    """Convert an AgentScope Msg into a ClawScope Msg."""
    payload = msg.to_dict()
    payload["invocation_id"] = getattr(msg, "invocation_id", None)
    return Msg.from_dict(payload)


def _build_toolkit(tool_registry: ToolRegistry) -> Any:
    """Build an AgentScope Toolkit from a ClawScope tool registry."""
    ensure_agentscope_importable()

    from agentscope.message import TextBlock
    from agentscope.tool import ToolResponse, Toolkit

    toolkit = Toolkit()

    for tool_name in tool_registry.list_tools():
        async def _tool_wrapper(
            _tool_name: str = tool_name,
            **kwargs: Any,
        ) -> ToolResponse:
            result = await tool_registry.execute(_tool_name, kwargs)
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=result,
                    ),
                ],
            )

        _tool_wrapper.__name__ = tool_name
        original = tool_registry.get(tool_name)
        if original is not None:
            _tool_wrapper.__doc__ = original.description

        toolkit.register_tool_function(
            _tool_wrapper,
            func_name=tool_name,
            func_description=original.description if original else tool_name,
            namesake_strategy="override",
        )

    return toolkit


class AgentScopeReActAgent(AgentBase):
    """ClawScope agent wrapper around AgentScope's ReAct agent."""

    def __init__(
        self,
        name: str,
        sys_prompt: str,
        agent: Any,
        tools: ToolRegistry | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, sys_prompt=sys_prompt, tools=tools, **kwargs)
        self._agent = agent

    async def reply(self, message: Msg | None = None, **kwargs: Any) -> Msg:
        """Generate a reply via AgentScope."""
        kwargs = await self._run_pre_reply_hooks(message=message, **kwargs)
        message = kwargs.pop("message", message)

        response = await self._agent(_to_agentscope_msg(message), **kwargs)
        result = _from_agentscope_msg(response)
        result = await self._run_post_reply_hooks(result) or result
        return result

    async def observe(self, message: Msg | list[Msg] | None) -> None:
        """Forward observed messages into AgentScope memory."""
        if message is None:
            return

        for hook in self._pre_observe_hooks:
            message = await hook(message=message)
            if message is None:
                return

        if isinstance(message, list):
            payload = [_to_agentscope_msg(item) for item in message]
        else:
            payload = _to_agentscope_msg(message)

        await self._agent.observe(payload)

        for hook in self._post_observe_hooks:
            await hook(message=message)


def create_agentscope_react_agent(
    agent_config: AgentConfig,
    model_config: ModelConfig,
    tool_registry: ToolRegistry,
) -> AgentScopeReActAgent:
    """Create a ClawScope agent backed by AgentScope."""
    ensure_agentscope_importable()

    from agentscope.agent import ReActAgent as AgentScopeAgent
    from agentscope.memory import InMemoryMemory

    model, formatter = _build_agentscope_model(model_config)
    toolkit = _build_toolkit(tool_registry)

    agent = AgentScopeAgent(
        name=agent_config.name,
        sys_prompt=agent_config.sys_prompt,
        model=model,
        formatter=formatter,
        toolkit=toolkit,
        memory=InMemoryMemory(),
        max_iters=agent_config.max_iterations,
    )

    return AgentScopeReActAgent(
        name=agent_config.name,
        sys_prompt=agent_config.sys_prompt,
        agent=agent,
        tools=tool_registry,
    )


__all__ = [
    "AgentScopeKernelError",
    "AgentScopeReActAgent",
    "create_agentscope_react_agent",
    "ensure_agentscope_importable",
]
