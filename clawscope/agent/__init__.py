"""ClawScope agent system."""

from clawscope.agent.base import AgentBase
from clawscope.agent.react import ReActAgent
from clawscope.agent.user import UserAgent, ChannelUserAgent
from clawscope.agent.a2a import (
    A2AAgent,
    A2AMessage,
    A2AMessageType,
    A2ARouter,
    AgentCard,
    AgentCapability,
    get_router,
)
from clawscope.agent.realtime import (
    RealtimeAgent,
    RealtimeConnection,
    OpenAIRealtimeConnection,
    AudioProvider,
    MicrophoneProvider,
    AudioConfig,
)

__all__ = [
    # Base
    "AgentBase",
    # Agents
    "ReActAgent",
    "UserAgent",
    "ChannelUserAgent",
    "RealtimeAgent",
    # A2A
    "A2AAgent",
    "A2AMessage",
    "A2AMessageType",
    "A2ARouter",
    "AgentCard",
    "AgentCapability",
    "get_router",
    # Realtime
    "RealtimeConnection",
    "OpenAIRealtimeConnection",
    "AudioProvider",
    "MicrophoneProvider",
    "AudioConfig",
]
