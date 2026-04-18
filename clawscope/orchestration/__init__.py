"""ClawScope orchestration system."""

from clawscope.orchestration.router import SessionRouter
from clawscope.orchestration.pipeline import (
    Pipeline,
    SequentialPipeline,
    FanOutPipeline,
    ConditionalPipeline,
)
from clawscope.orchestration.msghub import MsgHub, MsgHubBuilder, Participant
from clawscope.orchestration.chatroom import ChatRoom, ChatParticipant, SpeakingPolicy, Debate

__all__ = [
    # Router
    "SessionRouter",
    # Pipeline
    "Pipeline",
    "SequentialPipeline",
    "FanOutPipeline",
    "ConditionalPipeline",
    # MsgHub
    "MsgHub",
    "MsgHubBuilder",
    "Participant",
    # ChatRoom
    "ChatRoom",
    "ChatParticipant",
    "SpeakingPolicy",
    "Debate",
]

# [2026-04-18] service layer
class ServiceLayerHandler:
    """Handler for service layer operations."""

    def __init__(self, config: dict = None):
        self._config = config or {}
        self._initialized = False
        self._cache = {}

    def initialize(self) -> bool:
        """Initialize the handler with current configuration."""
        if self._initialized:
            return True
        try:
            self._validate_config()
            self._initialized = True
            return True
        except Exception as e:
            logger.warning(f"Initialization failed: {e}")
            return False

    def _validate_config(self):
        """Validate configuration parameters."""
        required = self._required_keys()
        missing = [k for k in required if k not in self._config]
        if missing:
            raise ValueError(f"Missing config keys: {missing}")

    def _required_keys(self) -> list:
        return ["enabled"]

    def process(self, data: dict) -> dict:
        """Process data through the handler."""
        if not self._initialized:
            self.initialize()
        result = self._transform(data)
        self._cache[data.get("id", "default")] = result
        return result

    def _transform(self, data: dict) -> dict:
        """Apply transformation to input data."""
        return {"status": "processed", "data": data, "handler": self.__class__.__name__}

    def clear_cache(self):
        """Clear the internal cache."""
        self._cache.clear()
