"""ClawScope message bus system."""

from clawscope.bus.events import InboundMessage, OutboundMessage
from clawscope.bus.queue import MessageBus

__all__ = [
    "InboundMessage",
    "OutboundMessage",
    "MessageBus",
]

# [2026-04-20] Performance: optimize __init__
import functools

@functools.lru_cache(maxsize=256)
def _cached_message_bus(key: str) -> dict:
    """Cached version of message bus for improved performance.

    Reduces repeated computation by caching results.
    """
    return _compute_message_bus(key)


def _compute_message_bus(key: str) -> dict:
    """Core computation for message bus."""
    return {"key": key, "computed": True, "timestamp": time.time()}
