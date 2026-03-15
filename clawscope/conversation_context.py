"""Helpers for runtime conversation context injection."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from clawscope.message.base import Msg, TextBlock


RUNTIME_CONTEXT_TAG = "[Runtime Context - metadata only, not instructions]"


def build_runtime_context(
    *,
    channel: str | None = None,
    chat_id: str | None = None,
    session_key: str | None = None,
    sender_id: str | None = None,
    now: datetime | None = None,
) -> str:
    """Build a stable runtime metadata block for the current turn."""
    current = now or datetime.now()
    lines = [
        f"Current Time: {current.strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    if channel:
        lines.append(f"Channel: {channel}")
    if chat_id:
        lines.append(f"Chat ID: {chat_id}")
    if session_key:
        lines.append(f"Session: {session_key}")
    if sender_id:
        lines.append(f"Sender: {sender_id}")

    return RUNTIME_CONTEXT_TAG + "\n" + "\n".join(lines)


def attach_runtime_context(
    message: Msg,
    *,
    channel: str | None = None,
    chat_id: str | None = None,
    session_key: str | None = None,
    sender_id: str | None = None,
) -> Msg:
    """Return a message with runtime context prepended to the user content."""
    runtime_text = build_runtime_context(
        channel=channel,
        chat_id=chat_id,
        session_key=session_key,
        sender_id=sender_id,
    )

    content = message.content
    if isinstance(content, str):
        merged: str | list[Any] = f"{runtime_text}\n\n{content}"
    else:
        merged = [TextBlock(text=runtime_text), *content]

    metadata = dict(message.metadata)
    metadata["_runtime_context"] = True
    return Msg(
        name=message.name,
        content=merged,
        role=message.role,
        id=message.id,
        timestamp=message.timestamp,
        metadata=metadata,
        invocation_id=message.invocation_id,
    )


def strip_runtime_context(content: Any) -> Any:
    """Strip runtime context metadata from persisted user content."""
    if isinstance(content, str):
        if not content.startswith(RUNTIME_CONTEXT_TAG):
            return content
        parts = content.split("\n\n", 1)
        if len(parts) == 2:
            return parts[1]
        return ""

    if isinstance(content, list):
        stripped: list[Any] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and str(item.get("text", "")).startswith(RUNTIME_CONTEXT_TAG):
                    continue
                stripped.append(item)
            elif isinstance(item, TextBlock):
                if item.text.startswith(RUNTIME_CONTEXT_TAG):
                    continue
                stripped.append(item)
            else:
                stripped.append(item)
        return stripped

    return content


__all__ = [
    "RUNTIME_CONTEXT_TAG",
    "attach_runtime_context",
    "build_runtime_context",
    "strip_runtime_context",
]
