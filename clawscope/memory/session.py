"""Session management for ClawScope."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles
from loguru import logger

from clawscope.conversation_context import strip_runtime_context
from clawscope.memory.base import MemoryBase
from clawscope.message import Msg


@dataclass
class Session:
    """
    Represents a conversation session.

    Sessions are identified by a key (typically channel:chat_id)
    and store message history.
    """

    key: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_consolidated: int = 0  # Index of last consolidated message
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_message(
        self,
        role: str,
        content: Any,
        name: str | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """Add a message to the session."""
        self.messages.append({
            "role": role,
            "content": content,
            "name": name or role,
            "timestamp": (timestamp or datetime.now()).isoformat(),
        })
        self.updated_at = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "messages": self.messages,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_consolidated": self.last_consolidated,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        """Create from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        else:
            created_at = datetime.now()

        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        else:
            updated_at = datetime.now()

        return cls(
            key=data.get("key", ""),
            messages=data.get("messages", []),
            created_at=created_at,
            updated_at=updated_at,
            last_consolidated=data.get("last_consolidated", 0),
            metadata=data.get("metadata", {}),
        )


class SessionManager:
    """
    Manages session persistence.

    Stores sessions as JSONL files for append-only efficiency.
    """

    def __init__(
        self,
        workspace: Path,
        backend: str = "jsonl",
    ):
        """
        Initialize session manager.

        Args:
            workspace: Workspace directory
            backend: Storage backend (jsonl, redis, sqlite)
        """
        self.workspace = workspace
        self.backend = backend
        self.sessions_dir = workspace / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, Session] = {}

    async def get_or_create(self, session_key: str) -> Session:
        """Get existing session or create new one."""
        if session_key in self._cache:
            return self._cache[session_key]

        # Try to load from file
        session = await self._load(session_key)
        if session is None:
            session = Session(key=session_key)

        self._cache[session_key] = session
        return session

    async def save(self, session: Session) -> None:
        """Save session to storage."""
        self._cache[session.key] = session
        await self._save(session)

    async def delete(self, session_key: str) -> None:
        """Delete a session."""
        if session_key in self._cache:
            del self._cache[session_key]

        file_path = self._session_file_path(session_key)
        if file_path.exists():
            file_path.unlink()

    def _session_file_path(self, session_key: str) -> Path:
        """Get file path for session."""
        # Sanitize key for filename
        safe_key = session_key.replace(":", "_").replace("/", "_")
        return self.sessions_dir / f"{safe_key}.jsonl"

    async def _load(self, session_key: str) -> Session | None:
        """Load session from file."""
        file_path = self._session_file_path(session_key)
        if not file_path.exists():
            return None

        try:
            messages = []
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                async for line in f:
                    line = line.strip()
                    if line:
                        messages.append(json.loads(line))

            session = Session(key=session_key, messages=messages)
            logger.debug(f"Loaded session {session_key} with {len(messages)} messages")
            return session

        except Exception as e:
            logger.error(f"Error loading session {session_key}: {e}")
            return None

    async def _save(self, session: Session) -> None:
        """Save session to file (append-only)."""
        file_path = self._session_file_path(session.key)

        try:
            # For JSONL, we append new messages
            async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                for msg in session.messages:
                    await f.write(json.dumps(msg, ensure_ascii=False) + "\n")

            logger.debug(f"Saved session {session.key}")

        except Exception as e:
            logger.error(f"Error saving session {session.key}: {e}")

    def list_sessions(self) -> list[str]:
        """List all session keys."""
        sessions = []
        for file_path in self.sessions_dir.glob("*.jsonl"):
            # Convert filename back to session key
            key = file_path.stem.replace("_", ":", 1)
            sessions.append(key)
        return sessions


class SessionMemory(MemoryBase):
    """
    Memory implementation backed by a Session.

    Provides the MemoryBase interface for session-based storage.
    """

    def __init__(self, session: Session):
        """
        Initialize session memory.

        Args:
            session: Session to use for storage
        """
        self.session = session

    async def add(self, messages: list[Msg], mark: str | None = None) -> None:
        """Add messages to session."""
        for msg in messages:
            content = strip_runtime_context(msg.content)
            if isinstance(content, list):
                # Convert content blocks to dicts
                content = [
                    block.to_dict() if hasattr(block, "to_dict") else block
                    for block in content
                ]

            self.session.add_message(
                role=msg.role,
                content=content,
                name=msg.name,
            )

    async def get(
        self,
        mark: str | None = None,
        limit: int | None = None,
    ) -> list[Msg]:
        """Get messages from session."""
        messages = self.session.messages

        if limit:
            messages = messages[-limit:]

        return [
            Msg(
                name=m.get("name", m.get("role", "")),
                content=m.get("content", ""),
                role=m.get("role", "user"),
                timestamp=m.get("timestamp", ""),
            )
            for m in messages
        ]

    async def clear(self) -> None:
        """Clear session messages."""
        self.session.messages.clear()

    async def size(self) -> int:
        """Get number of messages."""
        return len(self.session.messages)


__all__ = ["Session", "SessionManager", "SessionMemory"]
