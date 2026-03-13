"""Unified memory system for ClawScope."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

import aiofiles
from loguru import logger

from clawscope.memory.base import MemoryBase
from clawscope.memory.working import InMemoryMemory
from clawscope.memory.session import Session, SessionMemory
from clawscope.message import Msg

if TYPE_CHECKING:
    from clawscope.model import ChatModelBase


class UnifiedMemory(MemoryBase):
    """
    Unified memory system combining working memory, session storage,
    and long-term memory.

    Architecture:
    - Working Memory: Fast, in-memory message storage
    - Session Storage: Persistent JSONL-based history
    - Long-term Memory: MEMORY.md + HISTORY.md files

    Features:
    - Automatic consolidation from working to long-term
    - Grep-searchable history log
    - LLM-driven memory summarization
    """

    def __init__(
        self,
        workspace: Path,
        session: Session | None = None,
        working_memory: MemoryBase | None = None,
    ):
        """
        Initialize unified memory.

        Args:
            workspace: Workspace directory for persistent storage
            session: Optional session for history
            working_memory: Optional working memory backend
        """
        self.workspace = workspace
        self.memory_dir = workspace / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"

        # Memory layers
        self.working = working_memory or InMemoryMemory()
        self.session = SessionMemory(session) if session else None

    async def add(self, messages: list[Msg], mark: str | None = None) -> None:
        """Add messages to memory."""
        # Add to working memory
        await self.working.add(messages, mark)

        # Add to session if available
        if self.session:
            await self.session.add(messages, mark)

    async def get(
        self,
        mark: str | None = None,
        limit: int | None = None,
        prepend_memory: bool = True,
    ) -> list[Msg]:
        """
        Get messages from memory.

        Args:
            mark: Optional marker to filter by
            limit: Maximum number of messages
            prepend_memory: Whether to prepend long-term memory

        Returns:
            List of messages
        """
        messages = []

        # Prepend long-term memory as system message
        if prepend_memory:
            memory_content = await self.load_long_term_memory()
            if memory_content:
                messages.append(Msg(
                    name="system",
                    content=f"[Long-term Memory]\n{memory_content}",
                    role="system",
                ))

        # Get working memory
        working_messages = await self.working.get(mark, limit)
        messages.extend(working_messages)

        return messages

    async def clear(self) -> None:
        """Clear working memory."""
        await self.working.clear()

    async def size(self) -> int:
        """Get number of messages in working memory."""
        return await self.working.size()

    # ========== Long-term Memory ==========

    async def load_long_term_memory(self) -> str:
        """Load MEMORY.md content."""
        if not self.memory_file.exists():
            return ""

        try:
            async with aiofiles.open(self.memory_file, "r", encoding="utf-8") as f:
                return await f.read()
        except Exception as e:
            logger.error(f"Error loading memory: {e}")
            return ""

    async def save_long_term_memory(self, content: str) -> None:
        """Save to MEMORY.md."""
        try:
            async with aiofiles.open(self.memory_file, "w", encoding="utf-8") as f:
                await f.write(content)
            logger.debug("Saved long-term memory")
        except Exception as e:
            logger.error(f"Error saving memory: {e}")

    async def append_to_history(self, entry: str) -> None:
        """Append entry to HISTORY.md."""
        from datetime import datetime

        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M]")
        line = f"{timestamp} {entry}\n"

        try:
            async with aiofiles.open(self.history_file, "a", encoding="utf-8") as f:
                await f.write(line)
        except Exception as e:
            logger.error(f"Error appending to history: {e}")

    async def search_history(self, query: str) -> list[str]:
        """Search HISTORY.md for matching entries."""
        if not self.history_file.exists():
            return []

        matches = []
        try:
            async with aiofiles.open(self.history_file, "r", encoding="utf-8") as f:
                async for line in f:
                    if query.lower() in line.lower():
                        matches.append(line.strip())
        except Exception as e:
            logger.error(f"Error searching history: {e}")

        return matches

    # ========== Memory Consolidation ==========

    async def consolidate(
        self,
        model: "ChatModelBase | None" = None,
        max_messages: int = 50,
    ) -> bool:
        """
        Consolidate working memory to long-term storage.

        Uses LLM to summarize key information if model provided.

        Args:
            model: Optional model for summarization
            max_messages: Max messages to consolidate at once

        Returns:
            True if consolidation succeeded
        """
        # Get messages to consolidate
        messages = await self.working.get(limit=max_messages)
        if not messages:
            return True

        try:
            if model:
                # LLM-driven consolidation
                summary = await self._generate_summary(messages, model)

                # Update MEMORY.md
                current_memory = await self.load_long_term_memory()
                updated_memory = self._merge_memory(current_memory, summary)
                await self.save_long_term_memory(updated_memory)

            # Add timeline entry to HISTORY.md
            entry = f"Consolidated {len(messages)} messages"
            await self.append_to_history(entry)

            logger.info(f"Consolidated {len(messages)} messages")
            return True

        except Exception as e:
            logger.error(f"Consolidation error: {e}")
            return False

    async def _generate_summary(
        self,
        messages: list[Msg],
        model: "ChatModelBase",
    ) -> str:
        """Generate summary of messages using LLM."""
        # Build prompt
        conversation = "\n".join([
            f"{m.name} ({m.role}): {m.get_text_content()}"
            for m in messages
        ])

        prompt = Msg(
            name="system",
            content=f"""Summarize the key facts and decisions from this conversation.
Focus on:
- User preferences learned
- Important facts mentioned
- Decisions made
- Tasks completed or pending

Conversation:
{conversation}

Output a concise summary in markdown format.""",
            role="user",
        )

        response = await model.chat([prompt])
        return response.content or ""

    def _merge_memory(self, current: str, new_summary: str) -> str:
        """Merge new summary into existing memory."""
        if not current:
            return f"# Memory\n\n{new_summary}"

        # Simple append for now
        return f"{current}\n\n---\n\n{new_summary}"


__all__ = ["UnifiedMemory"]
