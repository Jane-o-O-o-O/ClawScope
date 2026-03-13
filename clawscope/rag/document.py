"""Document types for RAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
import hashlib


@dataclass
class Document:
    """
    Represents a source document.

    A document is the original content before splitting,
    containing metadata about its source.
    """

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str | None = None
    doc_id: str | None = None
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if self.doc_id is None:
            # Generate ID from content hash
            content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:16]
            self.doc_id = f"doc_{content_hash}"

    @classmethod
    def from_file(cls, path: str | Path, encoding: str = "utf-8") -> "Document":
        """Create document from file."""
        path = Path(path)
        content = path.read_text(encoding=encoding)
        return cls(
            content=content,
            source=str(path),
            metadata={
                "filename": path.name,
                "extension": path.suffix,
                "size": path.stat().st_size,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "metadata": self.metadata,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        """Create from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        return cls(
            doc_id=data.get("doc_id"),
            content=data.get("content", ""),
            metadata=data.get("metadata", {}),
            source=data.get("source"),
            created_at=created_at or datetime.now(),
        )


@dataclass
class DocumentChunk:
    """
    A chunk of a document after splitting.

    Contains the text content and maintains reference
    to the parent document.
    """

    content: str
    chunk_id: str | None = None
    doc_id: str | None = None
    chunk_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None

    def __post_init__(self):
        if self.chunk_id is None:
            content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:12]
            self.chunk_id = f"chunk_{self.doc_id}_{self.chunk_index}_{content_hash}"

    @property
    def has_embedding(self) -> bool:
        """Check if chunk has embedding."""
        return self.embedding is not None and len(self.embedding) > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "content": self.content,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
            "embedding": self.embedding,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentChunk":
        """Create from dictionary."""
        return cls(
            chunk_id=data.get("chunk_id"),
            doc_id=data.get("doc_id"),
            content=data.get("content", ""),
            chunk_index=data.get("chunk_index", 0),
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding"),
        )


@dataclass
class SearchResult:
    """Result from vector search."""

    chunk: DocumentChunk
    score: float
    distance: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk": self.chunk.to_dict(),
            "score": self.score,
            "distance": self.distance,
        }


__all__ = ["Document", "DocumentChunk", "SearchResult"]
