"""Vector stores for RAG."""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from loguru import logger

from clawscope.rag.document import DocumentChunk, SearchResult


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    async def add(self, chunks: list[DocumentChunk]) -> None:
        """Add chunks to the store."""
        pass

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar chunks."""
        pass

    @abstractmethod
    async def delete(self, chunk_ids: list[str]) -> None:
        """Delete chunks by IDs."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all chunks."""
        pass

    @property
    @abstractmethod
    def count(self) -> int:
        """Get number of chunks in store."""
        pass


class InMemoryVectorStore(VectorStore):
    """In-memory vector store using cosine similarity."""

    def __init__(self, persist_path: Path | None = None):
        """
        Initialize in-memory store.

        Args:
            persist_path: Optional path for persistence
        """
        self.persist_path = persist_path
        self._chunks: dict[str, DocumentChunk] = {}

        if persist_path and persist_path.exists():
            self._load()

    async def add(self, chunks: list[DocumentChunk]) -> None:
        """Add chunks to the store."""
        for chunk in chunks:
            if not chunk.has_embedding:
                logger.warning(f"Chunk {chunk.chunk_id} has no embedding, skipping")
                continue
            self._chunks[chunk.chunk_id] = chunk

        if self.persist_path:
            self._save()

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar chunks using cosine similarity."""
        results = []

        for chunk in self._chunks.values():
            if not chunk.has_embedding:
                continue

            # Apply filter
            if filter and not self._matches_filter(chunk, filter):
                continue

            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, chunk.embedding)
            results.append(SearchResult(
                chunk=chunk,
                score=similarity,
                distance=1 - similarity,
            ))

        # Sort by similarity (descending)
        results.sort(key=lambda x: x.score, reverse=True)

        return results[:top_k]

    async def delete(self, chunk_ids: list[str]) -> None:
        """Delete chunks by IDs."""
        for chunk_id in chunk_ids:
            self._chunks.pop(chunk_id, None)

        if self.persist_path:
            self._save()

    async def clear(self) -> None:
        """Clear all chunks."""
        self._chunks.clear()

        if self.persist_path and self.persist_path.exists():
            self.persist_path.unlink()

    @property
    def count(self) -> int:
        """Get number of chunks in store."""
        return len(self._chunks)

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def _matches_filter(self, chunk: DocumentChunk, filter: dict[str, Any]) -> bool:
        """Check if chunk matches filter criteria."""
        for key, value in filter.items():
            if key == "doc_id" and chunk.doc_id != value:
                return False
            elif key in chunk.metadata and chunk.metadata[key] != value:
                return False
        return True

    def _save(self) -> None:
        """Save to disk."""
        if not self.persist_path:
            return

        self.persist_path.parent.mkdir(parents=True, exist_ok=True)

        data = [chunk.to_dict() for chunk in self._chunks.values()]
        self.persist_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def _load(self) -> None:
        """Load from disk."""
        if not self.persist_path or not self.persist_path.exists():
            return

        try:
            data = json.loads(self.persist_path.read_text())
            for item in data:
                chunk = DocumentChunk.from_dict(item)
                self._chunks[chunk.chunk_id] = chunk
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")


class QdrantStore(VectorStore):
    """Qdrant vector store."""

    def __init__(
        self,
        collection_name: str = "clawscope",
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        dimension: int = 1536,
    ):
        """
        Initialize Qdrant store.

        Args:
            collection_name: Collection name
            url: Qdrant server URL
            api_key: API key for cloud
            dimension: Vector dimension
        """
        self.collection_name = collection_name
        self.url = url
        self.api_key = api_key
        self.dimension = dimension
        self._client = None

    def _get_client(self):
        """Get or create Qdrant client."""
        if self._client is None:
            try:
                from qdrant_client import QdrantClient

                self._client = QdrantClient(
                    url=self.url,
                    api_key=self.api_key,
                )

                # Ensure collection exists
                self._ensure_collection()

            except ImportError:
                raise ImportError(
                    "qdrant-client required. Install with: pip install qdrant-client"
                )

        return self._client

    def _ensure_collection(self) -> None:
        """Ensure collection exists."""
        from qdrant_client.models import Distance, VectorParams

        client = self._client
        collections = client.get_collections().collections
        names = [c.name for c in collections]

        if self.collection_name not in names:
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection: {self.collection_name}")

    async def add(self, chunks: list[DocumentChunk]) -> None:
        """Add chunks to Qdrant."""
        from qdrant_client.models import PointStruct

        client = self._get_client()

        points = []
        for chunk in chunks:
            if not chunk.has_embedding:
                continue

            points.append(PointStruct(
                id=chunk.chunk_id,
                vector=chunk.embedding,
                payload={
                    "content": chunk.content,
                    "doc_id": chunk.doc_id,
                    "chunk_index": chunk.chunk_index,
                    "metadata": chunk.metadata,
                },
            ))

        if points:
            client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search Qdrant."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        client = self._get_client()

        # Build filter
        qdrant_filter = None
        if filter:
            conditions = []
            for key, value in filter.items():
                conditions.append(FieldCondition(
                    key=key,
                    match=MatchValue(value=value),
                ))
            qdrant_filter = Filter(must=conditions)

        results = client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=qdrant_filter,
        )

        search_results = []
        for result in results:
            chunk = DocumentChunk(
                chunk_id=result.id,
                content=result.payload.get("content", ""),
                doc_id=result.payload.get("doc_id"),
                chunk_index=result.payload.get("chunk_index", 0),
                metadata=result.payload.get("metadata", {}),
                embedding=None,  # Don't store embedding in result
            )
            search_results.append(SearchResult(
                chunk=chunk,
                score=result.score,
                distance=1 - result.score,
            ))

        return search_results

    async def delete(self, chunk_ids: list[str]) -> None:
        """Delete chunks from Qdrant."""
        client = self._get_client()
        client.delete(
            collection_name=self.collection_name,
            points_selector=chunk_ids,
        )

    async def clear(self) -> None:
        """Clear collection."""
        client = self._get_client()
        client.delete_collection(self.collection_name)
        self._ensure_collection()

    @property
    def count(self) -> int:
        """Get collection count."""
        client = self._get_client()
        info = client.get_collection(self.collection_name)
        return info.points_count


class MilvusStore(VectorStore):
    """Milvus vector store."""

    def __init__(
        self,
        collection_name: str = "clawscope",
        uri: str = "http://localhost:19530",
        token: str | None = None,
        dimension: int = 1536,
    ):
        """
        Initialize Milvus store.

        Args:
            collection_name: Collection name
            uri: Milvus server URI
            token: Auth token
            dimension: Vector dimension
        """
        self.collection_name = collection_name
        self.uri = uri
        self.token = token
        self.dimension = dimension
        self._client = None

    def _get_client(self):
        """Get or create Milvus client."""
        if self._client is None:
            try:
                from pymilvus import MilvusClient

                self._client = MilvusClient(
                    uri=self.uri,
                    token=self.token,
                )

                # Ensure collection exists
                self._ensure_collection()

            except ImportError:
                raise ImportError(
                    "pymilvus required. Install with: pip install pymilvus"
                )

        return self._client

    def _ensure_collection(self) -> None:
        """Ensure collection exists."""
        client = self._client

        if not client.has_collection(self.collection_name):
            client.create_collection(
                collection_name=self.collection_name,
                dimension=self.dimension,
                metric_type="COSINE",
            )
            logger.info(f"Created Milvus collection: {self.collection_name}")

    async def add(self, chunks: list[DocumentChunk]) -> None:
        """Add chunks to Milvus."""
        client = self._get_client()

        data = []
        for chunk in chunks:
            if not chunk.has_embedding:
                continue

            data.append({
                "id": chunk.chunk_id,
                "vector": chunk.embedding,
                "content": chunk.content,
                "doc_id": chunk.doc_id or "",
                "chunk_index": chunk.chunk_index,
                "metadata": json.dumps(chunk.metadata),
            })

        if data:
            client.insert(
                collection_name=self.collection_name,
                data=data,
            )

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search Milvus."""
        client = self._get_client()

        # Build filter expression
        filter_expr = None
        if filter:
            conditions = []
            for key, value in filter.items():
                if isinstance(value, str):
                    conditions.append(f'{key} == "{value}"')
                else:
                    conditions.append(f'{key} == {value}')
            filter_expr = " and ".join(conditions)

        results = client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=top_k,
            filter=filter_expr,
            output_fields=["content", "doc_id", "chunk_index", "metadata"],
        )

        search_results = []
        for hits in results:
            for hit in hits:
                metadata = {}
                try:
                    metadata = json.loads(hit.get("metadata", "{}"))
                except Exception:
                    pass

                chunk = DocumentChunk(
                    chunk_id=hit.get("id"),
                    content=hit.get("content", ""),
                    doc_id=hit.get("doc_id"),
                    chunk_index=hit.get("chunk_index", 0),
                    metadata=metadata,
                )
                search_results.append(SearchResult(
                    chunk=chunk,
                    score=1 - hit.get("distance", 0),
                    distance=hit.get("distance"),
                ))

        return search_results

    async def delete(self, chunk_ids: list[str]) -> None:
        """Delete chunks from Milvus."""
        client = self._get_client()
        client.delete(
            collection_name=self.collection_name,
            ids=chunk_ids,
        )

    async def clear(self) -> None:
        """Clear collection."""
        client = self._get_client()
        client.drop_collection(self.collection_name)
        self._ensure_collection()

    @property
    def count(self) -> int:
        """Get collection count."""
        client = self._get_client()
        stats = client.get_collection_stats(self.collection_name)
        return stats.get("row_count", 0)


__all__ = [
    "VectorStore",
    "InMemoryVectorStore",
    "QdrantStore",
    "MilvusStore",
]
