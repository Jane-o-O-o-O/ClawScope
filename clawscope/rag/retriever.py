"""Retrievers for RAG."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from clawscope.rag.document import DocumentChunk, SearchResult
from clawscope.rag.embedding import EmbeddingProvider
from clawscope.rag.store import VectorStore


class Retriever(ABC):
    """Abstract base class for retrievers."""

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Retrieve relevant chunks for query."""
        pass


class VectorRetriever(Retriever):
    """Vector-based retriever using embeddings."""

    def __init__(
        self,
        store: VectorStore,
        embedding: EmbeddingProvider,
        top_k: int = 5,
        score_threshold: float | None = None,
    ):
        """
        Initialize vector retriever.

        Args:
            store: Vector store
            embedding: Embedding provider
            top_k: Default number of results
            score_threshold: Minimum score threshold
        """
        self.store = store
        self.embedding = embedding
        self.top_k = top_k
        self.score_threshold = score_threshold

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Retrieve relevant chunks."""
        # Embed query
        query_embedding = await self.embedding.embed(query)

        # Search store
        results = await self.store.search(
            query_embedding=query_embedding,
            top_k=top_k or self.top_k,
            filter=filter,
        )

        # Apply score threshold
        if self.score_threshold is not None:
            results = [r for r in results if r.score >= self.score_threshold]

        return results

    async def retrieve_with_context(
        self,
        query: str,
        top_k: int | None = None,
        filter: dict[str, Any] | None = None,
        max_tokens: int = 4000,
    ) -> str:
        """
        Retrieve and format context for LLM.

        Args:
            query: Search query
            top_k: Number of results
            filter: Filter criteria
            max_tokens: Maximum tokens in context

        Returns:
            Formatted context string
        """
        results = await self.retrieve(query, top_k, filter)

        if not results:
            return ""

        context_parts = []
        total_chars = 0
        char_limit = max_tokens * 4  # Rough estimate

        for i, result in enumerate(results):
            chunk_text = f"[{i + 1}] {result.chunk.content}"

            if total_chars + len(chunk_text) > char_limit:
                break

            context_parts.append(chunk_text)
            total_chars += len(chunk_text)

        return "\n\n".join(context_parts)


class HybridRetriever(Retriever):
    """Hybrid retriever combining vector and keyword search."""

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        keyword_weight: float = 0.3,
        vector_weight: float = 0.7,
    ):
        """
        Initialize hybrid retriever.

        Args:
            vector_retriever: Vector retriever
            keyword_weight: Weight for keyword matching
            vector_weight: Weight for vector similarity
        """
        self.vector_retriever = vector_retriever
        self.keyword_weight = keyword_weight
        self.vector_weight = vector_weight

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Retrieve using hybrid approach."""
        # Get vector results
        vector_results = await self.vector_retriever.retrieve(
            query, top_k * 2, filter
        )

        # Calculate keyword scores
        query_terms = set(query.lower().split())

        for result in vector_results:
            # Simple keyword matching
            content_terms = set(result.chunk.content.lower().split())
            overlap = len(query_terms & content_terms)
            keyword_score = overlap / max(len(query_terms), 1)

            # Combine scores
            combined_score = (
                self.vector_weight * result.score +
                self.keyword_weight * keyword_score
            )
            result.score = combined_score

        # Re-sort and limit
        vector_results.sort(key=lambda x: x.score, reverse=True)
        return vector_results[:top_k]


class ReRankRetriever(Retriever):
    """Retriever with re-ranking using cross-encoder."""

    def __init__(
        self,
        base_retriever: Retriever,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k_initial: int = 20,
    ):
        """
        Initialize re-ranking retriever.

        Args:
            base_retriever: Base retriever
            model: Cross-encoder model
            top_k_initial: Initial retrieval count before reranking
        """
        self.base_retriever = base_retriever
        self.model_name = model
        self.top_k_initial = top_k_initial
        self._model = None

    def _get_model(self):
        """Get or create cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers required for re-ranking. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Retrieve and re-rank results."""
        # Get initial results
        results = await self.base_retriever.retrieve(
            query, self.top_k_initial, filter
        )

        if not results:
            return []

        # Re-rank using cross-encoder
        model = self._get_model()

        pairs = [(query, r.chunk.content) for r in results]
        scores = model.predict(pairs)

        # Update scores
        for result, score in zip(results, scores):
            result.score = float(score)

        # Sort by new scores
        results.sort(key=lambda x: x.score, reverse=True)

        return results[:top_k]


class ContextualRetriever(Retriever):
    """
    Retriever that adds contextual information to chunks.

    Based on Anthropic's Contextual Retrieval approach.
    """

    def __init__(
        self,
        base_retriever: Retriever,
        model: Any = None,
        context_template: str | None = None,
    ):
        """
        Initialize contextual retriever.

        Args:
            base_retriever: Base retriever
            model: LLM model for generating context
            context_template: Template for context generation
        """
        self.base_retriever = base_retriever
        self.model = model
        self.context_template = context_template or (
            "Given the following document chunk, provide a brief context "
            "that situates this chunk within the broader document:\n\n"
            "Chunk: {chunk}\n\n"
            "Context:"
        )

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Retrieve with contextual enhancement."""
        results = await self.base_retriever.retrieve(query, top_k, filter)

        if not results or not self.model:
            return results

        # Add context to each result
        for result in results:
            if "context" not in result.chunk.metadata:
                # Generate context using LLM
                prompt = self.context_template.format(chunk=result.chunk.content)

                try:
                    from clawscope.message import Msg
                    msg = Msg(name="user", content=prompt, role="user")
                    response = await self.model(msg)
                    context = response.get_text_content()
                    result.chunk.metadata["context"] = context
                except Exception:
                    pass

        return results


__all__ = [
    "Retriever",
    "VectorRetriever",
    "HybridRetriever",
    "ReRankRetriever",
    "ContextualRetriever",
]
