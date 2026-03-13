"""Embedding providers for RAG."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

from loguru import logger


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get embedding dimension."""
        pass

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Embed single text."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        pass


class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI embedding provider."""

    DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        api_base: str | None = None,
        batch_size: int = 100,
    ):
        """
        Initialize OpenAI embedding provider.

        Args:
            model: Embedding model name
            api_key: OpenAI API key
            api_base: Custom API base URL
            batch_size: Maximum batch size
        """
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.batch_size = batch_size
        self._client = None

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self.DIMENSIONS.get(self.model, 1536)

    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            from openai import AsyncOpenAI

            kwargs = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.api_base:
                kwargs["base_url"] = self.api_base

            self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def embed(self, text: str) -> list[float]:
        """Embed single text."""
        embeddings = await self.embed_batch([text])
        return embeddings[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        client = self._get_client()
        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            try:
                response = await client.embeddings.create(
                    model=self.model,
                    input=batch,
                )
                batch_embeddings = [e.embedding for e in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Embedding error: {e}")
                # Return zero vectors on error
                all_embeddings.extend([[0.0] * self.dimension] * len(batch))

        return all_embeddings


class AnthropicVoyageEmbedding(EmbeddingProvider):
    """Voyage AI embedding provider (Anthropic partner)."""

    DIMENSIONS = {
        "voyage-large-2": 1536,
        "voyage-code-2": 1536,
        "voyage-2": 1024,
        "voyage-lite-02-instruct": 1024,
    }

    def __init__(
        self,
        model: str = "voyage-large-2",
        api_key: str | None = None,
        batch_size: int = 128,
    ):
        """
        Initialize Voyage embedding provider.

        Args:
            model: Embedding model name
            api_key: Voyage API key
            batch_size: Maximum batch size
        """
        self.model = model
        self.api_key = api_key
        self.batch_size = batch_size
        self._client = None

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self.DIMENSIONS.get(self.model, 1536)

    async def embed(self, text: str) -> list[float]:
        """Embed single text."""
        embeddings = await self.embed_batch([text])
        return embeddings[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        try:
            import voyageai

            if self._client is None:
                self._client = voyageai.AsyncClient(api_key=self.api_key)

            all_embeddings = []

            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                result = await self._client.embed(batch, model=self.model)
                all_embeddings.extend(result.embeddings)

            return all_embeddings

        except ImportError:
            logger.error("voyageai not installed")
            return [[0.0] * self.dimension] * len(texts)
        except Exception as e:
            logger.error(f"Voyage embedding error: {e}")
            return [[0.0] * self.dimension] * len(texts)


class LocalEmbedding(EmbeddingProvider):
    """Local embedding using sentence-transformers."""

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
    ):
        """
        Initialize local embedding provider.

        Args:
            model: Model name from sentence-transformers
            device: Device to use (cpu, cuda, mps)
        """
        self.model_name = model
        self.device = device
        self._model = None
        self._dimension = None

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self._dimension is None:
            model = self._get_model()
            self._dimension = model.get_sentence_embedding_dimension()
        return self._dimension

    def _get_model(self):
        """Get or create model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except ImportError:
                raise ImportError(
                    "sentence-transformers required for local embeddings. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    async def embed(self, text: str) -> list[float]:
        """Embed single text."""
        embeddings = await self.embed_batch([text])
        return embeddings[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        model = self._get_model()

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: model.encode(texts, convert_to_numpy=True).tolist(),
        )

        return embeddings


def get_embedding_provider(
    provider: str = "openai",
    **kwargs: Any,
) -> EmbeddingProvider:
    """
    Get embedding provider by name.

    Args:
        provider: Provider name (openai, voyage, local)
        **kwargs: Provider-specific arguments

    Returns:
        EmbeddingProvider instance
    """
    providers = {
        "openai": OpenAIEmbedding,
        "voyage": AnthropicVoyageEmbedding,
        "local": LocalEmbedding,
    }

    if provider not in providers:
        raise ValueError(f"Unknown embedding provider: {provider}")

    return providers[provider](**kwargs)


__all__ = [
    "EmbeddingProvider",
    "OpenAIEmbedding",
    "AnthropicVoyageEmbedding",
    "LocalEmbedding",
    "get_embedding_provider",
]
