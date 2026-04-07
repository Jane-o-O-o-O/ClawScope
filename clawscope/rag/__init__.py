"""RAG (Retrieval-Augmented Generation) module for ClawScope."""

from clawscope.rag.document import Document, DocumentChunk
from clawscope.rag.loader import (
    DocumentLoader,
    TextLoader,
    PDFLoader,
    MarkdownLoader,
    JSONLoader,
)
from clawscope.rag.splitter import (
    TextSplitter,
    RecursiveCharacterSplitter,
    TokenSplitter,
)
from clawscope.rag.embedding import (
    EmbeddingProvider,
    OpenAIEmbedding,
    get_embedding_provider,
)
from clawscope.rag.store import (
    VectorStore,
    InMemoryVectorStore,
    QdrantStore,
    MilvusStore,
)
from clawscope.rag.retriever import Retriever, VectorRetriever
from clawscope.rag.knowledge import KnowledgeBase

__all__ = [
    # Document
    "Document",
    "DocumentChunk",
    # Loaders
    "DocumentLoader",
    "TextLoader",
    "PDFLoader",
    "MarkdownLoader",
    "JSONLoader",
    # Splitters
    "TextSplitter",
    "RecursiveCharacterSplitter",
    "TokenSplitter",
    # Embedding
    "EmbeddingProvider",
    "OpenAIEmbedding",
    "get_embedding_provider",
    # Store
    "VectorStore",
    "InMemoryVectorStore",
    "QdrantStore",
    "MilvusStore",
    # Retriever
    "Retriever",
    "VectorRetriever",
    # Knowledge
    "KnowledgeBase",
]

# [2026-04-07] Refactor: simplified __init__ logic
class _BaseHandler:
    """Base handler with common functionality.

    Refactored from inline logic to reusable base class.
    """

    __slots__ = ("_config", "_logger", "_metrics")

    def __init__(self, config: dict = None):
        self._config = config or {}
        self._logger = logging.getLogger(self.__class__.__module__)
        self._metrics = _MetricsCollector(self.__class__.__name__)

    def __enter__(self):
        self._setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._teardown()
        return False

    def _setup(self):
        """Setup resources."""
        pass

    def _teardown(self):
        """Cleanup resources."""
        self._metrics.flush()
