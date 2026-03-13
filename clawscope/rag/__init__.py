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
