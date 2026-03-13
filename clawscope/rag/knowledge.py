"""Knowledge base management for RAG."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from loguru import logger

from clawscope.rag.document import Document, DocumentChunk, SearchResult
from clawscope.rag.loader import DocumentLoader, DirectoryLoader
from clawscope.rag.splitter import TextSplitter, RecursiveCharacterSplitter
from clawscope.rag.embedding import EmbeddingProvider, OpenAIEmbedding
from clawscope.rag.store import VectorStore, InMemoryVectorStore
from clawscope.rag.retriever import VectorRetriever


class KnowledgeBase:
    """
    High-level knowledge base for RAG.

    Combines document loading, splitting, embedding, and retrieval
    into a simple interface.
    """

    def __init__(
        self,
        name: str = "default",
        store: VectorStore | None = None,
        embedding: EmbeddingProvider | None = None,
        splitter: TextSplitter | None = None,
        persist_path: Path | None = None,
    ):
        """
        Initialize knowledge base.

        Args:
            name: Knowledge base name
            store: Vector store (default: in-memory)
            embedding: Embedding provider (default: OpenAI)
            splitter: Text splitter (default: recursive character)
            persist_path: Path for persistence
        """
        self.name = name
        self.persist_path = persist_path

        # Set up store
        if store:
            self.store = store
        elif persist_path:
            self.store = InMemoryVectorStore(persist_path / f"{name}_vectors.json")
        else:
            self.store = InMemoryVectorStore()

        # Set up embedding
        self.embedding = embedding or OpenAIEmbedding()

        # Set up splitter
        self.splitter = splitter or RecursiveCharacterSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        # Set up retriever
        self.retriever = VectorRetriever(
            store=self.store,
            embedding=self.embedding,
        )

        # Document registry
        self._documents: dict[str, Document] = {}

    async def add_document(self, document: Document) -> int:
        """
        Add a document to the knowledge base.

        Args:
            document: Document to add

        Returns:
            Number of chunks created
        """
        # Store document
        self._documents[document.doc_id] = document

        # Split into chunks
        chunks = self.splitter.split(document)
        logger.info(f"Split document {document.doc_id} into {len(chunks)} chunks")

        # Generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = await self.embedding.embed_batch(texts)

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

        # Add to store
        await self.store.add(chunks)

        return len(chunks)

    async def add_documents(
        self,
        documents: list[Document],
        batch_size: int = 10,
    ) -> int:
        """
        Add multiple documents.

        Args:
            documents: Documents to add
            batch_size: Processing batch size

        Returns:
            Total number of chunks created
        """
        total_chunks = 0

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            tasks = [self.add_document(doc) for doc in batch]
            results = await asyncio.gather(*tasks)
            total_chunks += sum(results)

        return total_chunks

    async def add_text(
        self,
        text: str,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """
        Add text directly.

        Args:
            text: Text content
            source: Source identifier
            metadata: Additional metadata

        Returns:
            Number of chunks created
        """
        document = Document(
            content=text,
            source=source,
            metadata=metadata or {},
        )
        return await self.add_document(document)

    async def add_file(self, path: str | Path) -> int:
        """
        Add a file.

        Args:
            path: File path

        Returns:
            Number of chunks created
        """
        document = Document.from_file(path)
        return await self.add_document(document)

    async def add_directory(
        self,
        directory: str | Path,
        recursive: bool = True,
        glob_pattern: str = "*",
    ) -> int:
        """
        Add all documents from a directory.

        Args:
            directory: Directory path
            recursive: Search recursively
            glob_pattern: File pattern

        Returns:
            Number of chunks created
        """
        loader = DirectoryLoader(recursive=recursive, glob_pattern=glob_pattern)
        documents = await loader.load(directory)
        return await self.add_documents(documents)

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Search the knowledge base.

        Args:
            query: Search query
            top_k: Number of results
            filter: Filter criteria

        Returns:
            List of search results
        """
        return await self.retriever.retrieve(query, top_k, filter)

    async def get_context(
        self,
        query: str,
        top_k: int = 5,
        max_tokens: int = 4000,
    ) -> str:
        """
        Get formatted context for LLM.

        Args:
            query: Search query
            top_k: Number of chunks
            max_tokens: Maximum tokens

        Returns:
            Formatted context string
        """
        return await self.retriever.retrieve_with_context(
            query, top_k, max_tokens=max_tokens
        )

    async def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document and its chunks.

        Args:
            doc_id: Document ID

        Returns:
            True if deleted
        """
        if doc_id not in self._documents:
            return False

        # Find chunk IDs for this document
        # Note: This requires iterating through the store
        # For production, consider maintaining a doc_id -> chunk_ids index
        del self._documents[doc_id]

        # For in-memory store, we can filter
        if hasattr(self.store, '_chunks'):
            chunk_ids = [
                cid for cid, chunk in self.store._chunks.items()
                if chunk.doc_id == doc_id
            ]
            await self.store.delete(chunk_ids)

        return True

    async def clear(self) -> None:
        """Clear all documents."""
        self._documents.clear()
        await self.store.clear()

    @property
    def document_count(self) -> int:
        """Get number of documents."""
        return len(self._documents)

    @property
    def chunk_count(self) -> int:
        """Get number of chunks."""
        return self.store.count

    def get_stats(self) -> dict[str, Any]:
        """Get knowledge base statistics."""
        return {
            "name": self.name,
            "documents": self.document_count,
            "chunks": self.chunk_count,
            "embedding_model": getattr(self.embedding, 'model', 'unknown'),
            "embedding_dimension": self.embedding.dimension,
        }


class RAGAgent:
    """
    Agent wrapper that adds RAG capabilities.

    Enhances any agent with knowledge retrieval.
    """

    def __init__(
        self,
        agent: Any,
        knowledge_base: KnowledgeBase,
        context_template: str | None = None,
        top_k: int = 5,
        max_context_tokens: int = 4000,
    ):
        """
        Initialize RAG agent.

        Args:
            agent: Base agent
            knowledge_base: Knowledge base for retrieval
            context_template: Template for injecting context
            top_k: Number of chunks to retrieve
            max_context_tokens: Maximum context tokens
        """
        self.agent = agent
        self.kb = knowledge_base
        self.top_k = top_k
        self.max_context_tokens = max_context_tokens

        self.context_template = context_template or (
            "Use the following context to answer the question:\n\n"
            "Context:\n{context}\n\n"
            "Question: {query}"
        )

    async def __call__(self, msg: Any) -> Any:
        """
        Process message with RAG.

        Args:
            msg: Input message

        Returns:
            Agent response
        """
        # Extract query from message
        query = msg.get_text_content() if hasattr(msg, 'get_text_content') else str(msg)

        # Retrieve context
        context = await self.kb.get_context(
            query,
            top_k=self.top_k,
            max_tokens=self.max_context_tokens,
        )

        if context:
            # Augment message with context
            augmented_content = self.context_template.format(
                context=context,
                query=query,
            )

            # Create new message with augmented content
            from clawscope.message import Msg
            augmented_msg = Msg(
                name=msg.name if hasattr(msg, 'name') else "user",
                content=augmented_content,
                role=msg.role if hasattr(msg, 'role') else "user",
            )

            return await self.agent(augmented_msg)
        else:
            return await self.agent(msg)


__all__ = ["KnowledgeBase", "RAGAgent"]
