"""Text splitters for RAG."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Callable

from clawscope.rag.document import Document, DocumentChunk


class TextSplitter(ABC):
    """Abstract base class for text splitters."""

    @abstractmethod
    def split(self, document: Document) -> list[DocumentChunk]:
        """Split document into chunks."""
        pass

    def split_text(self, text: str) -> list[str]:
        """Split text into strings."""
        pass


class RecursiveCharacterSplitter(TextSplitter):
    """
    Recursively split text by different separators.

    Tries to split on larger semantic units first (paragraphs),
    then falls back to smaller units (sentences, words).
    """

    DEFAULT_SEPARATORS = [
        "\n\n",  # Paragraphs
        "\n",    # Lines
        ". ",    # Sentences
        ", ",    # Clauses
        " ",     # Words
        "",      # Characters
    ]

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
        length_function: Callable[[str], int] | None = None,
    ):
        """
        Initialize splitter.

        Args:
            chunk_size: Maximum chunk size
            chunk_overlap: Overlap between chunks
            separators: List of separators to try
            length_function: Function to measure text length
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS
        self.length_function = length_function or len

    def split(self, document: Document) -> list[DocumentChunk]:
        """Split document into chunks."""
        texts = self._split_text(document.content, self.separators)

        chunks = []
        for i, text in enumerate(texts):
            chunk = DocumentChunk(
                content=text,
                doc_id=document.doc_id,
                chunk_index=i,
                metadata={
                    **document.metadata,
                    "source": document.source,
                },
            )
            chunks.append(chunk)

        return chunks

    def split_text(self, text: str) -> list[str]:
        """Split text into strings."""
        return self._split_text(text, self.separators)

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text."""
        final_chunks = []

        # Get appropriate separator
        separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        # Split by separator
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        # Merge splits into chunks
        good_splits = []
        current_chunk = []
        current_length = 0

        for split in splits:
            split_length = self.length_function(split)

            if current_length + split_length > self.chunk_size:
                if current_chunk:
                    # Create chunk from accumulated splits
                    chunk_text = separator.join(current_chunk)
                    if chunk_text.strip():
                        final_chunks.append(chunk_text)

                    # Handle overlap
                    while current_length > self.chunk_overlap and len(current_chunk) > 1:
                        removed = current_chunk.pop(0)
                        current_length -= self.length_function(removed) + len(separator)

                if split_length > self.chunk_size:
                    # Split is too large, need to recursively split
                    if new_separators:
                        sub_chunks = self._split_text(split, new_separators)
                        final_chunks.extend(sub_chunks)
                    else:
                        # Can't split further, add as is
                        final_chunks.append(split)
                else:
                    current_chunk = [split]
                    current_length = split_length
            else:
                current_chunk.append(split)
                current_length += split_length + len(separator)

        # Add remaining chunk
        if current_chunk:
            chunk_text = separator.join(current_chunk)
            if chunk_text.strip():
                final_chunks.append(chunk_text)

        return final_chunks


class TokenSplitter(TextSplitter):
    """Split text by token count."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        model: str = "gpt-4",
    ):
        """
        Initialize token splitter.

        Args:
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Token overlap between chunks
            model: Model name for tokenizer
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = model
        self._tokenizer = None

    def _get_tokenizer(self):
        """Get or create tokenizer."""
        if self._tokenizer is None:
            try:
                import tiktoken
                self._tokenizer = tiktoken.encoding_for_model(self.model)
            except Exception:
                import tiktoken
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
        return self._tokenizer

    def _token_count(self, text: str) -> int:
        """Count tokens in text."""
        tokenizer = self._get_tokenizer()
        return len(tokenizer.encode(text))

    def split(self, document: Document) -> list[DocumentChunk]:
        """Split document by tokens."""
        texts = self.split_text(document.content)

        chunks = []
        for i, text in enumerate(texts):
            chunk = DocumentChunk(
                content=text,
                doc_id=document.doc_id,
                chunk_index=i,
                metadata={
                    **document.metadata,
                    "source": document.source,
                    "token_count": self._token_count(text),
                },
            )
            chunks.append(chunk)

        return chunks

    def split_text(self, text: str) -> list[str]:
        """Split text by tokens."""
        tokenizer = self._get_tokenizer()
        tokens = tokenizer.encode(text)

        chunks = []
        start = 0

        while start < len(tokens):
            end = start + self.chunk_size

            # Get chunk tokens
            chunk_tokens = tokens[start:end]
            chunk_text = tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)

            # Move start with overlap
            start = end - self.chunk_overlap

        return chunks


class SentenceSplitter(TextSplitter):
    """Split text by sentences."""

    SENTENCE_ENDINGS = re.compile(r'(?<=[.!?])\s+')

    def __init__(
        self,
        chunk_size: int = 5,
        chunk_overlap: int = 1,
    ):
        """
        Initialize sentence splitter.

        Args:
            chunk_size: Sentences per chunk
            chunk_overlap: Sentence overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, document: Document) -> list[DocumentChunk]:
        """Split document by sentences."""
        texts = self.split_text(document.content)

        chunks = []
        for i, text in enumerate(texts):
            chunk = DocumentChunk(
                content=text,
                doc_id=document.doc_id,
                chunk_index=i,
                metadata={
                    **document.metadata,
                    "source": document.source,
                },
            )
            chunks.append(chunk)

        return chunks

    def split_text(self, text: str) -> list[str]:
        """Split text by sentences."""
        sentences = self.SENTENCE_ENDINGS.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        start = 0

        while start < len(sentences):
            end = min(start + self.chunk_size, len(sentences))
            chunk_sentences = sentences[start:end]
            chunk_text = " ".join(chunk_sentences)
            chunks.append(chunk_text)

            # Move start with overlap
            start = end - self.chunk_overlap
            if start <= 0:
                start = end

        return chunks


__all__ = [
    "TextSplitter",
    "RecursiveCharacterSplitter",
    "TokenSplitter",
    "SentenceSplitter",
]
