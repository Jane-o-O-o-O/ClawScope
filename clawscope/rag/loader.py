"""Document loaders for RAG."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from loguru import logger

from clawscope.rag.document import Document


class DocumentLoader(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    async def load(self, source: str | Path) -> list[Document]:
        """Load documents from source."""
        pass

    @abstractmethod
    def supports(self, source: str | Path) -> bool:
        """Check if loader supports the source type."""
        pass


class TextLoader(DocumentLoader):
    """Loader for plain text files."""

    EXTENSIONS = {".txt", ".text", ".log"}

    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding

    async def load(self, source: str | Path) -> list[Document]:
        """Load text file."""
        path = Path(source)
        try:
            content = path.read_text(encoding=self.encoding)
            return [Document(
                content=content,
                source=str(path),
                metadata={
                    "loader": "text",
                    "filename": path.name,
                    "extension": path.suffix,
                },
            )]
        except Exception as e:
            logger.error(f"Failed to load text file {path}: {e}")
            return []

    def supports(self, source: str | Path) -> bool:
        """Check if source is a text file."""
        return Path(source).suffix.lower() in self.EXTENSIONS


class MarkdownLoader(DocumentLoader):
    """Loader for Markdown files."""

    EXTENSIONS = {".md", ".markdown", ".mdown"}

    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding

    async def load(self, source: str | Path) -> list[Document]:
        """Load markdown file."""
        path = Path(source)
        try:
            content = path.read_text(encoding=self.encoding)

            # Extract metadata from frontmatter if present
            metadata = {"loader": "markdown", "filename": path.name}

            if content.startswith("---"):
                try:
                    import yaml
                    end = content.find("---", 3)
                    if end > 0:
                        frontmatter = yaml.safe_load(content[3:end])
                        if isinstance(frontmatter, dict):
                            metadata.update(frontmatter)
                        content = content[end + 3:].strip()
                except Exception:
                    pass

            return [Document(
                content=content,
                source=str(path),
                metadata=metadata,
            )]
        except Exception as e:
            logger.error(f"Failed to load markdown file {path}: {e}")
            return []

    def supports(self, source: str | Path) -> bool:
        """Check if source is a markdown file."""
        return Path(source).suffix.lower() in self.EXTENSIONS


class PDFLoader(DocumentLoader):
    """Loader for PDF files."""

    EXTENSIONS = {".pdf"}

    async def load(self, source: str | Path) -> list[Document]:
        """Load PDF file."""
        path = Path(source)

        try:
            # Try pypdf first
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(path))

                pages = []
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        pages.append(text)

                content = "\n\n".join(pages)

                return [Document(
                    content=content,
                    source=str(path),
                    metadata={
                        "loader": "pdf",
                        "filename": path.name,
                        "pages": len(reader.pages),
                    },
                )]

            except ImportError:
                logger.warning("pypdf not installed, trying pdfplumber")

                # Fallback to pdfplumber
                import pdfplumber

                pages = []
                with pdfplumber.open(str(path)) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            pages.append(text)

                content = "\n\n".join(pages)

                return [Document(
                    content=content,
                    source=str(path),
                    metadata={
                        "loader": "pdf",
                        "filename": path.name,
                        "pages": len(pages),
                    },
                )]

        except ImportError:
            logger.error("No PDF library available. Install pypdf or pdfplumber.")
            return []
        except Exception as e:
            logger.error(f"Failed to load PDF {path}: {e}")
            return []

    def supports(self, source: str | Path) -> bool:
        """Check if source is a PDF file."""
        return Path(source).suffix.lower() in self.EXTENSIONS


class JSONLoader(DocumentLoader):
    """Loader for JSON files."""

    EXTENSIONS = {".json", ".jsonl"}

    def __init__(
        self,
        content_key: str | None = None,
        metadata_keys: list[str] | None = None,
        encoding: str = "utf-8",
    ):
        """
        Initialize JSON loader.

        Args:
            content_key: Key to extract content from (if None, use entire JSON)
            metadata_keys: Keys to include in metadata
            encoding: File encoding
        """
        self.content_key = content_key
        self.metadata_keys = metadata_keys or []
        self.encoding = encoding

    async def load(self, source: str | Path) -> list[Document]:
        """Load JSON file."""
        path = Path(source)
        documents = []

        try:
            content = path.read_text(encoding=self.encoding)

            if path.suffix == ".jsonl":
                # JSONL: one JSON object per line
                for i, line in enumerate(content.strip().split("\n")):
                    if line.strip():
                        try:
                            data = json.loads(line)
                            doc = self._json_to_document(data, path, i)
                            if doc:
                                documents.append(doc)
                        except json.JSONDecodeError:
                            continue
            else:
                # Regular JSON
                data = json.loads(content)

                if isinstance(data, list):
                    for i, item in enumerate(data):
                        doc = self._json_to_document(item, path, i)
                        if doc:
                            documents.append(doc)
                else:
                    doc = self._json_to_document(data, path, 0)
                    if doc:
                        documents.append(doc)

            return documents

        except Exception as e:
            logger.error(f"Failed to load JSON {path}: {e}")
            return []

    def _json_to_document(
        self, data: dict[str, Any], path: Path, index: int
    ) -> Document | None:
        """Convert JSON object to Document."""
        if not isinstance(data, dict):
            return None

        # Extract content
        if self.content_key:
            content = data.get(self.content_key, "")
        else:
            content = json.dumps(data, ensure_ascii=False, indent=2)

        if not content:
            return None

        # Extract metadata
        metadata = {
            "loader": "json",
            "filename": path.name,
            "index": index,
        }

        for key in self.metadata_keys:
            if key in data:
                metadata[key] = data[key]

        return Document(content=str(content), source=str(path), metadata=metadata)

    def supports(self, source: str | Path) -> bool:
        """Check if source is a JSON file."""
        return Path(source).suffix.lower() in self.EXTENSIONS


class DirectoryLoader:
    """Load all documents from a directory."""

    def __init__(
        self,
        loaders: list[DocumentLoader] | None = None,
        recursive: bool = True,
        glob_pattern: str = "*",
    ):
        """
        Initialize directory loader.

        Args:
            loaders: List of document loaders to use
            recursive: Whether to search recursively
            glob_pattern: Glob pattern for file matching
        """
        self.loaders = loaders or [
            TextLoader(),
            MarkdownLoader(),
            PDFLoader(),
            JSONLoader(),
        ]
        self.recursive = recursive
        self.glob_pattern = glob_pattern

    async def load(self, directory: str | Path) -> list[Document]:
        """Load all documents from directory."""
        directory = Path(directory)
        documents = []

        if not directory.is_dir():
            logger.error(f"Not a directory: {directory}")
            return []

        # Get all files
        if self.recursive:
            files = list(directory.rglob(self.glob_pattern))
        else:
            files = list(directory.glob(self.glob_pattern))

        # Load each file
        for file_path in files:
            if not file_path.is_file():
                continue

            # Find appropriate loader
            for loader in self.loaders:
                if loader.supports(file_path):
                    docs = await loader.load(file_path)
                    documents.extend(docs)
                    break

        logger.info(f"Loaded {len(documents)} documents from {directory}")
        return documents


__all__ = [
    "DocumentLoader",
    "TextLoader",
    "MarkdownLoader",
    "PDFLoader",
    "JSONLoader",
    "DirectoryLoader",
]
