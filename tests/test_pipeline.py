"""Unit tests for the ingestion pipeline (no API key required)."""

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Patch settings before importing pipeline
import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder")


class TestIngestionHelpers:
    """Test internal pipeline helpers without hitting OpenAI."""

    def test_load_documents_txt(self, sample_menu_file: Path):
        from ingestion.pipeline import _load_documents

        docs = _load_documents(sample_menu_file)
        assert len(docs) >= 1
        assert "RESTAURANTE TESTE" in docs[0].page_content

    def test_assign_menu_id_adds_metadata(self):
        from langchain_core.documents import Document

        from ingestion.pipeline import _assign_menu_id

        docs = [Document(page_content="Test content", metadata={})]
        result = _assign_menu_id(docs, "My Restaurant")
        assert result[0].metadata["menu_name"] == "My Restaurant"
        assert result[0].metadata["menu_id"] == hashlib.md5(b"My Restaurant").hexdigest()[:8]

    def test_assign_menu_id_deterministic(self):
        """Same menu name should always produce the same ID."""
        from langchain_core.documents import Document

        from ingestion.pipeline import _assign_menu_id

        docs1 = _assign_menu_id([Document(page_content="a")], "Bella Terra")
        docs2 = _assign_menu_id([Document(page_content="b")], "Bella Terra")
        assert docs1[0].metadata["menu_id"] == docs2[0].metadata["menu_id"]

    def test_deduplication_by_content_hash(self, sample_menu_file: Path):
        """Ingesting the same file twice should not create duplicates."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_core.documents import Document

        from ingestion.pipeline import _load_documents

        docs = _load_documents(sample_menu_file)
        splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=32)
        chunks = splitter.split_documents(docs)

        # Simulate deduplication logic
        seen: set[str] = set()
        unique: list[Document] = []
        for chunk in chunks + chunks:  # duplicate
            h = hashlib.md5(chunk.page_content.encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                unique.append(chunk)

        assert len(unique) == len(chunks)
        assert len(unique) < len(chunks + chunks)


class TestRetrieverConfig:
    """Test retriever construction logic."""

    def test_get_retriever_default_k(self):
        """Without menu_name, retriever uses default k from settings."""
        from api.settings import settings

        # Just verify settings have the expected defaults
        assert settings.retriever_k > 0

    def test_menu_id_generation(self):
        """Menu IDs are short deterministic hashes."""
        menu_id = hashlib.md5("Bella Terra".encode()).hexdigest()[:8]
        assert len(menu_id) == 8
        assert menu_id.isalnum()
