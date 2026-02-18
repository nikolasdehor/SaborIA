"""
Ingestion pipeline: receives menu (PDF or plain text), chunks, embeds and
stores in ChromaDB.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from api.settings import settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "menus"


def _get_vector_store() -> Chroma:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=settings.openai_api_key,
    )
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )


def _load_documents(source: str | Path) -> list[Document]:
    source = Path(source)
    if source.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(source))
    else:
        loader = TextLoader(str(source), encoding="utf-8")
    return loader.load()


def _assign_menu_id(docs: list[Document], menu_name: str) -> list[Document]:
    menu_id = hashlib.md5(menu_name.encode()).hexdigest()[:8]
    for doc in docs:
        doc.metadata.update({"menu_id": menu_id, "menu_name": menu_name})
    return docs


def ingest_file(file_path: str | Path, menu_name: str) -> dict:
    """Load, chunk, embed and persist a menu document."""
    docs = _load_documents(file_path)
    docs = _assign_menu_id(docs, menu_name)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ".", ",", " "],
    )
    chunks = splitter.split_documents(docs)

    # Deduplicate by content hash
    seen: set[str] = set()
    unique_chunks: list[Document] = []
    for chunk in chunks:
        h = hashlib.md5(chunk.page_content.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            chunk.metadata["content_hash"] = h
            unique_chunks.append(chunk)

    logger.info(
        "Ingesting '%s': %d raw chunks → %d unique chunks",
        menu_name,
        len(chunks),
        len(unique_chunks),
    )

    vs = _get_vector_store()
    vs.add_documents(unique_chunks)

    return {
        "menu_name": menu_name,
        "menu_id": unique_chunks[0].metadata["menu_id"],
        "total_chunks": len(unique_chunks),
        "pages": len(docs),
    }


def ingest_text(text: str, menu_name: str) -> dict:
    """Ingest raw text directly (useful for tests and synthetic data)."""
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(text)
        tmp_path = tmp.name
    try:
        result = ingest_file(tmp_path, menu_name)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return result


def get_retriever(menu_name: str | None = None):
    vs = _get_vector_store()
    search_kwargs = {"k": settings.retriever_k}
    if menu_name:
        menu_id = hashlib.md5(menu_name.encode()).hexdigest()[:8]
        search_kwargs["filter"] = {"menu_id": menu_id}
    return vs.as_retriever(search_kwargs=search_kwargs)
