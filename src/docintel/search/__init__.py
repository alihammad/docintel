"""Semantic search: chunking, embeddings, vector store (ChromaDB), retrieval."""

from docintel.search.engine import (
    IndexResult,
    SearchResult,
    index_documents,
    search,
)

__all__ = ["IndexResult", "SearchResult", "index_documents", "search"]
