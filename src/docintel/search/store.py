"""ChromaDB-backed vector store for document chunks.

The collection stores chunk text plus metadata (source path, heading path,
chunk index, per-file content hash) and records which embedding method/model
built it, so queries embed consistently and model changes force a reindex.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

COLLECTION_NAME = "docintel_chunks"
DEFAULT_INDEX_DIR = Path(".docintel") / "chroma"


class StoreError(RuntimeError):
    """Raised when the index cannot be used (e.g. embedding model changed)."""


@dataclass
class StoredChunk:
    chunk_id: str
    doc_path: str
    chunk_index: int
    heading_path: str
    text: str


def file_content_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ChunkStore:
    """Thin wrapper around a persistent ChromaDB collection."""

    def __init__(self, index_path: Path):
        import chromadb  # deferred: keeps CLI startup fast

        index_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(index_path))
        # embedding_function=None: docintel always supplies its own vectors
        # (OpenAI or local fallback); never let Chroma download a default model.
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

    # -- collection-level embedding bookkeeping ----------------------------

    def embedding_info(self) -> tuple[str | None, str | None]:
        """Return (method, model) recorded when the index was built."""
        meta = self._collection.metadata or {}
        return meta.get("embed_method"), meta.get("embed_model")

    def set_embedding_info(self, method: str, model: str | None) -> None:
        self._collection.modify(metadata={"embed_method": method, "embed_model": model or ""})

    # -- document bookkeeping ----------------------------------------------

    def indexed_file_hash(self, doc_path: str) -> str | None:
        """Content hash of an already-indexed file, or None if not indexed."""
        result = self._collection.get(where={"doc_path": doc_path}, include=["metadatas"], limit=1)
        metas = result.get("metadatas") or []
        return metas[0].get("file_hash") if metas else None

    def delete_document(self, doc_path: str) -> None:
        self._collection.delete(where={"doc_path": doc_path})

    def document_count(self) -> int:
        paths = self._collection.get(include=["metadatas"]).get("metadatas") or []
        return len({m.get("doc_path") for m in paths if m.get("doc_path")})

    def chunk_count(self) -> int:
        return self._collection.count()

    # -- write / read --------------------------------------------------------

    def upsert_chunks(
        self,
        chunks: list[StoredChunk],
        vectors: list[list[float]],
        file_hash: str,
    ) -> None:
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "doc_path": c.doc_path,
                    "chunk_index": c.chunk_index,
                    "heading_path": c.heading_path,
                    "file_hash": file_hash,
                }
                for c in chunks
            ],
        )

    def query(self, vector: list[float], top_k: int) -> list[tuple[float, StoredChunk]]:
        """Cosine-similarity search; returns (score, chunk) ranked best-first."""
        if self._collection.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[vector],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        hits: list[tuple[float, StoredChunk]] = []
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        for i, _id in enumerate(ids):
            meta = metas[i] or {}
            # Chroma cosine distance = 1 - similarity.
            score = 1.0 - float(dists[i])
            hits.append(
                (
                    score,
                    StoredChunk(
                        chunk_id=_id,
                        doc_path=str(meta.get("doc_path", "")),
                        chunk_index=int(meta.get("chunk_index", 0)),
                        heading_path=str(meta.get("heading_path", "")),
                        text=docs[i] or "",
                    ),
                )
            )
        return hits
