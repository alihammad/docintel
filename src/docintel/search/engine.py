"""Semantic search engine: index documents and query the vector store.

Embeddings come from OpenAI when ``search.embedding_model`` is configured;
otherwise a deterministic local fallback is used and every result is flagged
(consistent with the classify/extract fallback policy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from docintel.models import DocFormat
from docintel.parsers import parse_document
from docintel.search.chunking import Chunk, chunk_document
from docintel.search.embeddings import LOCAL_METHOD, embed_texts
from docintel.search.store import ChunkStore, StoredChunk, StoreError, file_content_hash

Method = str  # "openai" | "local-fallback"


@dataclass
class IndexedDoc:
    path: str
    chunks: int
    status: str  # "indexed" | "unchanged" | "reindexed"


@dataclass
class IndexResult:
    docs: list[IndexedDoc] = field(default_factory=list)
    method: Method = LOCAL_METHOD
    model: str | None = None
    total_chunks: int = 0
    note: str | None = None
    error: str | None = None

    @property
    def used_fallback(self) -> bool:
        return self.method == LOCAL_METHOD


@dataclass
class SearchHit:
    score: float
    doc_path: str
    chunk_index: int
    heading_path: str
    snippet: str
    full_text: str = ""  # untruncated chunk text (used for RAG grounding)


@dataclass
class SearchResult:
    query: str
    hits: list[SearchHit] = field(default_factory=list)
    method: Method = LOCAL_METHOD
    model: str | None = None
    note: str | None = None
    error: str | None = None

    @property
    def used_fallback(self) -> bool:
        return self.method == LOCAL_METHOD


def _open_store(config) -> ChunkStore:
    index_path = Path(config.search.index_path)
    return ChunkStore(index_path)


def _embed_config(config):
    """Common kwargs for embed_texts pulled from config."""
    return dict(
        embedding_model=config.search.embedding_model,
        base_url=config.llm.base_url,
        api_key_env=config.llm.api_key_env,
    )


def _parse(path: Path):
    DocFormat.from_path(path)  # raises ValueError for unsupported types
    return parse_document(path)


def index_documents(paths: list[Path], config, reindex: bool = False) -> IndexResult:
    """Parse, chunk, embed, and store documents. Unchanged files are skipped."""
    try:
        store = _open_store(config)
    except Exception as exc:  # noqa: BLE001 - surface store failures to the CLI
        return IndexResult(error=f"failed to open index: {exc}")

    # Guard: an index built with a different embedder is not comparable.
    prev_method, prev_model = store.embedding_info()
    emb_kwargs = _embed_config(config)
    current_model = emb_kwargs["embedding_model"]
    if not reindex and prev_method and prev_model != (current_model or ""):
        return IndexResult(
            error=(
                f"index was built with embedding model {prev_model!r}, "
                f"but {current_model!r} is configured now. Run with --reindex to rebuild."
            )
        )

    result = IndexResult()
    pending: list[tuple[Path, list[Chunk], str]] = []  # (path, chunks, file_hash)

    for path in paths:
        resolved = path.resolve()
        if not resolved.exists():
            result.docs.append(IndexedDoc(str(resolved), 0, "missing"))
            continue
        try:
            doc = _parse(resolved)
        except Exception as exc:  # noqa: BLE001
            result.docs.append(IndexedDoc(str(resolved), 0, f"error: {exc}"))
            continue

        content_hash = file_content_hash(resolved)
        if not reindex and store.indexed_file_hash(str(resolved)) == content_hash:
            result.docs.append(IndexedDoc(str(resolved), 0, "unchanged"))
            continue

        chunks = chunk_document(
            doc,
            chunk_size=config.search.chunk_size,
            overlap_frac=config.search.overlap,
        )
        if not chunks:
            result.docs.append(IndexedDoc(str(resolved), 0, "empty"))
            continue
        pending.append((resolved, chunks, content_hash))

    if not pending:
        result.method = prev_method or LOCAL_METHOD
        result.model = prev_model or None
        result.note = "No new or changed documents to index."
        return result

    # Embed all chunks of all pending docs in one pass (batched by provider).
    all_chunks = [c for _, chunks, _ in pending for c in chunks]
    outcome = embed_texts([c.text for c in all_chunks], **emb_kwargs)
    result.method = outcome.method
    result.model = outcome.model
    if outcome.error:
        result.error = outcome.error

    offset = 0
    for resolved, chunks, content_hash in pending:
        vectors = outcome.vectors[offset: offset + len(chunks)]
        offset += len(chunks)
        existed = store.indexed_file_hash(str(resolved)) is not None
        if existed:
            store.delete_document(str(resolved))
        store.upsert_chunks(
            [
                StoredChunk(c.chunk_id, c.doc_path, c.chunk_index, c.heading_path, c.text)
                for c in chunks
            ],
            vectors,
            file_hash=content_hash,
        )
        result.docs.append(
            IndexedDoc(str(resolved), len(chunks), "reindexed" if existed else "indexed")
        )
        result.total_chunks += len(chunks)

    store.set_embedding_info(outcome.method, outcome.model)
    if outcome.method == LOCAL_METHOD:
        result.note = (
            "Local fallback embeddings (hashed bag-of-words): lexical matching only, "
            "not true semantic search. Configure search.embedding_model for real embeddings."
        )
    return result


def search(query: str, config, top_k: int | None = None) -> SearchResult:
    """Embed the query and return the top-k most similar chunks."""
    top_k = top_k or config.search.top_k
    try:
        store = _open_store(config)
    except Exception as exc:  # noqa: BLE001
        return SearchResult(query=query, error=f"failed to open index: {exc}")

    if store.chunk_count() == 0:
        return SearchResult(
            query=query,
            note="Index is empty. Run `docintel index <paths...>` first.",
        )

    prev_method, prev_model = store.embedding_info()
    emb_kwargs = _embed_config(config)
    current_model = emb_kwargs["embedding_model"]
    if prev_method and prev_model != (current_model or ""):
        raise StoreError(
            f"index was built with embedding model {prev_model!r}, "
            f"but {current_model!r} is configured now. Rebuild with `docintel index --reindex`."
        )

    outcome = embed_texts([query], **emb_kwargs)
    hits = store.query(outcome.vectors[0], top_k)

    result = SearchResult(
        query=query,
        hits=[
            SearchHit(
                score=round(score, 4),
                doc_path=chunk.doc_path,
                chunk_index=chunk.chunk_index,
                heading_path=chunk.heading_path,
                snippet=_snippet(chunk.text),
                full_text=chunk.text,
            )
            for score, chunk in hits
        ],
        method=outcome.method,
        model=outcome.model,
        error=outcome.error,
    )
    if outcome.method == LOCAL_METHOD:
        result.note = (
            "Local fallback embeddings: scores reflect lexical overlap only, "
            "not true semantic similarity."
        )
    return result


def _snippet(text: str, limit: int = 200) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"
