"""Embedding providers: OpenAI embeddings with a deterministic local fallback.

The fallback is a hashed bag-of-words embedder: each token is hashed into a
fixed-dimension vector space with a signed weight. It captures lexical overlap
only (no true semantics), so results produced with it are always flagged.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

LOCAL_DIM = 512
LOCAL_METHOD = "local-fallback"
OPENAI_METHOD = "openai"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass
class EmbeddingOutcome:
    vectors: list[list[float]]
    method: str  # "openai" | "local-fallback"
    model: str | None = None  # embedding model name (None for local fallback)
    error: str | None = None  # why the OpenAI path was not used


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def local_embed(texts: list[str], dim: int = LOCAL_DIM) -> list[list[float]]:
    """Deterministic hashed bag-of-words embeddings (L2-normalized)."""
    vectors: list[list[float]] = []
    for text in texts:
        vec = [0.0] * dim
        tokens = _tokenize(text)
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(x * x for x in vec))
        if norm:
            vec = [x / norm for x in vec]
        vectors.append(vec)
    return vectors


def openai_embed(texts: list[str], model: str, base_url: str | None, api_key_env: str) -> list[list[float]]:
    """Embed via the OpenAI embeddings API. Raises on any failure."""
    import os

    from openai import OpenAI  # deferred import, consistent with the other engines

    if not os.environ.get(api_key_env):
        raise RuntimeError(f"API key env var {api_key_env!r} is not set")
    client = OpenAI(base_url=base_url) if base_url else OpenAI()
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def embed_texts(
    texts: list[str],
    embedding_model: str | None,
    base_url: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
) -> EmbeddingOutcome:
    """Embed texts with OpenAI when configured; otherwise use the local fallback.

    ``embedding_model`` must be explicitly configured (no hardcoded default),
    mirroring the project's LLM model rule.
    """
    if not texts:
        return EmbeddingOutcome(vectors=[], method=LOCAL_METHOD)

    if not embedding_model:
        return EmbeddingOutcome(
            vectors=local_embed(texts),
            method=LOCAL_METHOD,
            error="no embedding model configured (search.embedding_model / DOCINTEL_EMBEDDING_MODEL)",
        )

    try:
        vectors = openai_embed(texts, embedding_model, base_url, api_key_env)
    except Exception as exc:  # noqa: BLE001 - any failure triggers the fallback
        return EmbeddingOutcome(
            vectors=local_embed(texts),
            method=LOCAL_METHOD,
            error=f"embedding API call failed: {exc}",
        )
    return EmbeddingOutcome(vectors=vectors, method=OPENAI_METHOD, model=embedding_model)
