"""Analysis of parsed documents: statistics, structure, and link checks."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from docintel.models import Document

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_STOPWORDS = frozenset(
    """
    a an and are as at be but by for if in into is it no not of on or such
    that the their then there these they this to was will with
    """.split()
)


@dataclass
class BrokenLink:
    text: str
    target: str
    reason: str


@dataclass
class Analysis:
    word_count: int
    char_count: int
    line_count: int
    sentence_count: int
    avg_words_per_sentence: float
    heading_count: int
    max_heading_depth: int
    top_words: list[tuple[str, int]]
    external_link_count: int
    internal_link_count: int
    broken_links: list[BrokenLink] = field(default_factory=list)


def _count_sentences(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return len(_SENTENCE_SPLIT_RE.split(stripped))


def _top_words(text: str, limit: int = 10) -> list[tuple[str, int]]:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    counts = Counter(w for w in words if w not in _STOPWORDS and len(w) > 1)
    return counts.most_common(limit)


def _check_local_link(doc: Document, target: str) -> str | None:
    """Return a reason string if a local link target is broken, else None."""
    # Strip anchors and query strings.
    path_part = target.split("#", 1)[0].split("?", 1)[0]
    if not path_part:
        return None  # pure in-page anchor; anchor validation not performed
    resolved = (doc.path.parent / path_part).resolve()
    if not resolved.exists():
        return f"file not found: {path_part}"
    return None


def analyze(doc: Document, check_links: bool = True) -> Analysis:
    sentences = _count_sentences(doc.text)
    word_count = doc.word_count

    broken: list[BrokenLink] = []
    if check_links:
        for link in doc.links:
            if link.is_external:
                continue  # no network checks in v1
            reason = _check_local_link(doc, link.target)
            if reason:
                broken.append(BrokenLink(text=link.text, target=link.target, reason=reason))

    external = sum(1 for link in doc.links if link.is_external)
    internal = len(doc.links) - external

    return Analysis(
        word_count=word_count,
        char_count=len(doc.text),
        line_count=doc.line_count,
        sentence_count=sentences,
        avg_words_per_sentence=round(word_count / sentences, 1) if sentences else 0.0,
        heading_count=len(doc.headings),
        max_heading_depth=max((h.level for h in doc.headings), default=0),
        top_words=_top_words(doc.text),
        external_link_count=external,
        internal_link_count=internal,
        broken_links=broken,
    )
