"""Document chunking for indexing.

Markdown is chunked heading-aware (sections are kept together where possible);
other formats use paragraph windows. Chunks overlap slightly so sentences
split across a boundary are still retrievable.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from docintel.models import DocFormat, Document

DEFAULT_CHUNK_SIZE = 1200  # chars
DEFAULT_OVERLAP = 0.15  # fraction of chunk_size


@dataclass
class Chunk:
    doc_path: str
    chunk_index: int
    text: str
    heading_path: str  # e.g. "Chapter 1 > Setup"; "" when no heading applies

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def chunk_id(self) -> str:
        """Stable id for the vector store: file hash + chunk index."""
        file_key = hashlib.sha256(self.doc_path.encode("utf-8")).hexdigest()[:16]
        return f"{file_key}:{self.chunk_index}"


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", re.MULTILINE)


def _window(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping windows at paragraph/sentence boundaries."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    paragraphs = re.split(r"\n\s*\n", text)
    windows: list[str] = []
    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # Paragraphs longer than the cap are hard-split on sentence boundaries.
        while len(para) > chunk_size:
            cut = para.rfind(". ", 0, chunk_size)
            cut = cut + 2 if cut > chunk_size // 2 else chunk_size
            if current:
                windows.append(current)
                current = ""
            windows.append(para[:cut].strip())
            para = para[cut:].strip()
        if not para:
            continue
        if current and len(current) + len(para) + 2 > chunk_size:
            windows.append(current)
            # Carry the tail of the previous window as overlap context.
            current = (current[-overlap:] + "\n\n" + para) if overlap else para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current.strip():
        windows.append(current.strip())
    return windows


def _markdown_sections(doc: Document) -> list[tuple[str, str]]:
    """Split markdown into (heading_path, section_text) pairs."""
    text = doc.text
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [("", text)]

    sections: list[tuple[str, str]] = []
    # Preamble before the first heading.
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(("", preamble))

    stack: list[tuple[int, str]] = []  # (level, title) for heading_path
    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        heading_path = " > ".join(t for _, t in stack)
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end(): end].strip()
        if body:
            sections.append((heading_path, body))
        else:
            # Heading with no body yet: keep the path so a following deeper
            # heading inherits it; nothing to index for this heading itself.
            continue
    return sections


def chunk_document(
    doc: Document,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap_frac: float = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Chunk a parsed document into overlapping, heading-aware pieces."""
    chunk_size = max(200, chunk_size)
    overlap = int(chunk_size * min(max(overlap_frac, 0.0), 0.5))
    doc_path = str(doc.path)

    if doc.format == DocFormat.MARKDOWN:
        sections = _markdown_sections(doc)
    else:
        sections = [("", doc.text)]

    chunks: list[Chunk] = []
    for heading_path, section in sections:
        for window in _window(section, chunk_size, overlap):
            chunks.append(
                Chunk(
                    doc_path=doc_path,
                    chunk_index=len(chunks),
                    text=window,
                    heading_path=heading_path,
                )
            )
    return chunks
