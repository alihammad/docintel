"""PDF parser built on pypdf."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from docintel.models import DocFormat, Document, Heading


def parse_pdf(path: Path) -> Document:
    reader = PdfReader(str(path))

    chunks: list[str] = []
    for page in reader.pages:
        chunks.append(page.extract_text() or "")
    text = "\n".join(chunks)

    # Headings from PDF outline/bookmarks, if present.
    headings: list[Heading] = []

    def _walk_outline(items: list, level: int = 1) -> None:
        for item in items:
            if isinstance(item, list):
                _walk_outline(item, level + 1)
            else:
                title = getattr(item, "title", str(item))
                headings.append(Heading(level=level, text=title))

    try:
        if reader.outline:
            _walk_outline(reader.outline)
    except Exception:  # noqa: BLE001 - malformed outlines should not break parsing
        pass

    metadata: dict[str, str] = {}
    if reader.metadata:
        for key in ("title", "author", "subject", "creator", "producer"):
            value = reader.metadata.get(f"/{key.capitalize()}")
            if value:
                metadata[key] = str(value)
    metadata["pages"] = str(len(reader.pages))

    return Document(
        path=path,
        format=DocFormat.PDF,
        text=text,
        headings=headings,
        links=[],
        metadata=metadata,
    )
