"""Format-specific document parsers."""

from __future__ import annotations

from pathlib import Path

from docintel.models import DocFormat, Document

from .docx_parser import parse_docx
from .markdown_parser import parse_markdown
from .pdf_parser import parse_pdf

_PARSERS = {
    DocFormat.MARKDOWN: parse_markdown,
    DocFormat.PDF: parse_pdf,
    DocFormat.DOCX: parse_docx,
}


def parse_document(path: Path) -> Document:
    """Parse a document file into a :class:`Document`, dispatching on file type."""
    doc_format = DocFormat.from_path(path)
    return _PARSERS[doc_format](path)


__all__ = ["parse_document"]
