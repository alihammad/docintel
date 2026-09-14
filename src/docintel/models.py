"""Core data model for parsed documents."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class DocFormat(str, Enum):
    MARKDOWN = "markdown"
    PDF = "pdf"
    DOCX = "docx"

    @classmethod
    def from_path(cls, path: Path) -> "DocFormat":
        suffix = path.suffix.lower()
        mapping = {
            ".md": cls.MARKDOWN,
            ".markdown": cls.MARKDOWN,
            ".pdf": cls.PDF,
            ".docx": cls.DOCX,
        }
        if suffix not in mapping:
            raise ValueError(
                f"Unsupported file type: {path.name!r}. "
                f"Supported: {', '.join(sorted(mapping))}"
            )
        return mapping[suffix]


@dataclass
class Heading:
    level: int
    text: str


@dataclass
class Link:
    text: str
    target: str
    is_external: bool


@dataclass
class Document:
    """A parsed document with its extracted content and structure."""

    path: Path
    format: DocFormat
    text: str
    headings: list[Heading] = field(default_factory=list)
    links: list[Link] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def words(self) -> list[str]:
        return self.text.split()

    @property
    def word_count(self) -> int:
        return len(self.words)

    @property
    def line_count(self) -> int:
        return self.text.count("\n") + 1 if self.text else 0
