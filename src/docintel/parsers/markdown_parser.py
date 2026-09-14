"""Markdown parser using stdlib regex (no external markdown dependency)."""

from __future__ import annotations

import re
from pathlib import Path

from docintel.models import DocFormat, Document, Heading, Link

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", re.MULTILINE)
_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"^(```|~~~)", re.MULTILINE)


def _strip_code_blocks(text: str) -> str:
    """Blank out fenced code blocks so their contents aren't parsed."""
    lines = text.splitlines(keepends=True)
    in_fence = False
    fence_marker = ""
    out: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        if not in_fence and (stripped.startswith("```") or stripped.startswith("~~~")):
            in_fence = True
            fence_marker = stripped[:3]
            out.append("\n")
            continue
        if in_fence:
            if stripped.startswith(fence_marker):
                in_fence = False
            out.append("\n")
            continue
        out.append(line)
    return "".join(out)


def _parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Extract simple YAML front matter (key: value pairs) and return the rest."""
    match = _FRONT_MATTER_RE.match(text)
    if not match:
        return {}, text
    metadata: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line and not line.lstrip().startswith("#"):
            key, _, value = line.partition(":")
            metadata[key.strip()] = value.strip().strip("\"'")
    return metadata, text[match.end():]


def _is_external(target: str) -> bool:
    return bool(re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", target)) or target.startswith(
        ("mailto:", "www.")
    )


def parse_markdown(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    metadata, body = _parse_front_matter(raw)
    scannable = _strip_code_blocks(body)

    headings = [
        Heading(level=len(m.group(1)), text=m.group(2).strip())
        for m in _HEADING_RE.finditer(scannable)
    ]
    links = [
        Link(text=m.group(1), target=m.group(2), is_external=_is_external(m.group(2)))
        for m in _LINK_RE.finditer(scannable)
    ]

    return Document(
        path=path,
        format=DocFormat.MARKDOWN,
        text=body,
        headings=headings,
        links=links,
        metadata=metadata,
    )
