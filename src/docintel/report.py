"""Render analysis results as text or JSON reports."""

from __future__ import annotations

import json
from dataclasses import asdict

from docintel.analysis import Analysis
from docintel.models import Document


def render_text(doc: Document, analysis: Analysis) -> str:
    lines: list[str] = []
    lines.append(f"Document: {doc.path}")
    lines.append(f"Format:   {doc.format.value}")
    if doc.metadata:
        lines.append("")
        lines.append("Metadata:")
        for key, value in doc.metadata.items():
            lines.append(f"  {key}: {value}")

    lines.append("")
    lines.append("Statistics:")
    lines.append(f"  words:                  {analysis.word_count}")
    lines.append(f"  characters:             {analysis.char_count}")
    lines.append(f"  lines:                  {analysis.line_count}")
    lines.append(f"  sentences:              {analysis.sentence_count}")
    lines.append(f"  avg words/sentence:     {analysis.avg_words_per_sentence}")

    lines.append("")
    lines.append("Structure:")
    lines.append(f"  headings:               {analysis.heading_count}")
    lines.append(f"  max heading depth:      {analysis.max_heading_depth}")
    if doc.headings:
        for heading in doc.headings[:20]:
            indent = "  " * heading.level
            lines.append(f"    {indent}{'#' * heading.level} {heading.text}")
        if len(doc.headings) > 20:
            lines.append(f"    ... and {len(doc.headings) - 20} more")

    lines.append("")
    lines.append("Links:")
    lines.append(f"  external:               {analysis.external_link_count}")
    lines.append(f"  internal:               {analysis.internal_link_count}")
    if analysis.broken_links:
        lines.append(f"  BROKEN:                 {len(analysis.broken_links)}")
        for broken in analysis.broken_links:
            label = f" [{broken.text}]" if broken.text else ""
            lines.append(f"    - {broken.target}{label}: {broken.reason}")

    if analysis.top_words:
        lines.append("")
        lines.append("Top words:")
        for word, count in analysis.top_words:
            lines.append(f"  {word:<20} {count}")

    return "\n".join(lines)


def render_json(doc: Document, analysis: Analysis) -> str:
    payload = {
        "path": str(doc.path),
        "format": doc.format.value,
        "metadata": doc.metadata,
        "headings": [asdict(h) for h in doc.headings],
        "links": [asdict(link) for link in doc.links],
        "analysis": asdict(analysis),
    }
    return json.dumps(payload, indent=2)
