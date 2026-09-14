"""Render analysis results as text or JSON reports."""

from __future__ import annotations

import json
from dataclasses import asdict

from docintel.analysis import Analysis
from docintel.classify.engine import ClassificationResult
from docintel.models import Document


def render_classification_text(result: ClassificationResult) -> str:
    lines: list[str] = []
    lines.append("Classification:")
    if result.used_fallback:
        lines.append("  *** RULE-BASED FALLBACK — no LLM was used ***")
    else:
        lines.append(f"  method:                 llm ({result.model})")
    for i, match in enumerate(result.matches, start=1):
        bar_len = int(round(match.score * 20))
        bar = "#" * bar_len + "-" * (20 - bar_len)
        lines.append(f"  {i}. {match.category:<28} {match.score:5.1%} |{bar}|")
        if match.reasoning:
            lines.append(f"       {match.reasoning}")
    if result.llm_error:
        lines.append("")
        lines.append(f"  LLM unavailable: {result.llm_error}")
    if result.note:
        lines.append(f"  Note: {result.note}")
    return "\n".join(lines)


def render_classification_json(result: ClassificationResult) -> dict:
    return asdict(result)


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


def render_json(
    doc: Document,
    analysis: Analysis,
    classification: ClassificationResult | None = None,
) -> str:
    payload = {
        "path": str(doc.path),
        "format": doc.format.value,
        "metadata": doc.metadata,
        "headings": [asdict(h) for h in doc.headings],
        "links": [asdict(link) for link in doc.links],
        "analysis": asdict(analysis),
    }
    if classification is not None:
        payload["classification"] = render_classification_json(classification)
    return json.dumps(payload, indent=2)
