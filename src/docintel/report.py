"""Render analysis results as text or JSON reports."""

from __future__ import annotations

import json
from dataclasses import asdict

from docintel.analysis import Analysis
from docintel.classify.engine import ClassificationResult
from docintel.extract.engine import ExtractionResult
from docintel.models import Document
from docintel.rag import AskResult
from docintel.search.engine import IndexResult, SearchResult


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


def render_extraction_text(result: ExtractionResult) -> str:
    lines: list[str] = []
    lines.append("Extracted fields:")
    lines.append(f"  category:               {result.category}")
    if result.used_fallback:
        lines.append("  *** RULE-BASED FALLBACK — no LLM was used ***")
    else:
        lines.append(f"  method:                 llm ({result.model})")
    for f in result.fields:
        value = f"{f.value}" if f.value is not None else ""
        lines.append(f"  {f.name:<22} {value:<24} ({f.confidence:.0%}, {f.source})")
        if f.evidence:
            lines.append(f"       from: {f.evidence}")
    if result.missing:
        lines.append(f"  missing:                {', '.join(result.missing)}")
    if result.llm_error:
        lines.append("")
        lines.append(f"  LLM unavailable: {result.llm_error}")
    if result.note:
        lines.append(f"  Note: {result.note}")
    return "\n".join(lines)


def render_extraction_json(result: ExtractionResult) -> dict:
    return asdict(result)


def render_index_text(result: IndexResult) -> str:
    lines: list[str] = []
    lines.append("Index:")
    if result.used_fallback:
        lines.append("  *** LOCAL FALLBACK EMBEDDINGS — lexical matching only ***")
    else:
        lines.append(f"  embeddings:             openai ({result.model})")
    for doc in result.docs:
        detail = f"{doc.chunks} chunks" if doc.chunks else ""
        lines.append(f"  {doc.status:<12} {doc.path}" + (f"  ({detail})" if detail else ""))
    lines.append(f"  total new chunks:       {result.total_chunks}")
    if result.error:
        lines.append("")
        lines.append(f"  Embedding API unavailable: {result.error}")
    if result.note:
        lines.append(f"  Note: {result.note}")
    return "\n".join(lines)


def render_index_json(result: IndexResult) -> dict:
    return asdict(result)


def render_search_text(result: SearchResult) -> str:
    lines: list[str] = []
    lines.append(f"Search: {result.query}")
    if result.used_fallback:
        lines.append("  *** LOCAL FALLBACK EMBEDDINGS — lexical matching only ***")
    else:
        lines.append(f"  embeddings:             openai ({result.model})")
    if not result.hits:
        lines.append("  (no results)")
    for i, hit in enumerate(result.hits, start=1):
        location = hit.doc_path + (f" — {hit.heading_path}" if hit.heading_path else "")
        bar_len = int(round(max(0.0, min(1.0, hit.score)) * 20))
        bar = "#" * bar_len + "-" * (20 - bar_len)
        lines.append(f"  {i}. [{hit.score:6.1%}] |{bar}| {location} (chunk {hit.chunk_index})")
        lines.append(f"       {hit.snippet}")
    if result.error:
        lines.append("")
        lines.append(f"  Embedding API unavailable: {result.error}")
    if result.note:
        lines.append(f"  Note: {result.note}")
    return "\n".join(lines)


def render_search_json(result: SearchResult) -> dict:
    return asdict(result)


def render_ask_text(result: AskResult) -> str:
    lines: list[str] = []
    lines.append(f"Question: {result.query}")
    if result.used_fallback_retrieval:
        lines.append("  *** LOCAL FALLBACK RETRIEVAL — lexical matching only ***")
    if result.answer:
        lines.append("")
        lines.append("Answer:")
        for line in result.answer.splitlines():
            lines.append(f"  {line}")
    elif result.llm_error:
        lines.append("")
        lines.append(f"  LLM unavailable: {result.llm_error}")
    if result.citations:
        lines.append("")
        lines.append("Sources:")
        for c in result.citations:
            location = c.doc_path + (f" — {c.heading_path}" if c.heading_path else "")
            lines.append(f"  [{c.number}] {location}")
            lines.append(f"      {c.snippet}")
    if result.note:
        lines.append(f"  Note: {result.note}")
    return "\n".join(lines)


def render_ask_json(result: AskResult) -> dict:
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
    extraction: ExtractionResult | None = None,
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
    if extraction is not None:
        payload["extraction"] = render_extraction_json(extraction)
    return json.dumps(payload, indent=2)
