"""docintel - CLI for document analysis (Markdown, PDF, DOCX)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from docintel.analysis import analyze
from docintel.classify import ConfigError, classify_document, load_config
from docintel.classify.engine import ClassificationResult
from docintel.extract.engine import ExtractionResult, extract_fields
from docintel.models import Document
from docintel.report import (
    render_classification_text,
    render_extraction_text,
    render_json,
    render_text,
)

app = typer.Typer(
    name="docintel",
    help="Analyze documents: structure, statistics, and link checks.",
    no_args_is_help=True,
)


@app.callback()
def _root() -> None:
    """docintel - document intelligence CLI."""


def _resolve(path: Path) -> Path:
    if not path.exists():
        typer.echo(f"Error: file not found: {path}", err=True)
        raise typer.Exit(code=2)
    return path.resolve()


def _load_doc(path: Path) -> Document:
    """Resolve, validate, and parse a document path (shared CLI helper)."""
    from docintel.models import DocFormat
    from docintel.parsers import parse_document

    resolved = _resolve(path)
    try:
        DocFormat.from_path(resolved)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    try:
        return parse_document(resolved)
    except Exception as exc:  # noqa: BLE001 - surface parse errors to the CLI user
        typer.echo(f"Error: failed to parse {resolved.name}: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _load_cfg(config_path: Path | None):
    try:
        return load_config(config_path)
    except ConfigError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=2) from exc


def _classify(doc: Document, config_path: Path | None) -> ClassificationResult:
    return classify_document(doc, _load_cfg(config_path))


@app.command("analyze")
def analyze_cmd(
    path: Annotated[Path, typer.Argument(help="Document file to analyze", exists=False)],
    json_output: Annotated[
        bool, typer.Option("--json", help="Output the report as JSON")
    ] = False,
    no_link_check: Annotated[
        bool, typer.Option("--no-link-check", help="Skip local link validation")
    ] = False,
    classify: Annotated[
        bool, typer.Option("--classify", help="Also classify the document")
    ] = False,
    extract: Annotated[
        bool, typer.Option("--extract", help="Also extract key fields (implies --classify)")
    ] = False,
    category: Annotated[
        str | None,
        typer.Option("--category", help="Document category (schema) to use; default: auto-classify"),
    ] = None,
    config: Annotated[
        Path | None, typer.Option("--config", help="Path to docintel YAML config file")
    ] = None,
) -> None:
    """Analyze a document and print a report."""
    doc = _load_doc(path)
    analysis = analyze(doc, check_links=not no_link_check)
    classification = _classify(doc, config) if (classify or extract) else None
    extraction = (
        extract_fields(
            doc,
            _load_cfg(config),
            category=category
            or (classification.best.category if classification and classification.best else None),
        )
        if extract
        else None
    )

    if json_output:
        report = render_json(doc, analysis, classification, extraction)
    else:
        report = render_text(doc, analysis)
        if classification is not None:
            report += "\n\n" + render_classification_text(classification)
        if extraction is not None:
            report += "\n\n" + render_extraction_text(extraction)
    typer.echo(report)

    if analysis.broken_links:
        raise typer.Exit(code=1)


@app.command("classify")
def classify_cmd(
    path: Annotated[Path, typer.Argument(help="Document file to classify", exists=False)],
    json_output: Annotated[
        bool, typer.Option("--json", help="Output the result as JSON")
    ] = False,
    top_n: Annotated[
        int | None, typer.Option("--top-n", help="Number of ranked categories to show")
    ] = None,
    config: Annotated[
        Path | None, typer.Option("--config", help="Path to docintel YAML config file")
    ] = None,
) -> None:
    """Classify a document into categories (LLM with rule-based fallback)."""
    from docintel.report import render_classification_json

    doc = _load_doc(path)
    cfg = _load_cfg(config)
    if top_n is not None:
        cfg.classification.top_n = max(1, top_n)

    result = classify_document(doc, cfg)

    if json_output:
        typer.echo(json.dumps(render_classification_json(result), indent=2))
    else:
        typer.echo(render_classification_text(result))


@app.command("extract")
def extract_cmd(
    path: Annotated[Path, typer.Argument(help="Document file to extract fields from", exists=False)],
    json_output: Annotated[
        bool, typer.Option("--json", help="Output the result as JSON")
    ] = False,
    category: Annotated[
        str | None,
        typer.Option("--category", help="Document category (schema) to use; default: auto-classify"),
    ] = None,
    config: Annotated[
        Path | None, typer.Option("--config", help="Path to docintel YAML config file")
    ] = None,
) -> None:
    """Extract key fields into structured data (LLM with rule-based fallback)."""
    from docintel.report import render_extraction_json

    doc = _load_doc(path)
    cfg = _load_cfg(config)
    result = extract_fields(doc, cfg, category=category)

    if json_output:
        typer.echo(json.dumps(render_extraction_json(result), indent=2))
    else:
        typer.echo(render_extraction_text(result))


@app.command("index")
def index_cmd(
    paths: Annotated[
        list[Path], typer.Argument(help="Document files or directories to index", exists=False)
    ],
    reindex: Annotated[
        bool, typer.Option("--reindex", help="Rebuild the index even for unchanged files")
    ] = False,
    json_output: Annotated[
        bool, typer.Option("--json", help="Output the result as JSON")
    ] = False,
    config: Annotated[
        Path | None, typer.Option("--config", help="Path to docintel YAML config file")
    ] = None,
) -> None:
    """Index documents for semantic search (chunk, embed, store in ChromaDB)."""
    from docintel.report import render_index_json, render_index_text
    from docintel.search.engine import index_documents

    files = _expand_paths(paths)
    if not files:
        typer.echo("Error: no supported documents found in the given paths", err=True)
        raise typer.Exit(code=2)

    result = index_documents(files, _load_cfg(config), reindex=reindex)

    if json_output:
        typer.echo(json.dumps(render_index_json(result), indent=2))
    else:
        typer.echo(render_index_text(result))

    if result.error and result.total_chunks == 0:
        raise typer.Exit(code=1)


@app.command("search")
def search_cmd(
    query: Annotated[str, typer.Argument(help="Natural-language search query")],
    top_k: Annotated[
        int | None, typer.Option("-k", "--top-k", help="Number of results to return")
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Output the result as JSON")
    ] = False,
    config: Annotated[
        Path | None, typer.Option("--config", help="Path to docintel YAML config file")
    ] = None,
) -> None:
    """Semantic search over indexed documents."""
    from docintel.report import render_search_json, render_search_text
    from docintel.search.engine import search as run_search
    from docintel.search.store import StoreError

    cfg = _load_cfg(config)
    try:
        result = run_search(query, cfg, top_k=top_k)
    except StoreError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    if json_output:
        typer.echo(json.dumps(render_search_json(result), indent=2))
    else:
        typer.echo(render_search_text(result))

    if result.error and not result.hits:
        raise typer.Exit(code=1)


@app.command("ask")
def ask_cmd(
    query: Annotated[str, typer.Argument(help="Question to answer from indexed documents")],
    top_k: Annotated[
        int | None, typer.Option("-k", "--top-k", help="Number of source chunks to retrieve")
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Output the result as JSON")
    ] = False,
    config: Annotated[
        Path | None, typer.Option("--config", help="Path to docintel YAML config file")
    ] = None,
) -> None:
    """Answer a question from indexed documents with source citations (RAG)."""
    from docintel.rag import ask as run_ask
    from docintel.report import render_ask_json, render_ask_text
    from docintel.search.store import StoreError

    cfg = _load_cfg(config)
    try:
        result = run_ask(query, cfg, top_k=top_k)
    except StoreError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    if json_output:
        typer.echo(json.dumps(render_ask_json(result), indent=2))
    else:
        typer.echo(render_ask_text(result))

    if result.answer is None:
        raise typer.Exit(code=1)


def _expand_paths(paths: list[Path]) -> list[Path]:
    """Expand directories into supported document files (recursive)."""
    from docintel.models import DocFormat

    supported = {".md", ".markdown", ".pdf", ".docx"}
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            for candidate in sorted(path.rglob("*")):
                if candidate.is_file() and candidate.suffix.lower() in supported:
                    files.append(candidate)
        elif path.is_file():
            try:
                DocFormat.from_path(path)
                files.append(path)
            except ValueError:
                typer.echo(f"Warning: skipping unsupported file: {path}", err=True)
        else:
            typer.echo(f"Warning: path not found: {path}", err=True)
    return files


def main() -> None:
    app()
