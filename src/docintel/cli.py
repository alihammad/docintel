"""docintel - CLI for document analysis (Markdown, PDF, DOCX)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from docintel.analysis import analyze
from docintel.classify import ConfigError, classify_document, load_config
from docintel.classify.engine import ClassificationResult
from docintel.models import Document
from docintel.report import (
    render_classification_text,
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
    config: Annotated[
        Path | None, typer.Option("--config", help="Path to docintel YAML config file")
    ] = None,
) -> None:
    """Analyze a document and print a report."""
    doc = _load_doc(path)
    analysis = analyze(doc, check_links=not no_link_check)
    classification = _classify(doc, config) if classify else None

    if json_output:
        report = render_json(doc, analysis, classification)
    else:
        report = render_text(doc, analysis)
        if classification is not None:
            report += "\n\n" + render_classification_text(classification)
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


def main() -> None:
    app()
