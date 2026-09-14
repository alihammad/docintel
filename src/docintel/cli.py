"""docintel - CLI for document analysis (Markdown, PDF, DOCX)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from docintel.analysis import analyze
from docintel.report import render_json, render_text

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


@app.command("analyze")
def analyze_cmd(
    path: Annotated[Path, typer.Argument(help="Document file to analyze", exists=False)],
    json_output: Annotated[
        bool, typer.Option("--json", help="Output the report as JSON")
    ] = False,
    no_link_check: Annotated[
        bool, typer.Option("--no-link-check", help="Skip local link validation")
    ] = False,
) -> None:
    """Analyze a document and print a report."""
    from docintel.models import DocFormat
    from docintel.parsers import parse_document

    resolved = _resolve(path)
    try:
        DocFormat.from_path(resolved)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    try:
        doc = parse_document(resolved)
    except Exception as exc:  # noqa: BLE001 - surface parse errors to the CLI user
        typer.echo(f"Error: failed to parse {resolved.name}: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    analysis = analyze(doc, check_links=not no_link_check)
    report = render_json(doc, analysis) if json_output else render_text(doc, analysis)
    typer.echo(report)

    if analysis.broken_links:
        raise typer.Exit(code=1)


def main() -> None:
    app()
