# docintel

A CLI tool for analyzing documents — structure, statistics, and link checks.

Supported formats: **Markdown**, **PDF** (via `pypdf`), **DOCX** (via `python-docx`).

## Install

```bash
uv sync
```

## Usage

```bash
# Analyze a document (text report)
uv run docintel analyze README.md

# JSON output
uv run docintel analyze report.pdf --json

# Skip local link validation
uv run docintel analyze docs/guide.md --no-link-check
```

Exit codes:

- `0` — success
- `1` — parse failure, or broken local links were found
- `2` — usage error (missing file, unsupported format)

## What it reports

- **Statistics**: word, character, line, and sentence counts; average words per sentence
- **Structure**: heading outline and depth
- **Links**: external/internal counts, plus broken local link detection (files that no longer exist)
- **Top words**: most frequent non-stopwords
- **Metadata**: front matter (Markdown), document properties (PDF/DOCX)

## Development

```bash
uv sync                 # install deps incl. dev group
uv run pytest           # run tests
uv run docintel --help  # CLI help
```

## Project structure

```
src/docintel/
  __init__.py          # entry point (main)
  cli.py               # Typer CLI
  models.py            # Document, Heading, Link dataclasses
  analysis.py          # statistics + link checking
  report.py            # text/JSON rendering
  parsers/
    markdown_parser.py
    pdf_parser.py
    docx_parser.py
tests/
```
