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

# Classify a document (top-N categories with scores + reasoning)
uv run docintel classify invoice.pdf

# Classification alongside the analysis report
uv run docintel analyze contract.docx --classify

# Index documents (or whole directories) for semantic search
uv run docintel index docs/ invoice.pdf

# Semantic search across the index
uv run docintel search "payment terms" -k 5

# Ask a question — answered from the index with source citations (RAG)
uv run docintel ask "How many vacation days do employees get?"
```

## Semantic search + RAG

`docintel index` parses, chunks (heading-aware for Markdown, overlapping
windows otherwise), embeds, and stores documents in a persistent
[ChromaDB](https://docs.trychroma.com/) index at `.docintel/chroma/`
(configurable). Re-running `index` skips unchanged files; `--reindex` forces a
rebuild.

`docintel search` embeds the query and returns the top-k most similar chunks
with scores, source files, and heading paths. `docintel ask` retrieves those
chunks and sends them to the LLM, which answers with `[n]` citations mapped
back to file + heading.

```yaml
search:
  # embedding_model: text-embedding-3-small  # no hardcoded default
  chunk_size: 1200
  overlap: 0.15
  top_k: 5
  index_path: .docintel/chroma
```

Environment override: `DOCINTEL_EMBEDDING_MODEL` (the OpenAI-compatible
endpoint and API key env var are shared with the `llm:` section).

**Fallback**: when no embedding model is configured (or the embeddings API
fails), docintel uses a deterministic local hashed bag-of-words embedder.
This is lexical matching only — the output clearly highlights it with a
`*** LOCAL FALLBACK EMBEDDINGS ***` banner. `ask` requires an LLM for the
answer itself; without one it prints the retrieved sources and exits non-zero.

## Classification

Documents are classified with an OpenAI LLM: the model ranks the top-N most
likely categories (from a built-in taxonomy of ~130 document types) with
confidence scores and short reasoning.

Configure it via `docintel.yaml` in the current directory (or
`~/.config/docintel/config.yaml`, or `--config path/to/file.yaml`) — see
[`docintel.example.yaml`](docintel.example.yaml):

```yaml
llm:
  provider: openai
  model: gpt-4o-mini          # required; no hardcoded default
  api_key_env: OPENAI_API_KEY
classification:
  top_n: 5
  # categories: [invoice, contract, other]   # override the built-in taxonomy
  # keywords:                                # extra phrases for the fallback
  #   invoice: ["remittance"]
```

Environment overrides: `DOCINTEL_MODEL`, `DOCINTEL_BASE_URL`,
`DOCINTEL_API_KEY_ENV`.

**Fallback**: when no model/API key is configured (or the LLM call fails),
docintel falls back to rule-based keyword matching. The output clearly
highlights this with a `*** RULE-BASED FALLBACK ***` banner and explains why
the LLM was not used.

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
  classify/
    categories.py      # built-in taxonomy + keyword hints
    config.py          # YAML config loader
    engine.py          # LLM classification + rule fallback
  extract/
    schemas.py         # built-in per-category field schemas
    engine.py          # LLM extraction + rule fallback
  search/
    chunking.py        # heading-aware / windowed chunking
    embeddings.py      # OpenAI embeddings + local fallback
    store.py           # ChromaDB persistent vector store
    engine.py          # index + search orchestration
  rag.py               # RAG question answering with citations
  parsers/
    markdown_parser.py
    pdf_parser.py
    docx_parser.py
tests/
```
