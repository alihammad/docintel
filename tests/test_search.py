"""Tests for semantic search + RAG (chunking, embeddings, store, CLI)."""

import json
import sys
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

from docintel.classify.config import Config
from docintel.cli import app
from docintel.models import DocFormat, Document
from docintel.rag import ask
from docintel.search.chunking import chunk_document
from docintel.search.embeddings import local_embed
from docintel.search.engine import index_documents, search

runner = CliRunner()


DOC_A = """# Invoice Guide

## Totals
The invoice total is calculated as subtotal plus tax.
Tax rate is 19 percent in Germany.

## Payment
Payment terms are Net 30 days from the invoice date.
"""

DOC_B = """# Onboarding Handbook

## Vacation
Employees receive 30 days of vacation per year.
Vacation must be requested two weeks in advance.

## Office
The Berlin office is open Monday through Friday.
"""


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture()
def cfg(tmp_path: Path) -> Config:
    config = Config()
    config.search.index_path = str(tmp_path / "index" / "chroma")
    return config


def _md_doc(path: Path, text: str) -> Document:
    return Document(path=path, format=DocFormat.MARKDOWN, text=text)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def test_chunking_is_heading_aware():
    doc = _md_doc(Path("guide.md"), DOC_A)
    chunks = chunk_document(doc, chunk_size=1200)
    assert chunks
    paths = {c.heading_path for c in chunks}
    assert "Invoice Guide > Totals" in paths
    assert "Invoice Guide > Payment" in paths
    # Chunk indices are sequential and ids are stable.
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert chunks[0].chunk_id == chunk_document(doc)[0].chunk_id


def test_chunking_windows_long_text():
    long_text = " ".join(f"Sentence number {i} about widgets." for i in range(300))
    doc = _md_doc(Path("long.md"), long_text)
    chunks = chunk_document(doc, chunk_size=400)
    assert len(chunks) > 1
    assert all(len(c.text) <= 400 + 100 for c in chunks)  # allow overlap slack


def test_chunking_empty_document():
    assert chunk_document(_md_doc(Path("e.md"), "")) == []


# ---------------------------------------------------------------------------
# Local fallback embeddings
# ---------------------------------------------------------------------------

def test_local_embed_deterministic_and_normalized():
    v1 = local_embed(["hello world"])[0]
    v2 = local_embed(["hello world"])[0]
    assert v1 == v2
    assert len(v1) == 512
    norm = sum(x * x for x in v1) ** 0.5
    assert abs(norm - 1.0) < 1e-6


def test_local_embed_lexical_similarity():
    a, b, c = local_embed(
        ["invoice total tax payment", "invoice total tax amount", "vacation office monday"]
    )

    def cos(x, y):
        return sum(i * j for i, j in zip(x, y))

    assert cos(a, b) > cos(a, c)


# ---------------------------------------------------------------------------
# Index + search round trip (local fallback, ChromaDB in tmp_path)
# ---------------------------------------------------------------------------

def test_index_and_search_roundtrip(tmp_path: Path, cfg: Config):
    a = _write(tmp_path, "invoice_guide.md", DOC_A)
    b = _write(tmp_path, "handbook.md", DOC_B)

    result = index_documents([a, b], cfg)
    assert result.error is None or "failed" not in (result.error or "")
    assert result.used_fallback  # no embedding model configured
    assert result.total_chunks >= 4
    assert {d.status for d in result.docs} == {"indexed"}

    found = search("How many vacation days do employees get?", cfg, top_k=3)
    assert found.used_fallback
    assert found.hits
    assert found.hits[0].doc_path == str(b.resolve())
    assert "vacation" in found.hits[0].snippet.lower() or found.hits[0].heading_path


def test_index_is_idempotent(tmp_path: Path, cfg: Config):
    a = _write(tmp_path, "invoice_guide.md", DOC_A)
    first = index_documents([a], cfg)
    assert first.docs[0].status == "indexed"
    second = index_documents([a], cfg)
    assert second.docs[0].status == "unchanged"
    assert second.total_chunks == 0


def test_reindex_flag_forces_rebuild(tmp_path: Path, cfg: Config):
    a = _write(tmp_path, "invoice_guide.md", DOC_A)
    index_documents([a], cfg)
    result = index_documents([a], cfg, reindex=True)
    assert result.docs[0].status == "reindexed"
    assert result.total_chunks > 0


def test_search_empty_index(cfg: Config):
    result = search("anything", cfg)
    assert result.hits == []
    assert "empty" in (result.note or "").lower()


def test_missing_file_reported(tmp_path: Path, cfg: Config):
    result = index_documents([tmp_path / "nope.md"], cfg)
    assert result.docs[0].status == "missing"


# ---------------------------------------------------------------------------
# RAG ask (no LLM configured -> sources only)
# ---------------------------------------------------------------------------

def test_ask_without_llm_returns_sources(tmp_path: Path, cfg: Config):
    b = _write(tmp_path, "handbook.md", DOC_B)
    index_documents([b], cfg)

    result = ask("How many vacation days?", cfg, top_k=2)
    assert result.method == "none"
    assert result.answer is None
    assert result.llm_error  # explains the LLM is not configured
    assert result.citations
    assert result.citations[0].number == 1


def test_ask_with_mocked_llm_cites_sources(tmp_path: Path, cfg: Config, monkeypatch):
    b = _write(tmp_path, "handbook.md", DOC_B)
    index_documents([b], cfg)
    cfg.llm.model = "mock-model"
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    captured: dict = {}

    class _FakeMessage:
        content = "Employees receive 30 days of vacation per year [1]."

    class _FakeChoice:
        message = _FakeMessage()

    class _FakeResponse:
        choices = [_FakeChoice()]

    class _FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _FakeResponse()

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeOpenAI:
        def __init__(self, *a, **k):
            self.chat = _FakeChat()

    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = _FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_module)

    result = ask("How many vacation days?", cfg, top_k=2)
    assert result.method == "llm"
    assert result.answer == "Employees receive 30 days of vacation per year [1]."
    # Only the referenced citation [1] is kept.
    assert [c.number for c in result.citations] == [1]
    # Grounding prompt contains the source text and the question.
    prompt = captured["messages"][1]["content"]
    assert "How many vacation days?" in prompt
    assert "vacation" in prompt.lower()


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

def test_cli_index_search_flow(tmp_path: Path, cfg: Config):
    b = _write(tmp_path, "handbook.md", DOC_B)
    config_file = tmp_path / "docintel.yaml"
    config_file.write_text(
        f"search:\n  index_path: {cfg.search.index_path}\n  top_k: 3\n",
        encoding="utf-8",
    )

    res = runner.invoke(app, ["index", str(b), "--config", str(config_file), "--json"])
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["total_chunks"] > 0
    assert payload["method"] == "local-fallback"

    res = runner.invoke(
        app, ["search", "vacation days", "--config", str(config_file), "--json"]
    )
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["hits"]
    assert "handbook.md" in payload["hits"][0]["doc_path"]

    res = runner.invoke(app, ["search", "vacation days", "--config", str(config_file)])
    assert res.exit_code == 0
    assert "LOCAL FALLBACK" in res.output  # fallback must be highlighted


def test_cli_index_directory_expansion(tmp_path: Path, cfg: Config):
    docs = tmp_path / "docs"
    docs.mkdir()
    _write(docs, "a.md", DOC_A)
    _write(docs, "b.md", DOC_B)
    _write(docs, "skip.txt", "not a document")
    config_file = tmp_path / "docintel.yaml"
    config_file.write_text(f"search:\n  index_path: {cfg.search.index_path}\n", encoding="utf-8")

    res = runner.invoke(app, ["index", str(docs), "--config", str(config_file), "--json"])
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert len(payload["docs"]) == 2  # txt file skipped


def test_cli_ask_without_llm_exits_nonzero(tmp_path: Path, cfg: Config):
    b = _write(tmp_path, "handbook.md", DOC_B)
    config_file = tmp_path / "docintel.yaml"
    config_file.write_text(f"search:\n  index_path: {cfg.search.index_path}\n", encoding="utf-8")
    runner.invoke(app, ["index", str(b), "--config", str(config_file)])

    res = runner.invoke(app, ["ask", "vacation?", "--config", str(config_file)])
    assert res.exit_code == 1
    assert "Sources:" in res.output
