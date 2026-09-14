"""Tests for document classification (rule fallback, config, CLI)."""

import json
from pathlib import Path

from typer.testing import CliRunner

from docintel.classify.config import ConfigError, load_config
from docintel.classify.engine import classify_document, classify_with_rules
from docintel.cli import app
from docintel.models import DocFormat, Document
from docintel.parsers import parse_document

runner = CliRunner()


def _write_md(tmp_path: Path, content: str, name: str = "doc.md") -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def _doc(path: Path) -> Document:
    return parse_document(path)


INVOICE_TEXT = """# Invoice

Invoice Number: INV-001
Bill To: Acme Corp
Amount Due: $1,500.00
Payment Terms: Net 30
Subtotal: $1,500.00
"""

RESUME_TEXT = """# Jane Doe

## Work Experience
Senior Engineer at Foo Inc.

## Education
BSc Computer Science

## Skills
Python, testing, CLI design
"""


# ---------------------------------------------------------------------------
# Rule-based classification
# ---------------------------------------------------------------------------

def test_rules_classify_invoice(tmp_path):
    doc = _doc(_write_md(tmp_path, INVOICE_TEXT))
    config = load_config()
    result = classify_document(doc, config)
    # No API key configured in tests -> rule fallback.
    assert result.used_fallback
    assert result.best is not None
    assert result.best.category == "invoice"
    assert result.note  # fallback must be highlighted


def test_rules_classify_resume(tmp_path):
    doc = _doc(_write_md(tmp_path, RESUME_TEXT))
    result = classify_document(doc, load_config())
    assert result.best.category == "resume"


def test_rules_top_n_ranking(tmp_path):
    doc = _doc(_write_md(tmp_path, INVOICE_TEXT))
    config = load_config()
    config.classification.top_n = 3
    result = classify_with_rules(doc, config.classification, 3)
    assert len(result.matches) <= 3
    scores = [m.score for m in result.matches]
    assert scores == sorted(scores, reverse=True)
    assert all(m.reasoning for m in result.matches)


def test_empty_document_classified_as_other(tmp_path):
    doc = _doc(_write_md(tmp_path, ""))
    result = classify_document(doc, load_config())
    assert result.best.category == "other"


def test_no_keyword_matches_falls_to_other(tmp_path):
    doc = _doc(_write_md(tmp_path, "xyzzy frobnicate qwertyuiop zxcvbnm\n"))
    result = classify_document(doc, load_config())
    assert result.best.category == "other"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def test_load_defaults_without_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "noconfig"))
    config = load_config()
    assert config.source is None
    assert "invoice" in config.classification.categories


def test_load_yaml_config(tmp_path):
    cfg_path = tmp_path / "docintel.yaml"
    cfg_path.write_text(
        "llm:\n"
        "  model: test-model\n"
        "classification:\n"
        "  top_n: 2\n"
        "  categories: [invoice, contract, other]\n"
        "  keywords:\n"
        "    invoice: ['Due Date']\n",
        encoding="utf-8",
    )
    config = load_config(cfg_path)
    assert config.llm.model == "test-model"
    assert config.classification.top_n == 2
    assert config.classification.categories == ["invoice", "contract", "other"]
    # user keywords merged (lowercased) and built-ins preserved
    assert "due date" in config.classification.keywords["invoice"]
    assert "bill to" in config.classification.keywords["invoice"]


def test_missing_explicit_config_errors(tmp_path):
    try:
        load_config(tmp_path / "nope.yaml")
        raise AssertionError("expected ConfigError")
    except ConfigError:
        pass


def test_invalid_yaml_errors(tmp_path):
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text("llm: [unclosed\n", encoding="utf-8")
    try:
        load_config(cfg_path)
        raise AssertionError("expected ConfigError")
    except ConfigError:
        pass


def test_env_override_model(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCINTEL_MODEL", "env-model")
    config = load_config()
    assert config.llm.model == "env-model"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_classify_command(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DOCINTEL_MODEL", raising=False)
    path = _write_md(tmp_path, INVOICE_TEXT)
    result = runner.invoke(app, ["classify", str(path)])
    assert result.exit_code == 0
    assert "invoice" in result.output
    assert "RULE-BASED FALLBACK" in result.output  # fallback highlighted


def test_cli_classify_json(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DOCINTEL_MODEL", raising=False)
    path = _write_md(tmp_path, INVOICE_TEXT)
    result = runner.invoke(app, ["classify", str(path), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["method"] == "rules"
    assert payload["matches"][0]["category"] == "invoice"
    assert payload["note"]


def test_cli_classify_top_n(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DOCINTEL_MODEL", raising=False)
    path = _write_md(tmp_path, INVOICE_TEXT)
    result = runner.invoke(app, ["classify", str(path), "--json", "--top-n", "2"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert len(payload["matches"]) <= 2


def test_cli_analyze_with_classify_flag(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DOCINTEL_MODEL", raising=False)
    path = _write_md(tmp_path, INVOICE_TEXT)
    result = runner.invoke(app, ["analyze", str(path), "--classify"])
    assert result.exit_code == 0
    assert "Statistics:" in result.output
    assert "Classification:" in result.output
    assert "invoice" in result.output


def test_cli_analyze_classify_json(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DOCINTEL_MODEL", raising=False)
    path = _write_md(tmp_path, INVOICE_TEXT)
    result = runner.invoke(app, ["analyze", str(path), "--classify", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert "classification" in payload
    assert payload["classification"]["matches"][0]["category"] == "invoice"


def test_cli_classify_missing_file(tmp_path):
    result = runner.invoke(app, ["classify", str(tmp_path / "ghost.md")])
    assert result.exit_code == 2
