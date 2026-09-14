"""Tests for key-field extraction (rules, typed parsers, config, CLI)."""

import json
from pathlib import Path

from typer.testing import CliRunner

from docintel.classify.config import ConfigError, load_config
from docintel.extract.engine import (
    ExtractedField,
    ExtractionResult,
    extract_fields,
    extract_with_rules,
    parse_date,
    parse_money,
    parse_typed,
)
from docintel.extract.schemas import schema_for
from docintel.cli import app
from docintel.parsers import parse_document

runner = CliRunner()


def _write_md(tmp_path: Path, content: str, name: str = "doc.md") -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


INVOICE_TEXT = """# Invoice

Invoice Number: INV-001
Invoice Date: 2026-09-01
Due Date: 15 September 2026
Bill To: Acme Corp
From: Widgets LLC
Subtotal: $1,500.00
Tax: $120.00
Total: $1,620.00
Payment Terms: Net 30
Contact: billing@widgets.example
"""

RESUME_TEXT = """# Jane Doe

Email: jane.doe@example.com
Phone: +1 (555) 123-4567
Location: Berlin, Germany

## Work Experience
Senior Engineer at Foo Inc.
"""


# ---------------------------------------------------------------------------
# Typed value parsers
# ---------------------------------------------------------------------------

def test_parse_money_variants():
    assert parse_money("$1,500.00") == (1500.0, "USD")
    assert parse_money("EUR 42") == (42.0, "EUR")
    assert parse_money("-$10.50") == (-10.5, "USD")
    assert parse_money("no money here") == (None, None)


def test_parse_date_variants():
    assert parse_date("2026-09-01") == "2026-09-01"
    assert parse_date("01/09/2026") == "2026-09-01"
    assert parse_date("15 September 2026") == "2026-09-15"
    assert parse_date("Sep 15, 2026") == "2026-09-15"
    assert parse_date("not a date") is None
    assert parse_date("2026-99-99") is None


def test_parse_typed_email_phone():
    assert parse_typed("reach me at a@b.co ok", "email") == "a@b.co"
    assert parse_typed("call +1 (555) 123-4567 now", "phone") == "+1 (555) 123-4567"
    assert parse_typed("", "text") is None


# ---------------------------------------------------------------------------
# Rule-based extraction
# ---------------------------------------------------------------------------

def test_rules_extract_invoice(tmp_path):
    doc = parse_document(_write_md(tmp_path, INVOICE_TEXT))
    result = extract_with_rules(doc, schema_for("invoice"))
    assert result.used_fallback
    assert result.note  # fallback must be highlighted
    values = result.values
    assert values["invoice_number"] == "INV-001"
    assert values["invoice_date"] == "2026-09-01"
    assert values["due_date"] == "2026-09-15"
    assert values["total_amount"] == 1620.0
    assert values["subtotal"] == 1500.0
    assert values["tax_amount"] == 120.0
    assert values["bill_to"] == "Acme Corp"
    assert values["payment_terms"] == "Net 30"
    assert values["currency"] == "USD"  # inferred from $ symbols


def test_rules_extract_resume(tmp_path):
    doc = parse_document(_write_md(tmp_path, RESUME_TEXT))
    result = extract_with_rules(doc, schema_for("resume"))
    values = result.values
    assert values["email"] == "jane.doe@example.com"
    assert values["phone"] == "+1 (555) 123-4567"
    assert values["location"] == "Berlin, Germany"


def test_extract_fields_auto_classifies(tmp_path):
    doc = parse_document(_write_md(tmp_path, INVOICE_TEXT))
    result = extract_fields(doc, load_config())
    assert result.category == "invoice"
    assert result.values["invoice_number"] == "INV-001"
    assert result.used_fallback
    assert result.llm_error  # no API key in tests


def test_extract_fields_explicit_category(tmp_path):
    doc = parse_document(_write_md(tmp_path, INVOICE_TEXT))
    result = extract_fields(doc, load_config(), category="receipt")
    assert result.category == "receipt"
    # Receipt schema should still find the total via its labels.
    assert result.values.get("total_amount") == 1620.0


def test_extract_empty_document(tmp_path):
    doc = parse_document(_write_md(tmp_path, ""))
    result = extract_fields(doc, load_config(), category="invoice")
    assert not result.fields
    assert result.missing
    assert result.note


# ---------------------------------------------------------------------------
# Config: user-defined schemas
# ---------------------------------------------------------------------------

def test_config_custom_schema(tmp_path):
    cfg_path = tmp_path / "docintel.yaml"
    cfg_path.write_text(
        "extraction:\n"
        "  schemas:\n"
        "    invoice:\n"
        "      - name: ref\n"
        "        type: text\n"
        "        labels: ['invoice number']\n"
        "    custom_doc:\n"
        "      - title\n",
        encoding="utf-8",
    )
    config = load_config(cfg_path)
    schema = config.extraction.schema_for("invoice")
    assert schema.field_names == ["ref"]  # replaces built-in entirely
    assert config.extraction.schema_for("custom_doc").field_names == ["title"]
    # Unknown category falls back to built-in/generic.
    assert config.extraction.schema_for("resume").field_names == schema_for("resume").field_names

    doc = parse_document(_write_md(tmp_path, INVOICE_TEXT, name="inv.md"))
    result = extract_fields(doc, config, category="invoice")
    assert result.values["ref"] == "INV-001"


def test_config_invalid_schema_type(tmp_path):
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(
        "extraction:\n"
        "  schemas:\n"
        "    invoice:\n"
        "      - name: x\n"
        "        type: float\n",
        encoding="utf-8",
    )
    try:
        load_config(cfg_path)
        raise AssertionError("expected ConfigError")
    except ConfigError as exc:
        assert "invalid type" in str(exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_extract_text(tmp_path):
    path = _write_md(tmp_path, INVOICE_TEXT)
    result = runner.invoke(app, ["extract", str(path), "--category", "invoice"])
    assert result.exit_code == 0
    assert "RULE-BASED FALLBACK" in result.output
    assert "INV-001" in result.output


def test_cli_extract_json(tmp_path):
    path = _write_md(tmp_path, INVOICE_TEXT)
    result = runner.invoke(app, ["extract", str(path), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["category"] == "invoice"
    assert payload["method"] == "rules"
    values = {f["name"]: f["value"] for f in payload["fields"]}
    assert values["invoice_number"] == "INV-001"


def test_cli_analyze_extract_flag(tmp_path):
    path = _write_md(tmp_path, INVOICE_TEXT)
    result = runner.invoke(app, ["analyze", str(path), "--extract", "--no-link-check"])
    assert result.exit_code == 0
    assert "Classification:" in result.output
    assert "Extracted fields:" in result.output
