"""Tests for the analysis engine and CLI."""

from pathlib import Path

from typer.testing import CliRunner

from docintel.analysis import analyze
from docintel.cli import app
from docintel.parsers import parse_document

runner = CliRunner()


def _write_md(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "doc.md"
    p.write_text(content, encoding="utf-8")
    return p


def test_analyze_stats(tmp_path):
    path = _write_md(
        tmp_path,
        "# Title\n\nHello world. This is a test document! Really?\n",
    )
    doc = parse_document(path)
    result = analyze(doc)
    assert result.word_count == 10
    assert result.sentence_count == 3
    assert result.heading_count == 1
    assert result.broken_links == []


def test_broken_local_link_detected(tmp_path):
    path = _write_md(tmp_path, "See [missing](./nope.md) for details.\n")
    doc = parse_document(path)
    result = analyze(doc)
    assert len(result.broken_links) == 1
    assert "nope.md" in result.broken_links[0].target


def test_valid_local_link_ok(tmp_path):
    (tmp_path / "other.md").write_text("hi", encoding="utf-8")
    path = _write_md(tmp_path, "See [other](./other.md).\n")
    doc = parse_document(path)
    result = analyze(doc)
    assert result.broken_links == []


def test_cli_text_report(tmp_path):
    path = _write_md(tmp_path, "# Doc\n\nSome words here.\n")
    result = runner.invoke(app, ["analyze", str(path)])
    assert result.exit_code == 0
    assert "Statistics:" in result.output
    assert "words:" in result.output


def test_cli_json_report(tmp_path):
    path = _write_md(tmp_path, "# Doc\n\nSome words here.\n")
    result = runner.invoke(app, ["analyze", str(path), "--json"])
    assert result.exit_code == 0
    import json

    payload = json.loads(result.output)
    assert payload["format"] == "markdown"
    assert payload["analysis"]["word_count"] == 5


def test_cli_missing_file(tmp_path):
    result = runner.invoke(app, ["analyze", str(tmp_path / "ghost.md")])
    assert result.exit_code == 2


def test_cli_unsupported_format(tmp_path):
    p = tmp_path / "file.txt"
    p.write_text("hello", encoding="utf-8")
    result = runner.invoke(app, ["analyze", str(p)])
    assert result.exit_code == 2


def test_cli_exit_code_on_broken_link(tmp_path):
    path = _write_md(tmp_path, "See [missing](./nope.md).\n")
    result = runner.invoke(app, ["analyze", str(path)])
    assert result.exit_code == 1
    assert "BROKEN" in result.output
