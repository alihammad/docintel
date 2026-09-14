"""Tests for the Markdown parser."""

from pathlib import Path

from docintel.parsers.markdown_parser import parse_markdown


def _write(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "doc.md"
    p.write_text(content, encoding="utf-8")
    return p


def test_headings_and_links(tmp_path):
    path = _write(
        tmp_path,
        """# Title

Some intro text with a [link](https://example.com) and a [local one](./other.md).

## Section

More text.

### Subsection
""",
    )
    doc = parse_markdown(path)
    assert [(h.level, h.text) for h in doc.headings] == [
        (1, "Title"),
        (2, "Section"),
        (3, "Subsection"),
    ]
    assert len(doc.links) == 2
    assert doc.links[0].is_external is True
    assert doc.links[1].is_external is False
    assert doc.links[1].target == "./other.md"


def test_code_blocks_are_skipped(tmp_path):
    path = _write(
        tmp_path,
        """# Real heading

```
# not a heading
[fake](http://fake.example)
```

Text after.
""",
    )
    doc = parse_markdown(path)
    assert [h.text for h in doc.headings] == ["Real heading"]
    assert doc.links == []


def test_front_matter(tmp_path):
    path = _write(
        tmp_path,
        """---
title: My Doc
author: Ali
---

# Body
""",
    )
    doc = parse_markdown(path)
    assert doc.metadata == {"title": "My Doc", "author": "Ali"}
    assert [h.text for h in doc.headings] == ["Body"]


def test_word_count(tmp_path):
    path = _write(tmp_path, "one two three four five")
    doc = parse_markdown(path)
    assert doc.word_count == 5
