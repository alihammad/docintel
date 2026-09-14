"""Key-field extraction: LLM-based (OpenAI) with a rule-based fallback.

Mirrors the classify engine's structure: try the LLM when configured,
otherwise (or on any failure) fall back to deterministic label/regex rules.
The fallback is always clearly flagged in the result.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from docintel.extract.schemas import FieldSpec, Schema, schema_for
from docintel.models import Document

if TYPE_CHECKING:  # avoid a circular import at runtime (classify.config -> extract.schemas)
    from docintel.classify.config import Config, LLMConfig

# Cap how much document text is sent to the LLM (chars).
MAX_LLM_CHARS = 12_000

Method = str  # "llm" | "rules"


@dataclass
class ExtractedField:
    name: str
    value: str | float | None
    type: str = "text"
    source: Method = "rules"
    confidence: float = 1.0
    evidence: str | None = None  # the source line/snippet the value came from


@dataclass
class ExtractionResult:
    category: str
    fields: list[ExtractedField] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)  # schema fields not found
    method: Method = "rules"
    model: str | None = None
    note: str | None = None
    llm_error: str | None = None

    @property
    def used_fallback(self) -> bool:
        return self.method == "rules"

    @property
    def values(self) -> dict[str, str | float | None]:
        return {f.name: f.value for f in self.fields}


# ---------------------------------------------------------------------------
# Value parsing / normalization per field type
# ---------------------------------------------------------------------------

_MONEY_RE = re.compile(
    # Lookbehind: don't match digits inside identifiers like "INV-001".
    r"(?<![\w-])"
    r"(?P<sign>-)?\s*(?P<symbol>[$€£])?\s*(?P<num>\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?P<suffix>\s?(?:USD|EUR|GBP|CAD|AUD|INR|JPY))?(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"(?P<y>\d{4})[-/.](?P<m>\d{1,2})[-/.](?P<d>\d{1,2})"
    r"|(?P<d2>\d{1,2})[-/.](?P<m2>\d{1,2})[-/.](?P<y2>\d{4})"
    r"|(?P<mon>\d{1,2})\s+(?P<month>[A-Za-z]{3,9})\.?,?\s+(?P<y3>\d{4})"
    r"|(?P<month2>[A-Za-z]{3,9})\.?\s+(?P<d3>\d{1,2})(?:st|nd|rd|th)?,?\s+(?P<y4>\d{4})",
)
_MONTHS: dict[str, int] = {}
for _i, _m in enumerate(
    "january february march april may june july august september october november december".split()
):
    _MONTHS[_m] = _i + 1
    _MONTHS[_m[:3]] = _i + 1  # abbreviations: jan, feb, ...
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"\+?\d[\d\s().-]{6,}\d")
_CURRENCY_CODES = ("usd", "eur", "gbp", "cad", "aud", "inr", "jpy")
_CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP"}


def parse_money(raw: str) -> tuple[float | None, str | None]:
    """Return (amount, currency) parsed from a money string, or (None, None)."""
    m = _MONEY_RE.search(raw)
    if not m:
        return None, None
    num = m.group("num").replace(",", "").replace(" ", "")
    try:
        amount = float(num)
    except ValueError:
        return None, None
    if m.group("sign"):
        amount = -amount
    currency: str | None = None
    if m.group("suffix"):
        currency = m.group("suffix").strip().upper()
    elif m.group("symbol"):
        currency = _CURRENCY_SYMBOLS.get(m.group("symbol"))
    else:
        lowered = raw.lower()
        for code in _CURRENCY_CODES:
            if re.search(rf"\b{code}\b", lowered):
                currency = code.upper()
                break
    return amount, currency


def parse_date(raw: str) -> str | None:
    """Normalize a date string to ISO format (YYYY-MM-DD), or None."""
    m = _DATE_RE.search(raw)
    if not m:
        return None
    if m.group("y"):
        y, mo, d = int(m.group("y")), int(m.group("m")), int(m.group("d"))
    elif m.group("y2"):
        d, mo, y = int(m.group("d2")), int(m.group("m2")), int(m.group("y2"))
    elif m.group("y3"):
        mo = _MONTHS.get(m.group("month").lower())
        if mo is None:
            return None
        d, y = int(m.group("mon")), int(m.group("y3"))
    else:
        mo = _MONTHS.get(m.group("month2").lower())
        if mo is None:
            return None
        d, y = int(m.group("d3")), int(m.group("y4"))
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        return None
    return f"{y:04d}-{mo:02d}-{d:02d}"


def parse_typed(value: str, ftype: str) -> str | float | None:
    """Parse/normalize a raw string value according to the field type."""
    value = value.strip()
    if not value:
        return None
    if ftype == "money":
        amount, _ = parse_money(value)
        return amount
    if ftype == "date":
        return parse_date(value)
    if ftype == "email":
        m = _EMAIL_RE.search(value)
        return m.group(0) if m else None
    if ftype == "phone":
        m = _PHONE_RE.search(value)
        return m.group(0).strip() if m else None
    return value


# ---------------------------------------------------------------------------
# Rule-based extraction (label: value lines + typed regex scan)
# ---------------------------------------------------------------------------

def _split_lines(doc: Document) -> list[str]:
    return [line.strip() for line in doc.text.splitlines() if line.strip()]


def _clean_value(raw: str) -> str:
    """Strip markdown emphasis/bold and trailing punctuation from a value."""
    raw = re.sub(r"\*{1,3}|_{2}", "", raw).strip()
    return raw.rstrip("|").strip()


def _find_by_labels(lines: list[str], spec: FieldSpec) -> ExtractedField | None:
    """Find the first line matching 'label: value' (or 'label - value') for any synonym."""
    for line in lines:
        # Skip markdown table separator rows.
        if re.fullmatch(r"[\s|:-]+", line):
            continue
        lowered = line.lower()
        for label in spec.labels:
            for sep in (":", " - ", " – ", "\t"):
                idx = lowered.find(label + sep)
                if idx == -1:
                    continue
                # Require a word boundary before the label so 'total' does not
                # match inside 'subtotal'.
                if idx > 0 and (lowered[idx - 1].isalnum()):
                    continue
                value = _clean_value(line[idx + len(label) + len(sep):])
                # Table cells: take the first non-empty cell after the label.
                if "|" in value:
                    cells = [c.strip() for c in value.split("|") if c.strip()]
                    value = cells[0] if cells else ""
                parsed = parse_typed(value, spec.type)
                if parsed is not None:
                    return ExtractedField(
                        name=spec.name,
                        value=parsed,
                        type=spec.type,
                        source="rules",
                        confidence=0.9 if sep == ":" else 0.7,
                        evidence=line[:120],
                    )
    return None


def _find_by_pattern(lines: list[str], spec: FieldSpec) -> ExtractedField | None:
    """Type-driven scan when no labeled line matched.

    Only unambiguous types are scanned (email/phone); guessing the first
    money/date occurrence in the document would produce misleading values.
    A 'currency' field is inferred from money symbols/codes in the text.
    """
    text = "\n".join(lines)
    if spec.type == "email" and (m := _EMAIL_RE.search(text)):
        return ExtractedField(spec.name, m.group(0), spec.type, "rules", 0.6, m.group(0))
    if spec.type == "phone" and (m := _PHONE_RE.search(text)):
        return ExtractedField(spec.name, m.group(0).strip(), spec.type, "rules", 0.5, m.group(0))
    if spec.name == "currency":
        for symbol, code in _CURRENCY_SYMBOLS.items():
            if symbol in text:
                return ExtractedField(spec.name, code, spec.type, "rules", 0.5, None)
        for code in _CURRENCY_CODES:
            if re.search(rf"\b{code}\b", text, re.IGNORECASE):
                return ExtractedField(spec.name, code.upper(), spec.type, "rules", 0.5, None)
    return None


def extract_with_rules(doc: Document, schema: Schema) -> ExtractionResult:
    lines = _split_lines(doc)
    found: list[ExtractedField] = []
    missing: list[str] = []
    for spec in schema.fields:
        extracted = _find_by_labels(lines, spec) or _find_by_pattern(lines, spec)
        if extracted is not None:
            found.append(extracted)
        else:
            missing.append(spec.name)
    return ExtractionResult(
        category=schema.category,
        fields=found,
        missing=missing,
        method="rules",
        note=(
            "Rule-based fallback: no LLM was used (missing API key/model or LLM call failed). "
            "Values come from labeled lines and typed pattern scans."
        ),
    )


# ---------------------------------------------------------------------------
# LLM extraction (OpenAI)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a document field-extraction engine. Given a document's text and a list of "
    "fields to extract, return the values found in the document. Respond ONLY with JSON. "
    "Use null for fields that are not present. Never invent values."
)


def _build_user_prompt(doc: Document, schema: Schema) -> str:
    text = doc.text
    if len(text) > MAX_LLM_CHARS:
        text = text[:MAX_LLM_CHARS] + "\n...[truncated]"
    field_defs = [
        {"name": f.name, "type": f.type, "hint_labels": list(f.labels)} for f in schema.fields
    ]
    return (
        f"Document filename: {doc.path.name}\n"
        f"Document category: {schema.category}\n\n"
        f"Document text:\n{text}\n\n"
        f"Fields to extract:\n{json.dumps(field_defs, indent=2)}\n\n"
        "Return JSON with this exact shape:\n"
        '{"fields": [{"name": "<field name>", "value": <string|number|null>, '
        '"confidence": <float 0-1>, "evidence": "<short quote from the document>"}]}\n'
        "Normalize dates to YYYY-MM-DD and money to plain numbers (no symbols). "
        "Include every requested field, using null when not found."
    )


def _parse_llm_json(content: str, schema: Schema) -> tuple[list[ExtractedField], list[str]]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", content)
    data = json.loads(content)
    raw_fields = data.get("fields", [])
    if not isinstance(raw_fields, list):
        raise ValueError("'fields' must be a list")

    by_name = {f.name: f for f in schema.fields}
    found: dict[str, ExtractedField] = {}
    for item in raw_fields:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        spec = by_name.get(name)
        if spec is None:
            continue  # ignore hallucinated fields
        raw_value = item.get("value")
        if raw_value is None or (isinstance(raw_value, str) and not raw_value.strip()):
            continue
        # Re-validate/normalize through the same typed parser as the rules path.
        if spec.type == "money" and isinstance(raw_value, (int, float)):
            value: str | float | None = float(raw_value)
        else:
            value = parse_typed(str(raw_value), spec.type)
            if value is None and spec.type == "text":
                value = str(raw_value).strip()  # keep free text even if unparseable
        if value is None:
            continue
        try:
            confidence = max(0.0, min(1.0, float(item.get("confidence", 0.9))))
        except (TypeError, ValueError):
            confidence = 0.9
        found[name] = ExtractedField(
            name=name,
            value=value,
            type=spec.type,
            source="llm",
            confidence=round(confidence, 3),
            evidence=(str(item["evidence"]).strip()[:120] if item.get("evidence") else None),
        )
    if not found:
        raise ValueError("LLM returned no valid fields")
    missing = [f.name for f in schema.fields if f.name not in found]
    return list(found.values()), missing


def extract_with_llm(
    doc: Document, schema: Schema, llm: "LLMConfig"
) -> tuple[list[ExtractedField], list[str]]:
    """Call OpenAI to extract fields. Raises on any failure."""
    from openai import OpenAI  # deferred import so the CLI works without the SDK installed

    client = OpenAI(base_url=llm.base_url) if llm.base_url else OpenAI()
    response = client.chat.completions.create(
        model=llm.model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(doc, schema)},
        ],
    )
    content = response.choices[0].message.content or ""
    return _parse_llm_json(content, schema)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def resolve_schema(category: str | None, config: "Config") -> Schema:
    """Pick the schema for an explicit category, or classify the doc first (done by caller)."""
    if category:
        return schema_for(category)
    return schema_for("other")


def extract_fields(
    doc: Document,
    config: "Config",
    category: str | None = None,
) -> ExtractionResult:
    """Extract key fields for a document.

    If ``category`` is None, the document is classified first and the best
    category's schema is used. Uses the LLM when available, else falls back
    to deterministic rules (always flagged in the result).
    """
    from docintel.classify.engine import _llm_available, classify_document

    if category is None:
        classification = classify_document(doc, config)
        category = classification.best.category if classification.best else "other"

    schema = config.extraction.schema_for(category)

    if not doc.text.strip():
        return ExtractionResult(
            category=schema.category,
            missing=list(schema.field_names),
            method="rules",
            note="Document is empty; nothing to extract.",
        )

    available, reason = _llm_available(config.llm)
    if not available:
        result = extract_with_rules(doc, schema)
        result.llm_error = reason
        return result

    try:
        fields, missing = extract_with_llm(doc, schema, config.llm)
    except Exception as exc:  # noqa: BLE001 - any LLM failure triggers the fallback
        result = extract_with_rules(doc, schema)
        result.llm_error = f"LLM call failed: {exc}"
        return result

    return ExtractionResult(
        category=schema.category,
        fields=fields,
        missing=missing,
        method="llm",
        model=config.llm.model,
    )
