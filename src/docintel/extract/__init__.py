"""Key-field extraction: LLM-based with rule-based fallback."""

from docintel.extract.engine import (
    ExtractedField,
    ExtractionResult,
    extract_fields,
    extract_with_rules,
)
from docintel.extract.schemas import FieldSpec, Schema, schema_for

__all__ = [
    "ExtractedField",
    "ExtractionResult",
    "FieldSpec",
    "Schema",
    "extract_fields",
    "extract_with_rules",
    "schema_for",
]
