"""Document classification: LLM-based with rule-based fallback."""

from docintel.classify.config import Config, ConfigError, load_config
from docintel.classify.engine import (
    CategoryMatch,
    ClassificationResult,
    classify_document,
)

__all__ = [
    "CategoryMatch",
    "ClassificationResult",
    "Config",
    "ConfigError",
    "classify_document",
    "load_config",
]
