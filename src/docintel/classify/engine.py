"""Document classification: LLM-based (OpenAI) with a rule-based fallback."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from docintel.classify.config import ClassificationConfig, Config, LLMConfig
from docintel.models import Document

# Cap how much document text is sent to the LLM (chars).
MAX_LLM_CHARS = 12_000

Method = str  # "llm" | "rules"


@dataclass
class CategoryMatch:
    category: str
    score: float  # 0.0 - 1.0 confidence
    reasoning: str


@dataclass
class ClassificationResult:
    matches: list[CategoryMatch] = field(default_factory=list)  # ranked, best first
    method: Method = "rules"
    model: str | None = None
    note: str | None = None  # e.g. fallback warning
    llm_error: str | None = None

    @property
    def best(self) -> CategoryMatch | None:
        return self.matches[0] if self.matches else None

    @property
    def used_fallback(self) -> bool:
        return self.method == "rules"


class ClassificationError(RuntimeError):
    """Raised when classification cannot proceed at all."""


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------

def _count_phrase(text: str, phrase: str) -> int:
    """Count whole-word occurrences of a phrase (avoids 'cv' matching 'zxcvbnm')."""
    return len(re.findall(rf"\b{re.escape(phrase)}\b", text))


def _rule_scores(doc: Document, cfg: ClassificationConfig) -> list[CategoryMatch]:
    text = doc.text.lower()
    heading_text = " ".join(h.text.lower() for h in doc.headings)
    # First ~200 chars often contain the title/type of the document.
    head_text = text[:200]

    matches: list[CategoryMatch] = []
    for category in cfg.categories:
        if category == "other":
            continue
        score = 0.0
        hits: list[str] = []

        name_phrase = category.replace("_", " ")
        if _count_phrase(text, name_phrase):
            score += 2.0
            hits.append(name_phrase)
            if _count_phrase(heading_text, name_phrase) or _count_phrase(head_text, name_phrase):
                score += 2.0  # appears in title/headings: strong signal

        for phrase in cfg.keywords.get(category, ()):  # built-in hints
            phrase_l = phrase.lower()
            count = _count_phrase(text, phrase_l)
            if count:
                score += min(count, 5) * 0.5
                hits.append(phrase_l)

        if score > 0:
            matches.append(
                CategoryMatch(
                    category=category,
                    score=score,
                    reasoning=f"keyword matches: {', '.join(sorted(set(hits))[:6])}",
                )
            )

    # Normalize raw scores into 0..1 confidence relative to the best match.
    if matches:
        best = max(m.score for m in matches)
        for m in matches:
            m.score = round(m.score / best, 3) if best else 0.0
        matches.sort(key=lambda m: m.score, reverse=True)
    return matches


def classify_with_rules(doc: Document, cfg: ClassificationConfig, top_n: int) -> ClassificationResult:
    matches = _rule_scores(doc, cfg)[:top_n]
    if not matches:
        matches = [CategoryMatch(category="other", score=1.0, reasoning="no category keywords matched")]
    return ClassificationResult(
        matches=matches,
        method="rules",
        note=(
            "Rule-based fallback: no LLM was used (missing API key/model or LLM call failed). "
            "Scores are relative keyword-match confidence, not model probabilities."
        ),
    )


# ---------------------------------------------------------------------------
# LLM classification (OpenAI)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a document classification engine. Given a document's text and a list of "
    "allowed categories, rank the most likely categories. Respond ONLY with JSON."
)


def _build_user_prompt(doc: Document, cfg: ClassificationConfig, top_n: int) -> str:
    text = doc.text
    if len(text) > MAX_LLM_CHARS:
        text = text[:MAX_LLM_CHARS] + "\n...[truncated]"
    headings = "\n".join(f"{'#' * h.level} {h.text}" for h in doc.headings[:40])
    return (
        f"Document filename: {doc.path.name}\n"
        f"Document format: {doc.format.value}\n"
        f"Headings:\n{headings or '(none)'}\n\n"
        f"Document text:\n{text}\n\n"
        f"Allowed categories:\n{json.dumps(cfg.categories)}\n\n"
        f"Return JSON with this exact shape:\n"
        f'{{"matches": [{{"category": "<one of the allowed categories>", '
        f'"score": <float 0-1 confidence>, "reasoning": "<one short sentence>"}}]}}\n'
        f"Return the top {top_n} categories, best first. Scores must sum to at most 1."
    )


def _llm_available(llm: LLMConfig) -> tuple[bool, str | None]:
    if llm.provider != "openai":
        return False, f"unsupported provider: {llm.provider!r} (only 'openai' is supported)"
    if not llm.model:
        return False, "no model configured (set llm.model in config YAML or DOCINTEL_MODEL env var)"
    if not os.environ.get(llm.api_key_env):
        return False, f"API key env var {llm.api_key_env} is not set"
    return True, None


def _parse_llm_json(content: str, allowed: list[str], top_n: int) -> list[CategoryMatch]:
    # Tolerate code fences around the JSON payload.
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", content)
    data = json.loads(content)
    raw_matches = data.get("matches", [])
    if not isinstance(raw_matches, list):
        raise ValueError("'matches' must be a list")
    allowed_set = set(allowed)
    matches: list[CategoryMatch] = []
    for item in raw_matches[:top_n]:
        category = str(item.get("category", "")).strip()
        if category not in allowed_set:
            continue  # ignore hallucinated categories
        score = float(item.get("score", 0.0))
        matches.append(
            CategoryMatch(
                category=category,
                score=round(max(0.0, min(1.0, score)), 3),
                reasoning=str(item.get("reasoning", "")).strip(),
            )
        )
    if not matches:
        raise ValueError("LLM returned no valid categories")
    matches.sort(key=lambda m: m.score, reverse=True)
    return matches


def classify_with_llm(doc: Document, cfg: ClassificationConfig, llm: LLMConfig) -> list[CategoryMatch]:
    """Call OpenAI to classify the document. Raises on any failure."""
    from openai import OpenAI  # deferred import so the CLI works without the SDK installed

    client = OpenAI(base_url=llm.base_url) if llm.base_url else OpenAI()
    response = client.chat.completions.create(
        model=llm.model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(doc, cfg, cfg.top_n)},
        ],
    )
    content = response.choices[0].message.content or ""
    return _parse_llm_json(content, cfg.categories, cfg.top_n)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def classify_document(doc: Document, config: Config) -> ClassificationResult:
    """Classify a document using the LLM when available, else fall back to rules.

    The fallback is always clearly flagged in the result (``method='rules'``,
    ``note`` and ``llm_error`` when the LLM was attempted but failed).
    """
    cfg = config.classification
    if not doc.text.strip():
        return ClassificationResult(
            matches=[CategoryMatch(category="other", score=1.0, reasoning="document has no text content")],
            method="rules",
            note="Document is empty; nothing to classify.",
        )

    available, reason = _llm_available(config.llm)
    if not available:
        result = classify_with_rules(doc, cfg, cfg.top_n)
        result.llm_error = reason
        return result

    try:
        matches = classify_with_llm(doc, cfg, config.llm)
    except Exception as exc:  # noqa: BLE001 - any LLM failure triggers the fallback
        result = classify_with_rules(doc, cfg, cfg.top_n)
        result.llm_error = f"LLM call failed: {exc}"
        return result

    return ClassificationResult(matches=matches, method="llm", model=config.llm.model)
