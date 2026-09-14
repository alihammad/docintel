"""RAG question answering: retrieve top-k chunks, then answer with citations.

LLM-only by design — there is no meaningful rule-based fallback for open-ended
questions. When no LLM is configured the result carries a clear error, and the
retrieved sources are still returned so the CLI can show them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from docintel.search.engine import SearchHit, search

_SYSTEM_PROMPT = (
    "You are a document question-answering engine. Answer the user's question using ONLY "
    "the numbered sources provided. Cite the sources you used with bracketed numbers like "
    "[1] or [2][3] immediately after the relevant statement. If the sources do not contain "
    "the answer, say so explicitly — never invent information."
)


@dataclass
class Citation:
    number: int
    doc_path: str
    heading_path: str
    snippet: str


@dataclass
class AskResult:
    query: str
    answer: str | None = None
    citations: list[Citation] = field(default_factory=list)
    method: str = "llm"  # "llm" | "none"
    model: str | None = None
    retrieval_method: str = "local-fallback"  # how sources were retrieved
    note: str | None = None
    llm_error: str | None = None

    @property
    def used_fallback_retrieval(self) -> bool:
        return self.retrieval_method == "local-fallback"


def _build_user_prompt(query: str, hits: list[SearchHit]) -> str:
    sources = []
    for i, hit in enumerate(hits, start=1):
        location = f"{hit.doc_path}" + (f" — {hit.heading_path}" if hit.heading_path else "")
        text = hit.full_text or hit.snippet
        sources.append(f"[{i}] ({location})\n{text}")
    return (
        f"Question: {query}\n\nSources:\n" + "\n\n".join(sources) +
        "\n\nAnswer the question, citing sources as [n]."
    )


def ask(query: str, config, top_k: int | None = None) -> AskResult:
    """Retrieve relevant chunks and answer the question with citations."""
    from docintel.classify.engine import _llm_available

    retrieval = search(query, config, top_k=top_k)
    if retrieval.error and not retrieval.hits:
        return AskResult(
            query=query,
            method="none",
            retrieval_method=retrieval.method,
            llm_error=retrieval.error,
        )
    if not retrieval.hits:
        return AskResult(
            query=query,
            method="none",
            retrieval_method=retrieval.method,
            note=retrieval.note or "No matching content found in the index.",
        )

    citations = [
        Citation(
            number=i,
            doc_path=hit.doc_path,
            heading_path=hit.heading_path,
            snippet=hit.snippet,
        )
        for i, hit in enumerate(retrieval.hits, start=1)
    ]

    available, reason = _llm_available(config.llm)
    if not available:
        return AskResult(
            query=query,
            method="none",
            citations=citations,
            retrieval_method=retrieval.method,
            llm_error=reason,
            note=(
                "LLM not configured — showing retrieved sources only. "
                "Set llm.model and the API key env var to get answers."
            ),
        )

    # Full chunk text is used for grounding (snippets are display-truncated).
    prompt = _build_user_prompt(query, retrieval.hits)

    try:
        from openai import OpenAI

        client = OpenAI(base_url=config.llm.base_url) if config.llm.base_url else OpenAI()
        response = client.chat.completions.create(
            model=config.llm.model,
            temperature=0,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        answer = (response.choices[0].message.content or "").strip()
    except Exception as exc:  # noqa: BLE001 - any LLM failure is reported, sources kept
        return AskResult(
            query=query,
            method="none",
            citations=citations,
            retrieval_method=retrieval.method,
            llm_error=f"LLM call failed: {exc}",
        )

    if not answer:
        return AskResult(
            query=query,
            method="none",
            citations=citations,
            retrieval_method=retrieval.method,
            llm_error="LLM returned an empty answer",
        )

    # Keep only citations actually referenced in the answer.
    referenced = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    if referenced:
        citations = [c for c in citations if c.number in referenced]

    return AskResult(
        query=query,
        answer=answer,
        citations=citations,
        method="llm",
        model=config.llm.model,
        retrieval_method=retrieval.method,
    )
