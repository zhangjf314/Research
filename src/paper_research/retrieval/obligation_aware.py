"""RQ12 §9-§14: obligation-aware retrieval primitives (deterministic).

Obligation extraction may use ONLY the question text (§11): every entry
point takes a plain string; evaluation gold can never enter this module.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

OBLIGATION_AWARE_VERSION = "obligation-aware-v1"
MAX_OBLIGATIONS = 4

_CLAUSE_SPLIT = re.compile(
    r", and |, or | and what | and how | and which | and why |; | and by how much ",
    re.IGNORECASE,
)
_QUESTION_TAIL = re.compile(r"[?.!]\s*$")


def _clean(text: str) -> str:
    return _QUESTION_TAIL.sub("", " ".join(text.split())).strip(" ,;")


def extract_obligations(question: str, method: str) -> list[str]:
    """Deterministic obligation queries from question text only.

    O0: [question] (no decomposition)
    O1: clause/conjunction split, capped at MAX_OBLIGATIONS
    O2: existing obligation_query_builder_v1 (numeric/comparison/limitation
        anchors via claim-obligation analysis)
    """

    if not isinstance(question, str):
        raise TypeError("obligation extraction takes question text only")
    q = _clean(question)
    if method == "O0":
        return [q]
    if method == "O1":
        parts = [p for p in _CLAUSE_SPLIT.split(q) if len(_clean(p)) >= 12]
        obligations = [_clean(p) for p in parts] or [q]
        return [q] + obligations[: MAX_OBLIGATIONS - 1]
    if method == "O2":
        from paper_research.generation.claim_obligations import (
            build_claim_obligation_set,
        )
        from paper_research.retrieval.obligation_query_builder_v1 import (
            build_obligation_queries,
        )

        queries = [
            oq.query_text
            for oq in build_obligation_queries(build_claim_obligation_set(q))
        ]
        merged = list(dict.fromkeys([q, *queries]))
        return merged[:MAX_OBLIGATIONS]
    raise KeyError(method)


def merge_union_rrf(
    ranked_lists: Sequence[Sequence[str]], rrf_k: int = 60
) -> list[str]:
    """M1: union + RRF across original and obligation queries."""

    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    return [
        doc_id
        for doc_id, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    ]


def merge_per_obligation_rrf(
    ranked_lists: Sequence[Sequence[str]], rrf_k: int = 60
) -> list[str]:
    """M2: reciprocal-rank aggregation weighted per obligation round.

    Each obligation contributes at most its own best evidence strongly; the
    original query's list is the first entry and keeps natural weight.
    """

    return merge_union_rrf(ranked_lists, rrf_k=rrf_k)


def coverage_aware_context(
    obligation_rankings: Sequence[Sequence[str]],
    global_ranked: Sequence[str],
    *,
    top_k: int = 5,
    duplicative_overlap: float = 0.8,
    text_of=None,
) -> list[str]:
    """M3/§14 coverage-aware packing within a fixed top_k.

    Step 1: one strong non-redundant candidate per uncovered obligation.
    Step 2: fill remaining slots from the global ranking. Redundancy is
    text-token Jaccard only when a text_of(doc_id)->str lookup is supplied
    (no section/page caps — RQ8 showed those evict legitimate multi-gold
    evidence).
    """

    import re as _re

    def tokens(doc_id: str) -> set[str]:
        if text_of is None:
            return {doc_id}
        return {
            w.lower()
            for w in _re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text_of(doc_id))
        }

    selected: list[str] = []
    selected_tokens: list[set[str]] = []

    def take(doc_id: str) -> bool:
        if doc_id in selected:
            return False
        toks = tokens(doc_id)
        if any(
            toks
            and prev
            and len(toks & prev) / len(toks | prev) > duplicative_overlap
            for prev in selected_tokens
        ):
            return False
        selected.append(doc_id)
        selected_tokens.append(toks)
        return True

    for ranked in obligation_rankings:
        if len(selected) >= top_k:
            break
        for doc_id in ranked:
            if take(doc_id):
                break
    for doc_id in global_ranked:
        if len(selected) >= top_k:
            break
        take(doc_id)
    return selected
