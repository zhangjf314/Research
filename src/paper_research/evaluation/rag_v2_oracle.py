"""RQ13 §4: gold-informed oracle Top-5 construction — EVALUATION ONLY.

This module MUST NOT be imported by runtime retrieval/selection code; it
exists solely to estimate the theoretical post-retrieval headroom of a
candidate pool using development gold (upper-bound estimator).
"""

from __future__ import annotations

from collections.abc import Sequence

ORACLE_VERSION = "rag-v2-oracle-v1"


def oracle_top_k(
    pool: Sequence[str],
    claim_golds: Sequence[set[str]],
    *,
    top_k: int = 5,
) -> list[str]:
    """Greedily build the best Top-k set from `pool` using gold.

    Each slot picks the pool candidate maximizing, lexicographically:
    (1) number of not-yet-covered claims it would cover,
    (2) gold-block membership (for claim-free / leftover slots),
    (3) original pool rank (stable tie-break).
    Greedy coverage is near-optimal for this submodular objective; it is an
    upper-bound ESTIMATOR, not a runtime algorithm.
    """

    deduped = list(dict.fromkeys(pool))
    gold_all: set[str] = set().union(*claim_golds) if claim_golds else set()
    covered: set[int] = set()
    selected: list[str] = []
    while len(selected) < min(top_k, len(deduped)):
        best_doc, best_key = None, None
        for rank, doc_id in enumerate(deduped):
            if doc_id in selected:
                continue
            new_claims = sum(
                1
                for index, gold in enumerate(claim_golds)
                if index not in covered and doc_id in gold
            )
            key = (-new_claims, 0 if doc_id in gold_all else 1, rank)
            if best_key is None or key < best_key:
                best_doc, best_key = doc_id, key
        if best_doc is None:
            break
        selected.append(best_doc)
        covered.update(
            index
            for index, gold in enumerate(claim_golds)
            if best_doc in gold
        )
    return selected
