"""RQ12 §5: claim / obligation-level retrieval metrics.

A "claim" here is an evidence obligation with its own supporting gold set.
One gold hit must not be counted as solving a multi-evidence question.
"""

from __future__ import annotations

from collections.abc import Sequence

CLAIM_METRICS_VERSION = "rag-v2-claim-metrics-v1"
METRIC_NOT_COMPUTABLE = "METRIC_NOT_COMPUTABLE"


def claim_covered(ranked: Sequence[str], claim_gold: set[str], k: int) -> bool:
    if not claim_gold:
        return False
    return bool(set(ranked[:k]) & claim_gold)


def required_claim_coverage_at_k(
    ranked: Sequence[str],
    claim_golds: Sequence[set[str]],
    k: int,
) -> float:
    """Fraction of required claims with >=1 supporting gold block in top-k."""

    if not claim_golds:
        return 1.0
    covered = sum(claim_covered(ranked, gold, k) for gold in claim_golds)
    return covered / len(claim_golds)


def all_claims_covered_at_k(
    ranked: Sequence[str],
    claim_golds: Sequence[set[str]],
    k: int,
) -> bool:
    if not claim_golds:
        return True
    return all(claim_covered(ranked, gold, k) for gold in claim_golds)


def evidence_obligation_recall_at_k(
    ranked: Sequence[str],
    claim_golds: Sequence[set[str]],
    k: int,
) -> float:
    """Mean over obligations of |top-k ∩ obligation gold| / |obligation gold|."""

    if not claim_golds:
        return 1.0
    scores = []
    for gold in claim_golds:
        if not gold:
            scores.append(0.0)
            continue
        scores.append(len(set(ranked[:k]) & gold) / len(gold))
    return sum(scores) / len(scores)


def multi_evidence_complete_rate_at_k(
    rankings: Sequence[Sequence[str]],
    claim_golds_list: Sequence[Sequence[set[str]]],
    k: int,
) -> float:
    """Fraction of multi-claim questions whose claims are ALL covered in top-k."""

    rows = [
        (ranked, golds)
        for ranked, golds in zip(rankings, claim_golds_list, strict=True)
        if len(golds) > 1
    ]
    if not rows:
        return METRIC_NOT_COMPUTABLE
    return sum(
        all_claims_covered_at_k(ranked, golds, k) for ranked, golds in rows
    ) / len(rows)


def mean_claim_coverage_at_k(
    rankings: Sequence[Sequence[str]],
    claim_golds_list: Sequence[Sequence[set[str]]],
    k: int,
) -> float:
    if not rankings:
        return METRIC_NOT_COMPUTABLE
    return sum(
        required_claim_coverage_at_k(ranked, golds, k)
        for ranked, golds in zip(rankings, claim_golds_list, strict=True)
    ) / len(rankings)
