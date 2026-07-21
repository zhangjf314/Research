"""Stage 13.23 precision-constrained selection safeguards."""

from __future__ import annotations

import json

from paper_research.generation.citation_selection import CitationCandidate
from paper_research.generation.evidence_selection_v3 import (
    CANDIDATE_BUDGET,
    EVIDENCE_SELECTION_V3_VERSION,
    evaluate_evidence_eligibility,
    select_evidence_v3,
)
from scripts.audit_evidence_selection_v2_wrong_evidence_v1 import OUT_JSON
from scripts.audit_evidence_selection_v2_wrong_evidence_v1 import main as wrong_main
from scripts.audit_evidence_selection_v3_feature_leakage import audit as leakage_audit
from scripts.replay_dev_v3_6_evidence_selection_v3 import build as build_replay


def candidate(cid: str, text: str, score: float = 1.0) -> CitationCandidate:
    return CitationCandidate(
        citation_id=cid,
        paper_id="p",
        page=1,
        block_id=cid,
        text=text,
        retrieval_score=score,
        original_selected=True,
        token_cost=20,
    )


def test_numeric_missing_endpoint_is_hard_veto() -> None:
    result = evaluate_evidence_eligibility(
        "The model uses a 512 to 2048 hidden dimension range.",
        candidate("E1", "The model uses a hidden dimension of 512."),
    )

    assert result.eligible is False
    assert "numeric_anchor_missing_or_conflicting" in result.hard_fail_reasons


def test_lexical_only_false_positive_is_ineligible() -> None:
    result = evaluate_evidence_eligibility(
        "The method improves ROUGE by using coordinate ascent.",
        candidate("E1", "This survey mentions ROUGE and methods in general."),
    )

    assert result.eligible is False
    assert result.hard_fail_reasons


def test_baseline_protection_retains_eligible_baseline() -> None:
    baseline = candidate(
        "E1",
        "The method improves ROUGE by using coordinate ascent in the reranking stage.",
    )
    challenger = candidate("E2", "The paper reports a method and ROUGE.", score=2.0)

    result = select_evidence_v3(
        "The method improves ROUGE by using coordinate ascent.",
        (challenger, baseline),
        ("E1",),
    )

    assert result.version == EVIDENCE_SELECTION_V3_VERSION
    assert result.baseline_retained is True
    assert result.primary_citation_ids == ("E1",)


def test_candidate_budget_is_fixed() -> None:
    candidates = tuple(candidate(f"E{i:02d}", f"evidence method claim {i}") for i in range(20))

    result = select_evidence_v3("evidence method claim", candidates)

    assert len(result.eligibility_results) == CANDIDATE_BUDGET


def test_wrong_evidence_audit_has_no_unknowns() -> None:
    wrong_main()
    audit = json.loads(OUT_JSON.read_text(encoding="utf-8"))

    assert audit["records"] == 15
    assert audit["unknown"] == 0


def test_selection_v3_replay_fails_closed_before_live() -> None:
    replay = build_replay()

    assert replay["EVIDENCE_SELECTION_V3_ENGINEERING_GATE"] == "PASSED"
    assert replay["EVIDENCE_SELECTION_V3_QUALITY_PREFLIGHT"] == "FAILED"
    assert replay["NEXT_LIVE_READY"] is False
    assert replay["NEXT_LIVE_AUTHORIZED"] is False


def test_selection_v3_feature_leakage_gate_passes() -> None:
    audit = leakage_audit()

    assert audit["gate"] == "PASSED"
    assert audit["gold_online_leakage"] == 0
    assert audit["human_label_online_leakage"] == 0
    assert audit["fixed_id_special_cases"] == 0
