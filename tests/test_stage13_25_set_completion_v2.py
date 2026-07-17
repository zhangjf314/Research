from __future__ import annotations

import importlib.util
from pathlib import Path

from paper_research.generation.citation_selection import CitationCandidate
from paper_research.generation.claim_obligations import build_claim_obligation_set
from paper_research.generation.set_completion_v2 import (
    CANDIDATE_BUDGET,
    SET_COMPLETION_V2_VERSION,
    directness_score,
    evaluate_set_coverage_v2,
    select_set_completion_v2,
)


def cand(cid: str, text: str, *, score: float = 1.0) -> CitationCandidate:
    return CitationCandidate(
        citation_id=cid,
        paper_id="p",
        page=1,
        block_id=cid,
        text=text,
        retrieval_score=score,
    )


def test_canonical_obligation_set_is_stable() -> None:
    first = build_claim_obligation_set("The model uses Adam and warmup.")
    second = build_claim_obligation_set("The model uses Adam and warmup.")
    assert first.deterministic_hash == second.deterministic_hash
    assert len({ob.obligation_id for ob in first.obligations}) == len(first.obligations)


def test_numeric_obligation_stable() -> None:
    obligation_set = build_claim_obligation_set("Dataset size varies from 22M to 23B tokens.")
    assert any(ob.numeric_anchors for ob in obligation_set.obligations)


def test_set_completion_keeps_complete_baseline_untouched() -> None:
    claim = "The Transformer eschews recurrence."
    baseline = cand("E1", "The Transformer eschews recurrence and relies on attention.")
    noisy = cand("E2", "The paper discusses training data.")
    result = select_set_completion_v2(claim, (baseline, noisy), ("E1",))
    assert result.primary_citation_ids == ("E1",)
    assert result.supporting_citation_ids == ()


def test_partial_baseline_receives_complement() -> None:
    claim = "The first model uses recurrence while the second uses attention."
    baseline = cand("E1", "The first model uses recurrence.", score=0.8)
    complement = cand(
        "E2",
        "The second model uses attention in contrast to recurrent models.",
        score=1.0,
    )
    result = select_set_completion_v2(claim, (baseline, complement), ("E1",))
    assert result.primary_citation_ids == ("E1",)
    assert "E2" in result.supporting_citation_ids


def test_no_gain_complement_rejected() -> None:
    claim = "The Transformer eschews recurrence."
    baseline = cand("E1", "The Transformer eschews recurrence.")
    duplicate = cand("E2", "The Transformer eschews recurrence.")
    result = select_set_completion_v2(claim, (baseline, duplicate), ("E1",))
    assert "E2" not in result.supporting_citation_ids


def test_range_endpoint_missing_is_incomplete() -> None:
    claim = "Dataset size varies from 22M to 23B tokens."
    obligation_set = build_claim_obligation_set(claim)
    coverage = evaluate_set_coverage_v2(
        claim,
        obligation_set,
        (cand("E1", "Dataset size is 22M tokens."),),
    )
    assert coverage.numeric_applicable
    assert not coverage.numeric_complete


def test_non_numeric_claim_not_in_numeric_denominator() -> None:
    claim = "The model uses attention."
    obligation_set = build_claim_obligation_set(claim)
    coverage = evaluate_set_coverage_v2(
        claim,
        obligation_set,
        (cand("E1", "The model uses attention."),),
    )
    assert not coverage.numeric_applicable
    assert coverage.numeric_complete


def test_comparison_left_only_is_incomplete_when_comparison_detected() -> None:
    claim = "The first model uses recurrence while the second uses attention."
    obligation_set = build_claim_obligation_set(claim)
    coverage = evaluate_set_coverage_v2(
        claim,
        obligation_set,
        (cand("E1", "The first model uses recurrence."),),
    )
    assert coverage.comparison_applicable
    assert not coverage.comparison_complete


def test_candidate_budget_and_version() -> None:
    assert CANDIDATE_BUDGET == 12
    assert SET_COMPLETION_V2_VERSION == "set-completion-v2-candidate"


def test_directness_score_prefers_phrase_match() -> None:
    claim = "bidirectional Transformer encoder representation"
    better = directness_score(
        claim,
        (cand("E1", "bidirectional Transformer encoder representation"),),
    )
    weaker = directness_score(claim, (cand("E2", "Transformer model representation"),))
    assert better > weaker


def test_no_forbidden_runtime_features_in_set_completion_modules() -> None:
    for path in [
        Path("src/paper_research/generation/claim_obligations.py"),
        Path("src/paper_research/generation/set_completion_v2.py"),
    ]:
        lowered = path.read_text(encoding="utf-8").lower()
        assert "question_id" not in lowered
        assert "required_claim_id" not in lowered
        assert "human_label" not in lowered
        assert "oracle" not in lowered


def test_set_completion_leakage_audit_passes() -> None:
    path = Path("scripts/audit_set_completion_v2_feature_leakage.py")
    spec = importlib.util.spec_from_file_location("audit_set_completion_leakage", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    body = module.build()
    assert body["gate"] == "PASSED"
