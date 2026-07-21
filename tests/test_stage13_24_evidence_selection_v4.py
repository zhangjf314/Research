from __future__ import annotations

import importlib.util
from pathlib import Path

from paper_research.generation.citation_selection import CitationCandidate
from paper_research.generation.evidence_selection_v4 import (
    CANDIDATE_BUDGET,
    CITATION_BUDGET,
    CandidateRole,
    evaluate_candidate_admissibility,
    evaluate_role_eligibility,
    evaluate_set_sufficiency,
    prove_replacement,
    select_evidence_v4,
)


def cand(
    citation_id: str,
    text: str,
    *,
    score: float = 1.0,
    original: bool = False,
    adjacent: bool = False,
) -> CitationCandidate:
    return CitationCandidate(
        citation_id=citation_id,
        paper_id="paper",
        page=1,
        block_id=citation_id,
        text=text,
        original_selected=original,
        adjacent_completion=adjacent,
        retrieval_score=score,
    )


def test_candidate_admissibility_admits_partial_obligation_contribution() -> None:
    candidate = cand("E1", "The model uses a bidirectional Transformer encoder.")
    result = evaluate_candidate_admissibility(
        "The model uses a bidirectional Transformer encoder for representation.",
        candidate,
    )
    assert result.admissible
    assert result.covered_obligations


def test_candidate_admissibility_rejects_no_obligation_contribution() -> None:
    result = evaluate_candidate_admissibility(
        "The optimizer uses Adam with warmup.",
        cand("E1", "The paper discusses unrelated datasets and background."),
    )
    assert not result.admissible
    assert "no_obligation_contribution" in result.hard_fail_reasons


def test_numeric_partial_role_is_support_not_primary() -> None:
    claim = "Dataset size is varied from 22M to 23B tokens."
    candidate = cand("E1", "The experiment varies dataset size D in tokens.")
    admissibility = evaluate_candidate_admissibility(claim, candidate)
    role = evaluate_role_eligibility(claim, candidate, admissibility)
    assert admissibility.admissible
    assert role.role is CandidateRole.COMPLEMENTARY_SUPPORT


def test_comparison_side_role_is_side_specific() -> None:
    claim = "The first model uses recurrence while the second uses attention."
    candidate = cand("E1", "The second model relies on attention.")
    admissibility = evaluate_candidate_admissibility(claim, candidate)
    role = evaluate_role_eligibility(claim, candidate, admissibility)
    assert admissibility.admissible
    assert role.role in {
        CandidateRole.SIDE_SPECIFIC_PRIMARY,
        CandidateRole.COMPLEMENTARY_SUPPORT,
    }


def test_set_sufficiency_accepts_two_complementary_candidates() -> None:
    claim = "The first model uses recurrence while the second uses attention."
    selected = (
        cand("E1", "The first model uses recurrent networks."),
        cand("E2", "The second model uses attention mechanisms."),
    )
    result = evaluate_set_sufficiency(claim, selected)
    assert result.comparison_complete


def test_baseline_first_retains_exact_baseline_and_rejects_no_gain_replacement() -> None:
    claim = "The Transformer eschews recurrence."
    baseline = cand(
        "E1",
        "The Transformer eschews recurrence and instead relies on attention.",
        original=True,
    )
    weaker = cand("E2", "The Transformer is a neural sequence model.", score=2.0)
    result = select_evidence_v4(claim, (baseline, weaker), ("E1",))
    assert result.primary_citation_ids == ("E1",)
    assert not result.baseline_replaced


def test_partial_baseline_can_receive_direct_complement() -> None:
    claim = "BERT is a bidirectional Transformer-based language representation model."
    baseline = cand("E1", "BERT uses masked language modeling.", original=True, score=0.1)
    complement = cand(
        "E2",
        "BERT stands for Bidirectional Encoder Representations from Transformers.",
        score=1.0,
    )
    result = select_evidence_v4(claim, (baseline, complement), ("E1",))
    assert result.primary_citation_ids == ("E1",)
    assert "E2" in result.supporting_citation_ids
    assert result.baseline_added_to


def test_empty_baseline_does_not_create_unsupported_new_set() -> None:
    claim = "Training uses eight P100 GPUs and an Adam schedule with warmup."
    weak = cand("E1", "Training used eight GPUs for 3.5 days.", original=True)
    result = select_evidence_v4(claim, (weak,), ())
    assert result.primary_citation_ids == ()
    assert result.supporting_citation_ids == ()


def test_replacement_proof_rejects_score_only_replacement() -> None:
    claim = "The Transformer eschews recurrence."
    baseline = (cand("E1", "The Transformer eschews recurrence."),)
    proposed = (cand("E2", "The Transformer is a popular model.", score=10.0),)
    proof = prove_replacement(claim, baseline, proposed)
    assert not proof.passed


def test_fixed_candidate_and_citation_budgets() -> None:
    assert CANDIDATE_BUDGET == 12
    assert CITATION_BUDGET == 3


def test_v4_module_has_no_forbidden_online_features() -> None:
    source = Path("src/paper_research/generation/evidence_selection_v4.py").read_text(
        encoding="utf-8"
    )
    lowered = source.lower()
    assert "question_id" not in lowered
    assert "required_claim_id" not in lowered
    assert "human_label" not in lowered
    assert "root_cause" not in lowered
    assert "retrieval_gold" not in lowered


def test_leakage_audit_script_reports_passed() -> None:
    path = Path("scripts/audit_evidence_selection_v4_feature_leakage.py")
    spec = importlib.util.spec_from_file_location("audit_v4_leakage", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    body = module.build()
    assert body["gate"] == "PASSED"
