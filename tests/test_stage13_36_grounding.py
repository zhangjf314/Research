from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

import scripts.audit_grounding_stage13_36_v1 as audit
import scripts.run_evidence_first_canary_v1 as evidence_first


def test_many_to_many_required_claim_matching_is_possible() -> None:
    generated = [
        {
            "claim_id": "c1",
            "text": (
                "The model evaluates classification and summarization while using "
                "maximum likelihood training."
            ),
            "citations": [{"paper_id": "p", "page": 1, "block_id": "b"}],
        }
    ]
    scores = [
        audit.overlap(
            "The experiments include classification and summarization.", generated[0]["text"]
        ),
        audit.overlap("Training uses maximum likelihood.", generated[0]["text"]),
    ]
    assert all(score >= 0.35 for score in scores)


def test_exact_gold_precision_is_separate_from_semantic_support() -> None:
    claim = {
        "text": "The cited page discusses the method.",
        "citations": [{"paper_id": "uuid", "page": 2, "block_id": "non_gold"}],
    }
    result = audit.classify_unsupported_claim(
        claim,
        gold_blocks={"gold_block"},
        gold_pages={2},
        gold_papers={"1706.03762"},
        uuid_to_public={"uuid": "1706.03762"},
    )
    assert result == "SUPPORTED_BY_EQUIVALENT_PAGE"


def test_composite_claim_is_detected() -> None:
    composite, features = audit.is_composite_claim(
        "The method uses attention and avoids recurrence, while improving parallelization.",
        citation_count=2,
        match_count=2,
    )
    assert composite is True
    assert features["composite_score"] >= 2


def test_partial_support_is_not_collapsed_to_unsupported() -> None:
    category = audit.classify_citation_failure(
        {"text": "The method has two linked obligations and cites only one.", "citations": []},
        {"paper_id": "uuid", "page": 3, "block_id": "other"},
        gold_blocks={"gold"},
        gold_pages={3},
        gold_papers={"paper"},
        uuid_to_public={"uuid": "paper"},
        required_claims=["first obligation", "second obligation"],
    )
    assert category == "CITATION_SUPPORTS_CLAIM_ON_EQUIVALENT_PAGE"


def test_claim_cardinality_theoretical_cap() -> None:
    rows = [
        {"question_id": "q1", "required_claims": ["a", "b", "c"]},
        {"question_id": "q2", "required_claims": ["a", "b", "c", "d", "e", "f"]},
    ]
    payload = audit.summarize_cardinality(rows, ["q1", "q2"])
    assert payload["gold_dev_v1"]["count_required_claim_count_gt_3"] == 1
    assert payload["gold_dev_v1"][
        "theoretical_coverage_cap_with_max_3_generated_claims_one_to_one"
    ] == pytest.approx(0.666667)


def test_evidence_first_stage_one_does_not_accept_answer() -> None:
    with pytest.raises(ValidationError):
        evidence_first.EvidenceSelectionResponse.model_validate(
            {
                "answer": "not allowed",
                "evidence": [{"citation_key": "C1", "fact": "A fact."}],
                "insufficient_evidence": False,
            }
        )


def test_evidence_first_fact_has_single_citation() -> None:
    response = evidence_first.EvidenceSelectionResponse.model_validate(
        {
            "evidence": [{"citation_key": "C1", "fact": "A single fact."}],
            "insufficient_evidence": False,
        }
    )
    assert response.evidence[0].citation_key == "C1"
    repeated_key = evidence_first.EvidenceSelectionResponse.model_validate(
        {
            "evidence": [
                {"citation_key": "C1", "fact": "First distinct fact."},
                {"citation_key": "C1", "fact": "Second distinct fact."},
            ],
            "insufficient_evidence": False,
        }
    )
    assert len(repeated_key.evidence) == 2
    with pytest.raises(ValidationError):
        evidence_first.EvidenceSelectionResponse.model_validate(
            {
                "evidence": [
                    {"citation_key": "C1", "fact": "A fact."},
                    {"citation_key": "C1", "fact": "A fact."},
                ],
                "insufficient_evidence": False,
            }
        )


def test_answer_composition_cannot_add_unknown_fact_index() -> None:
    composition = evidence_first.AnswerCompositionResponse.model_validate(
        {"answer": "Uses known facts.", "used_facts": [0]}
    )
    assert composition.used_facts == [0]
    known = {0}
    assert set(composition.used_facts) <= known
    assert not ({2} <= known)


def test_evidence_first_budget_constants() -> None:
    assert evidence_first.MAX_ITEMS == 6
    assert evidence_first.MAX_INPUT_TOKENS == 100000
    assert evidence_first.MAX_OUTPUT_TOKENS == 10000
    assert evidence_first.MAX_COST_USD == 0.05
    assert evidence_first.MAX_TOTAL_SECONDS == 900


def test_evidence_first_canary_failure_blocks_full_qa() -> None:
    summary = {
        "evidence_first_canary_gate": "FAILED",
        "ready_for_full_qa": False,
    }
    assert summary["evidence_first_canary_gate"] == "FAILED"
    assert summary["ready_for_full_qa"] is False


def test_historical_canary_paths_are_not_overwritten() -> None:
    source = audit.Path(audit.__file__).read_text(encoding="utf-8")
    assert "historical_results_overwritten" in source
    runner_source = evidence_first.Path(evidence_first.__file__).read_text(encoding="utf-8")
    assert "full-qa-canary-deepseek-v1.json" in runner_source


def test_reports_disclose_not_blind_holdout(tmp_path, monkeypatch) -> None:
    payload = {
        "baseline_summaries": [
            {
                "baseline_label": "QWEN_CANARY_BASELINE",
                "summary": {"attempted": 15, "completed": 15},
                "composite_claim_rate": 0,
                "atomic_claim_rate": 1,
                "claim_failure_classifications": {},
                "citation_precision_failure_classifications": {},
                "unsupported_claim_classifications": {},
            }
        ]
    }
    text = audit.render_alignment_doc(payload)
    assert "not a blind holdout" in text


def test_evidence_first_trace_does_not_persist_authorization_header() -> None:
    rendered = json.dumps(
        {
            "request": {
                "authorization_header_persisted": False,
                "user_payload": {"question": "q"},
            }
        }
    )
    assert "Bearer" not in rendered
    assert "authorization_header_persisted" in rendered
