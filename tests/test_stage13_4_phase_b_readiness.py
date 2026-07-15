# ruff: noqa: E501
from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_research.generation.required_claim_output import (
    RequiredClaimValidationError,
    parse_and_validate_required_claim_response,
    required_claim_output_token_budget,
)
from scripts.evidence_qa_dev_lib_v1 import DEV_IDS, read_jsonl
from scripts.evidence_qa_dev_v3_lib import (
    FIXTURE_SUMMARY,
    MANIFEST,
    SOURCE_MANIFEST_HASH,
    build_required_claim_input,
)
from scripts.import_stage13_4_human_review_v1 import (
    CANDIDATES,
)
from scripts.review_evidence_qa_dev_v2_citations_v1 import AUDIT

DATA = Path("data/evaluation")
def test_human_review_import_57_and_4_is_immutable() -> None:
    reviewed = read_jsonl(AUDIT)
    imported = read_jsonl(DATA / "dev-v2-claim-coverage-audit-v1.jsonl")
    assert len(reviewed) == 57 and len(CANDIDATES) == 4 and len(imported) == 27
    assert all(row["human_review_status"] == "approved" and row["reviewer"] and row["review_notes"] for row in reviewed)
    by_id = {row["required_claim_id"]: row for row in imported}
    assert all(by_id[claim_id]["matcher_human_decision"] == decision for claim_id, (decision, _credit, _partial) in CANDIDATES.items())


def test_citation_summary_exact_rates_and_strata() -> None:
    payload = json.loads((DATA / "evidence-qa-dev-v2-citation-audit-summary-v1.json").read_text(encoding="utf-8"))
    assert payload["labels"]["fully_supported"] == 35
    assert payload["labels"]["partially_supported"] == 17
    assert payload["strict_support_rate"] == 0.614035
    assert payload["lenient_support_rate"] == 0.912281
    assert payload["strata"]["evidence_source"]["original_selected"]["n"] == 43
    assert payload["strata"]["evidence_source"]["adjacent_completion"]["n"] == 14
    assert payload["q019_compound_claim"]["single_citation_partially_supports_compound_claim"] == 9
    assert payload["gold_annotation_too_narrow_present"] is False
    assert payload["malformed_evidence_present"] is False


def test_matcher_adjudication_preserves_historical_metric() -> None:
    payload = json.loads((DATA / "dev-v2-claim-coverage-human-adjudication-v1.json").read_text(encoding="utf-8"))
    assert payload["historical_formal_dev_v2"] == {"covered": 14, "required": 27, "rate": 0.518519, "historical_metric_modified": False}
    assert payload["human_adjudicated_diagnostic"]["covered"] == 16
    decisions = payload["matcher_candidates"]
    assert decisions["cl-q001-c41ea3191cab92907d83"]["formal_coverage_credit"] == 1
    assert decisions["cl-q002-3cab1a55474dcd47a64a"]["formal_coverage_credit"] == 1
    assert decisions["cl-q004-ba317f113c67f72f7260"]["formal_coverage_credit"] == 0
    assert decisions["cl-q004-ba317f113c67f72f7260"]["diagnostic_partial_credit"] == 0.5
    assert decisions["cl-q015-281a83daa0567f6ae7ac"]["matcher_human_decision"] == "false_positive"


def test_dev_v3_manifest_and_quality_gates_are_frozen() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["manifest_hash"] == SOURCE_MANIFEST_HASH
    assert manifest["question_ids"] == DEV_IDS
    assert manifest["quality_candidate_gate_frozen"]["required_claim_coverage_strictly_greater_than"] == 0.592593
    assert manifest["configuration"]["reranker_enabled"] is False
    assert manifest["configuration"]["retries"] == 0


def test_dev_v3_inputs_allocate_without_gold_or_human_evidence() -> None:
    for question_id in DEV_IDS:
        payload, registry, _contexts, _trace = build_required_claim_input(question_id)
        assert payload["gold_evidence_used_for_allocation"] is False
        assert payload["oracle_used_for_allocation"] is False
        assert payload["human_pilot_used_for_allocation"] is False
        assert registry.registry_hash
        assert len(payload["required_claims"]) == len({row["required_claim_id"] for row in payload["required_claims"]})


def test_fixture_runs_are_isolated_complete_and_no_live_calls() -> None:
    fixture = json.loads(FIXTURE_SUMMARY.read_text(encoding="utf-8"))
    assert fixture["fixture_count"] == fixture["fixture_passed"] == 15
    assert fixture["silent_omission_rate"] == fixture["unknown_citation_id_rate"] == fixture["cross_claim_citation_rate"] == 0
    assert fixture["live_llm_called"] is False
    assert fixture["question_input_count"] == 10
    assert all(item["passed"] for item in fixture["fixtures"])


def test_dev_v3_live_requires_explicit_authorization(monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.run_evidence_qa_dev_v3 as runner

    monkeypatch.delenv("DEV_V3_LIVE_AUTHORIZED", raising=False)
    with pytest.raises(RuntimeError, match="DEV_V3_LIVE_NOT_AUTHORIZED"):
        runner.assert_live_authorized()


def test_error_codes_and_budget_are_auditable() -> None:
    payload, registry, _, _ = build_required_claim_input("q001")
    with pytest.raises(RequiredClaimValidationError) as failure:
        parse_and_validate_required_claim_response("{", expected_claim_ids=[row["required_claim_id"] for row in payload["required_claims"]], registry=registry, allowed_by_claim={}, expected_registry_hash=registry.registry_hash)
    assert failure.value.code == "malformed_json"
    assert required_claim_output_token_budget(0).model_dump() == {"required_claim_count": 0, "calculated_max_output_tokens": 256, "capped": False, "budget_formula_version": "required-claim-output-budget-v1"}
    assert required_claim_output_token_budget(30).capped is True


def test_readiness_is_true_but_authorization_false() -> None:
    payload = json.loads((DATA / "evidence-qa-dev-v3-readiness-v1.json").read_text(encoding="utf-8"))
    assert payload["ready_for_dev_v3"] is True
    assert payload["dev_v3_authorized"] is False
    assert payload["dev_v3_live_run"] is False
    assert payload["full_qa"] == payload["deep_research"] == "blocked"
