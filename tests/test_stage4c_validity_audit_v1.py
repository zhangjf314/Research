from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import audit_stage4c_benchmark_validity_v1 as audit


def _json(path: Path) -> dict:
    if not path.exists():
        pytest.skip(f"{path} has not been generated")
    return json.loads(path.read_text(encoding="utf-8"))


def test_metric_provenance_classifies_coverage_as_proxy() -> None:
    payload = _json(audit.OUT_PROVENANCE)

    assert payload["content_level_rubric_validated"] is False
    assert payload["structured_proxy_metrics_valid"] is True
    assert payload["metrics"]["required_dimension_coverage"]["classification"] == "STRUCTURAL_PROXY"
    assert payload["metrics"]["required_claim_coverage"]["claims_individually_scored"] == 0
    assert payload["metrics"]["evidence_coverage"]["evidence_items_individually_scored"] == 0


def test_coverage_vectors_equal_completion_vectors() -> None:
    payload = _json(audit.OUT_PROVENANCE)

    for system in ("workflow", "agent"):
        identity = payload["metric_vector_identity"][system]
        assert identity["dimension_vector_equals_completion_vector"] is True
        assert identity["claim_vector_equals_completion_vector"] is True
        assert identity["evidence_vector_equals_completion_vector"] is True


def test_denominators_and_judge_gap_are_reported() -> None:
    payload = _json(audit.OUT_VALIDITY)
    failure = payload["failure_and_body_audit"]

    assert failure["citation_validity_denominator"]["convention"] == (
        "VACUOUS_VALIDITY_CONVENTION_FOR_EMPTY_CITATION_SETS"
    )
    assert failure["unsupported_claim_denominator"]["evaluated_core_claim_count"] == 0
    assert payload["semantic_judge_complete"] is False
    assert payload["semantic_judge_requests"] == 0
    assert payload["judge_requests"] == 0


def test_original_blind_score_hash_unchanged() -> None:
    payload = _json(audit.OUT_VALIDITY)

    assert payload["original_blind_score_hash_unchanged"] is True
    assert (
        payload["pre_validity_audit_lock"]["original_blind_score_bundle_hash"]
        == audit.ORIGINAL_BLIND_SCORE_HASH
    )


def test_attempt4_metrics_and_failure_counts_unchanged() -> None:
    payload = _json(audit.OUT_VALIDITY)
    failure = payload["failure_and_body_audit"]

    assert payload["attempt4_integrity_unchanged"] is True
    assert failure["workflow_failure_categories"]["SYSTEM_SCHEMA_FAILURE"]["count"] == 10
    assert failure["workflow_failure_categories"]["SYSTEM_VERIFICATION_FAILURE"]["count"] == 50
    assert failure["agent_failure_categories"]["SYSTEM_PROVIDER_FAILURE"]["count"] == 4
    assert failure["workflow_http_runtime_succeeded"] == 60
    assert failure["workflow_units_with_provider_calls"] == 60


def test_release_claim_boundary_and_readiness() -> None:
    boundary = _json(audit.OUT_BOUNDARY)
    readiness = _json(audit.OUT_READINESS)

    assert boundary["release_readiness"] == "READY_WITH_SEMANTIC_EVALUATION_LIMITATION"
    assert readiness["status"] == "READY_WITH_LIMITATIONS"
    assert readiness["readiness_detail"] == "READY_WITH_SEMANTIC_EVALUATION_LIMITATION"
    assert any("semantic research quality" in item for item in boundary["not_allowed"])
    assert "FULLY_VALIDATED_SEMANTIC_BENCHMARK" in readiness["must_not_claim"]


def test_validity_audit_records_zero_new_runtime_activity() -> None:
    payload = _json(audit.OUT_VALIDITY)

    assert payload["new_provider_requests"] == 0
    assert payload["new_agent_runs"] == 0
    assert payload["new_workflow_runs"] == 0
    assert payload["semantic_judge_requests"] == 0
