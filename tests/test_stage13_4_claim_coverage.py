# ruff: noqa: E501
from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_research.evaluation.canonical_hash import sha256_canonical_jsonl_file
from paper_research.generation.citation_registry import CitationRegistry
from paper_research.generation.prompts import QA_REQUIRED_CLAIMS_CITATION_ID_V3, qa_system_prompt
from paper_research.generation.required_claim_output import (
    RequiredClaimsQA,
    RequiredClaimValidationError,
    required_claim_output_token_budget,
    validate_required_claim_slots,
)
from scripts.evidence_qa_dev_lib_v1 import read_jsonl
from scripts.review_evidence_qa_dev_v2_citations_v1 import (
    AUDIT,
    EVIDENCE,
    LABELS,
    validate,
)

DATA = Path("data/evaluation")
FAILURE_STAGES = {"retrieval_candidate_missing", "retrieval_ranked_out", "context_truncated", "claim_allocation_missing", "evidence_marked_incomplete", "prompt_claim_omitted", "model_omitted_claim", "model_merged_claim", "parser_dropped_claim", "schema_validation_failure", "malformed_json", "required_claim_matching_failure", "evaluator_false_negative", "gold_claim_ambiguous", "unknown", None}


def test_dev_v2_citation_audit_is_frozen_approved_and_source_valid() -> None:
    rows = read_jsonl(AUDIT)
    validate(rows)
    assert len(rows) == len({row["sample_id"] for row in rows}) == 57
    assert all(row["human_review_status"] == "approved" and row["human_label"] in LABELS for row in rows)
    assert all(row["source_hash"] == row["source_raw_sha256_at_review"] for row in rows)
    assert all(row["source_canonical_sha256"] == sha256_canonical_jsonl_file(EVIDENCE) for row in rows)
    required = {"question", "answerable", "required_claim_match", "citation_id", "citation_triple", "cited_evidence_context", "evidence_source", "block_type", "registry_hash", "source_record_hash", "immutable_record_hash"}
    assert all(required.issubset(row) for row in rows)


def test_review_validation_rejects_missing_reviewer_and_immutable_change() -> None:
    rows = read_jsonl(AUDIT)
    row = rows[0]
    row.update({"human_review_status": "approved", "human_label": next(iter(LABELS)), "reviewer": "", "reviewed_at": "2026-01-01T00:00:00Z", "review_notes": "note"})
    with pytest.raises(RuntimeError, match="incomplete approved"):
        validate(rows)
    rows = read_jsonl(AUDIT)
    rows[0]["claim_text"] += " changed"
    with pytest.raises(RuntimeError, match="immutable fields changed"):
        validate(rows)


def test_review_pack_is_minimal_and_secret_free() -> None:
    canonical = [AUDIT, DATA / "evidence-qa-dev-v2.json", DATA / "gold-set-v1.jsonl"]
    body = b"".join(path.read_bytes() for path in canonical)
    assert all(path.exists() for path in canonical)
    assert b'"Authorization":' not in body and b"Bearer sk-" not in body


def test_claim_coverage_audit_is_complete_unique_and_failures_visible() -> None:
    rows = read_jsonl(DATA / "dev-v2-claim-coverage-audit-v1.jsonl")
    assert len(rows) == 27
    assert len({(row["question_id"], row["required_claim_id"]) for row in rows}) == 27
    assert all(row["coverage_failure_stage"] in FAILURE_STAGES for row in rows)
    assert all(sum(bool(row[key]) for key in ("generated", "omitted", "merged_into_other_claim")) == 1 for row in rows)
    q050 = [row for row in rows if row["question_id"] == "q050"]
    assert len(q050) == 3 and all(row["coverage_failure_stage"] == "malformed_json" for row in q050)
    assert sum(row["coverage_credit"] for row in rows) == 14


def test_counterfactual_is_offline_and_does_not_replace_q050() -> None:
    payload = json.loads((DATA / "dev-v2-claim-coverage-counterfactual-v1.json").read_text(encoding="utf-8"))
    assert payload["live_llm_calls"] == payload["embedding_api_calls"] == 0
    assert payload["historical_runs_modified"] is False
    assert payload["parser_replay"]["valid_raw_responses"] == 9
    assert payload["parser_replay"]["diagnostic_salvage_only"] == ["q050"]
    assert payload["matcher_replay"]["original_coverage"] == pytest.approx(14 / 27)
    assert payload["matcher_replay"]["human_adjudication_required"] is True
    for variants in payload["context_budget_replay"].values():
        assert variants["cap_adjacent_25pct"]["token_count"] <= variants["current"]["token_count"]
        assert variants["remove_weak_adjacent"]["token_count"] <= variants["current"]["token_count"]


def allocated_registry() -> CitationRegistry:
    from paper_research.generation.citation_registry import CitationRegistryEntry

    entries = [
        CitationRegistryEntry(citation_id="E001", evidence_id="ev1", paper_id="p1", page=1, block_id="b1", claim_ids=["rc1"], context_position=1),
        CitationRegistryEntry(citation_id="E002", evidence_id="ev2", paper_id="p1", page=1, block_id="b2", claim_ids=["rc2"], context_position=2),
    ]
    digest = CitationRegistry.compute_hash(entries)
    entries = [item.model_copy(update={"registry_hash": digest}) for item in entries]
    return CitationRegistry(entries=entries, registry_hash=digest)


def test_prompt_v3_slots_and_claim_local_citations() -> None:
    registry = allocated_registry()
    output = RequiredClaimsQA.model_validate({"question_id": "qfixture", "answerable": True, "required_claim_results": [{"required_claim_id": "rc1", "status": "answered", "claim_text": "supported", "citation_ids": ["E001"], "omission_reason": None}, {"required_claim_id": "rc2", "status": "unsupported", "claim_text": None, "citation_ids": [], "omission_reason": "insufficient evidence"}], "refusal_reason": None, "prompt_version": "qa-required-claims-citation-id-v3", "citation_protocol": "citation-id-v2"})
    allocations = {"rc1": {"E001"}, "rc2": {"E002"}}
    validate_required_claim_slots(output, ["rc1", "rc2"], registry, allocations)
    assert "Never silently omit" in qa_system_prompt(QA_REQUIRED_CLAIMS_CITATION_ID_V3)
    bad = output.model_copy(update={"required_claim_results": [output.required_claim_results[0].model_copy(update={"citation_ids": ["E002"]}), output.required_claim_results[1]]})
    with pytest.raises(RequiredClaimValidationError, match="cross_claim_citation"):
        validate_required_claim_slots(bad, ["rc1", "rc2"], registry, allocations)
    with pytest.raises(RequiredClaimValidationError, match="missing_required_claim_id"):
        validate_required_claim_slots(output.model_copy(update={"required_claim_results": output.required_claim_results[:1]}), ["rc1", "rc2"], registry, allocations)


def test_prompt_v3_unsupported_unknown_unanswerable_and_token_budget() -> None:
    registry = allocated_registry()
    invalid = RequiredClaimsQA.model_validate({"question_id": "qfixture", "answerable": True, "required_claim_results": [{"required_claim_id": "rc1", "status": "unsupported", "claim_text": None, "citation_ids": ["E001"], "omission_reason": "none"}], "refusal_reason": None, "prompt_version": "qa-required-claims-citation-id-v3", "citation_protocol": "citation-id-v2"})
    with pytest.raises(RequiredClaimValidationError, match="unsupported_or_na_has_citation"):
        validate_required_claim_slots(invalid, ["rc1"], registry, {"rc1": {"E001"}})
    refused = RequiredClaimsQA.model_validate({"question_id": "q005", "answerable": False, "required_claim_results": [], "refusal_reason": "insufficient evidence", "prompt_version": "qa-required-claims-citation-id-v3", "citation_protocol": "citation-id-v2"})
    assert refused.answerable is False
    assert required_claim_output_token_budget(0).calculated_max_output_tokens == 256
    assert required_claim_output_token_budget(1).calculated_max_output_tokens == 448
    assert required_claim_output_token_budget(3).calculated_max_output_tokens == 832
    assert required_claim_output_token_budget(100).calculated_max_output_tokens == 4096
    assert required_claim_output_token_budget(100).capped is True
    with pytest.raises(ValueError):
        required_claim_output_token_budget(-1)


def test_q050_malformed_raw_remains_strictly_invalid() -> None:
    raw = '{"question_id":"q050","answerable":true'
    with pytest.raises(json.JSONDecodeError):
        json.loads(raw)
