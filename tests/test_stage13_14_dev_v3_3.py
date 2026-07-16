from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from paper_research.generation.required_claim_output import RequiredClaimValidationError
from paper_research.generation.schema_reliability import (
    DEV_V3_3_PROMPT_VERSION,
    dev_v3_3_system_prompt,
    parse_minimal_payload,
)
from scripts.evidence_qa_dev_lib_v1 import DEV_IDS
from scripts.evidence_qa_dev_v3_3_lib import (
    build_freeze,
    output_budget,
    safe_model_input,
    write_visible_id_audit,
)
from scripts.run_evidence_qa_dev_v3_3 import verify_safe_request

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/evaluation"


def test_protocol_identity_and_fixed_manifest() -> None:
    freeze = build_freeze()
    assert freeze["evaluation_version"] == "evidence-qa-dev-v3.3"
    assert freeze["prompt_version"] == DEV_V3_3_PROMPT_VERSION
    assert freeze["fixed_manifest"]["question_ids"] == DEV_IDS
    assert freeze["fixed_manifest"]["required_claims"] == 27
    assert freeze["fixed_manifest"]["q005_required_claims"] == 0
    assert freeze["frozen_before_live"] is True


def test_minimal_prompt_contains_no_legacy_or_output_identifier_namespace() -> None:
    prompt = dev_v3_3_system_prompt().lower()
    for token in (
        "qa-required-claims-citation-id-v3.1",
        "qa-required-claims-citation-id-v3.2",
        "citation_ids",
        "evidence_id",
        "block_id",
        "paper_id",
    ):
        assert token not in prompt


def test_all_model_inputs_hide_internal_gold_and_human_fields() -> None:
    for question_id in DEV_IDS:
        safe, _full, _registry, _trace = safe_model_input(question_id)
        encoded = json.dumps(safe, ensure_ascii=False).lower()
        for token in (
            "evidence_id",
            "citation_id",
            "block_id",
            "paper_id",
            "relation_id",
            "gold_",
            "human_label",
        ):
            assert token not in encoded
    assert write_visible_id_audit()["gate"] == "PASSED"


def test_output_budget_is_frozen_and_not_question_specific() -> None:
    assert output_budget(0)["calculated_max_output_tokens"] == 256
    assert output_budget(1)["calculated_max_output_tokens"] == 384
    assert output_budget(3)["calculated_max_output_tokens"] == 640
    assert output_budget(30)["calculated_max_output_tokens"] == 3072
    assert output_budget(30)["capped"] is True


def test_three_slot_complex_payload_and_q005_refusal_fit_schema() -> None:
    _, full, _, _ = safe_model_input("q019")
    slots = [
        {
            "required_claim_id": row["required_claim_id"],
            "status": "answered",
            "claim_text": row["required_claim_text"],
            "omission_reason": None,
        }
        for row in full["required_claims"]
    ]
    parsed = parse_minimal_payload(
        json.dumps(
            {
                "answerable": True,
                "required_claim_results": slots,
                "refusal_reason": None,
            }
        ),
        expected_claim_ids=[row["required_claim_id"] for row in full["required_claims"]],
    )
    assert len(parsed.required_claim_results) == 3
    refusal = parse_minimal_payload(
        json.dumps(
            {
                "answerable": False,
                "required_claim_results": [],
                "refusal_reason": "The requested evidence is not reported.",
            }
        ),
        expected_claim_ids=[],
    )
    assert refusal.answerable is False


@pytest.mark.parametrize(
    "extra",
    [
        {"question_id": "q001"},
        {"prompt_version": DEV_V3_3_PROMPT_VERSION},
        {"citation_protocol": "citation-id-v2"},
        {"citation_ids": ["E001"]},
        {"evidence_id": "ev-x"},
    ],
)
def test_model_payload_rejects_forbidden_extra_fields(extra: dict) -> None:
    body = {
        "answerable": False,
        "required_claim_results": [],
        "refusal_reason": "insufficient",
        **extra,
    }
    with pytest.raises(RequiredClaimValidationError) as exc:
        parse_minimal_payload(json.dumps(body), expected_claim_ids=[])
    assert exc.value.code == "schema_validation_failure"


def test_delivered_hashes_are_exact_and_fail_on_legacy_prompt() -> None:
    freeze = build_freeze()
    safe, _, _, _ = safe_model_input("q005")
    messages = [
        {"role": "system", "content": dev_v3_3_system_prompt()},
        {"role": "user", "content": json.dumps(safe, ensure_ascii=False)},
    ]
    body = {
        "model": "Qwen/Qwen3-8B",
        "messages": messages,
        "temperature": 0,
        "max_tokens": 256,
        "stream": False,
        "enable_thinking": False,
        "response_format": {"type": "json_object"},
    }
    hashes = verify_safe_request(safe, messages, body, freeze)
    assert len(hashes["exact_delivered_request_body_hash"]) == 64
    bad = [
        {
            **messages[0],
            "content": messages[0]["content"] + " qa-required-claims-citation-id-v3.2",
        },
        messages[1],
    ]
    with pytest.raises(RuntimeError, match="DEV_V3_3_CONFIGURATION_INVALID"):
        verify_safe_request(safe, bad, {**body, "messages": bad}, freeze)


def test_historical_stage13_12_and_reconciliation_are_unchanged() -> None:
    stage12 = json.loads(
        (DATA / "evidence-qa-dev-v3-2-final-audit.json").read_text(encoding="utf-8")
    )
    reconciliation = json.loads(
        (DATA / "stage13-12-reservation-reconciliation-v1.json").read_text(encoding="utf-8")
    )
    assert stage12["dev_v3_2_engineering_gate"] == "FAILED"
    assert reconciliation["effective_active_reservations"] == 0
    assert reconciliation["historical_ledgers_modified"] is False


def test_controlled_live_batch_is_single_attempt_and_accounting_closed() -> None:
    root = DATA / "evidence-qa-dev-v3-3/runs"
    results = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in root.glob("live-dev-v3-3-*/final-result.json")
    ]
    assert len(results) == 10
    assert sorted(row["question_id"] for row in results) == sorted(DEV_IDS)
    assert sum(row["request_attempt_count"] for row in results) == 10
    assert sum(row["provider_completed_request_count"] for row in results) == 10
    assert sum(row["settled_reservation_count"] for row in results) == 10
    assert sum(row["released_reservation_count"] for row in results) == 0
    assert sum(row["billing_unknown_reservation_count"] for row in results) == 0
    assert sum(row["active_reserved_tokens"] for row in results) == 0
    assert all(row["retries"] == 0 for row in results)
    assert all(row["reranker_called"] is False for row in results)


def test_live_run_artifacts_complete_and_secret_free() -> None:
    required = {
        "required-claims-input.json",
        "model-payload-schema.json",
        "local-envelope-schema.json",
        "citation-registry.json",
        "candidate-evidence.json",
        "rendered-system-prompt.txt",
        "rendered-user-prompt.txt",
        "delivered-request-metadata.json",
        "protocol-snapshot.json",
        "accounting-reservation.json",
        "run-metadata.json",
        "raw-provider-response.json",
        "provider-response-envelope.json",
        "raw-model-payload.json",
        "payload-validation.json",
        "local-envelope-binding.json",
        "obligation-analysis.json",
        "citation-selection-trace.json",
        "numeric-validation.json",
        "comparison-validation.json",
        "claim-fallback-trace.json",
        "final-result.json",
        "request-ledger.jsonl",
    }
    for run_dir in (DATA / "evidence-qa-dev-v3-3/runs").glob("live-dev-v3-3-*"):
        assert required <= {path.name for path in run_dir.iterdir()}
        for path in run_dir.iterdir():
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            assert "authorization: bearer" not in text
            assert "llm_api_key" not in text


def test_formal_negative_result_is_preserved_without_normalization() -> None:
    summary = json.loads((DATA / "evidence-qa-dev-v3-3.json").read_text(encoding="utf-8"))
    audit = json.loads((DATA / "evidence-qa-dev-v3-3-final-audit.json").read_text(encoding="utf-8"))
    assert summary["raw_model_layer"]["raw_json_valid"] == 10
    assert summary["raw_model_layer"]["structural_schema_success"] == 10
    assert summary["raw_model_layer"]["model_payload_schema_success"] == 2
    assert summary["raw_model_layer"]["structural_slot_count"] == 27
    assert summary["raw_model_layer"]["required_slot_success"] == 3
    assert summary["all_manifest_conservative"]["validation_failures"] == {
        "answerable_has_refusal_reason": 8
    }
    assert summary["final_policy_layer"]["final_schema_success"] == 2
    assert summary["final_policy_layer"]["final_slot_success"] == 3
    assert audit["dev_v3_3_raw_payload_gate"] == "FAILED"
    assert audit["dev_v3_3_final_policy_engineering_gate"] == "FAILED"
    assert audit["dev_v3_3_engineering_gate"] == "FAILED"
    assert audit["ready_for_full_qa"] is False


def test_protocol_freeze_historical_hashes_still_match() -> None:
    freeze = json.loads(
        (DATA / "evidence-qa-dev-v3-3-protocol-freeze-v1.json").read_text(encoding="utf-8")
    )
    paths = {
        "stage13_12_failure_freeze": DATA / "stage13-12-dev-v3-2-failure-freeze-v1.json",
        "stage13_13_reconciliation": DATA / "stage13-12-reservation-reconciliation-v1.json",
    }
    for key, path in paths.items():
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == freeze["historical_protection_hashes"][key]
        )


def test_stage13_14_failure_freeze_preserves_negative_gate() -> None:
    freeze = json.loads(
        (DATA / "stage13-14-dev-v3-3-failure-freeze-v1.json").read_text(
            encoding="utf-8"
        )
    )
    file_audit = json.loads(
        (DATA / "stage13-14-checkpoint-file-audit-v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert freeze["immutable"] is True
    assert len(freeze["runs"]) == 10
    assert freeze["failure_taxonomy"] == {
        "answerable_has_refusal_reason": 8,
        "completed": 2,
    }
    assert freeze["gate_results"]["engineering"] == "FAILED"
    assert freeze["gate_results"]["ready_for_full_qa"] is False
    assert file_audit["counts"]["uncertain"] == 0
    assert file_audit["raw_run_directory_count"] == 10
