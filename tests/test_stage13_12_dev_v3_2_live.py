from __future__ import annotations

import json

from scripts.evidence_qa_dev_lib_v1 import DATA, DEV_IDS

RUN_ROOT = DATA / "evidence-qa-dev-v3-2/runs"


def load(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def test_controlled_batch_has_exactly_one_run_per_question() -> None:
    results = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in RUN_ROOT.glob("live-dev-v3-2-*/final-result.json")
    ]
    assert len(results) == 10
    assert sorted(row["question_id"] for row in results) == sorted(DEV_IDS)
    assert all(row["request_attempt_count"] == 1 for row in results)
    assert all(row["retries"] == 0 for row in results)
    assert all(row["reranker_called"] is False for row in results)
    assert all(row["monetary_cost_usd"] == "0" for row in results)


def test_run_artifacts_are_complete_and_secret_free() -> None:
    required = {
        "required-claims-input.json",
        "exact-json-schema.json",
        "prompt-metadata.json",
        "rendered-system-prompt.txt",
        "rendered-user-prompt.txt",
        "provider-capability-snapshot.json",
        "response-format-parameters.json",
        "citation-registry.json",
        "candidate-evidence.json",
        "citation-policy-input.json",
        "raw-provider-response.json",
        "provider-response-envelope.json",
        "parsed-v3-2-output.json",
        "citation-selection-trace.json",
        "obligation-analysis.json",
        "numeric-validation.json",
        "comparison-validation.json",
        "claim-fallback-trace.json",
        "final-result.json",
        "retrieval-trace.json",
        "context-trace.json",
        "request-ledger.jsonl",
        "run-metadata.json",
    }
    for run_dir in RUN_ROOT.glob("live-dev-v3-2-*"):
        assert required <= {path.name for path in run_dir.iterdir() if path.is_file()}
        for path in run_dir.iterdir():
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            assert "authorization: bearer" not in text
            assert "llm_api_key" not in text


def test_formal_failures_are_preserved_without_retry() -> None:
    summary = load("evidence-qa-dev-v3-2.json")
    failures = summary["all_manifest_conservative"]["validation_failures"]
    assert failures == {
        "valid_json_wrong_schema": 1,
        "malformed_json": 2,
        "unknown_citation_id": 1,
    }
    assert summary["raw_model_layer"]["provider_completed"] == 10
    assert summary["raw_model_layer"]["raw_json_valid"] == 8
    assert summary["raw_model_layer"]["raw_schema_success"] == 7
    assert summary["final_policy_layer"]["final_schema_success"] == 6


def test_frozen_gates_fail_closed_and_human_review_is_pending() -> None:
    audit = load("evidence-qa-dev-v3-2-final-audit.json")
    assert audit["dev_v3_2_engineering_gate"] == "FAILED"
    assert audit["dev_v3_2_automated_quality_gate"] == "FAILED"
    assert audit["dev_v3_2_human_support_gate"] == "PENDING"
    assert audit["dev_v3_2_quality_candidate_gate"] == "FAILED"
    assert audit["ready_for_full_qa"] is False
    assert audit["full_qa_executed"] is False
    assert audit["deep_research_executed"] is False


def test_citation_audit_does_not_auto_label_new_pairs() -> None:
    rows = [
        json.loads(line)
        for line in (DATA / "evidence-qa-dev-v3-2-citation-audit-v1.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    assert len(rows) == 14
    pending = [row for row in rows if row["requires_new_review"]]
    assert len(pending) == 7
    assert all(row["human_review_status"] == "pending" for row in pending)
    assert all(row["human_label"] is None for row in pending)


def test_historical_dev_v3_1_and_claim_gold_are_unchanged() -> None:
    historical = load("evidence-qa-dev-v3-1.json")
    freeze = load("claim-evidence-gold-dev-v1-freeze.json")
    assert historical["metrics"]["all_manifest_conservative"]["citation_recall"] == 0.295
    assert historical["dev_v3_1_quality_candidate_gate"] is False
    assert freeze["reviewed_file_hash"]["value"] == (
        "3cee289380c4b2ba861079d5f8470719a0d880f98812a5b55f28fb65693d37a6"
    )


def test_stage13_12_failure_freeze_is_stable_and_failed() -> None:
    freeze = load("stage13-12-dev-v3-2-failure-freeze-v1.json")
    audit = load("stage13-12-checkpoint-file-audit-v1.json")
    assert freeze["immutable"] is True
    assert len(freeze["runs"]) == 10
    assert freeze["historical_gate_status"] == {
        "engineering": "FAILED",
        "automated_quality": "FAILED",
        "human_support": "PENDING",
        "quality_candidate": "FAILED",
    }
    assert freeze["historical_active_reservations"] == 4
    assert freeze["historical_active_reserved_tokens"] == 96000
    assert audit["counts"]["uncertain"] == 0
    assert audit["raw_run_directory_count"] == 10
