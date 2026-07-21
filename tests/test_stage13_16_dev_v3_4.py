from __future__ import annotations

import json
from pathlib import Path

from paper_research.config import Settings
from paper_research.generation.schema_reliability import (
    DEV_V3_4_PROMPT_VERSION,
    REFUSAL_CANONICALIZATION_VERSION,
    canonicalize_model_payload_v2,
    dev_v3_4_system_prompt,
)
from scripts.evidence_qa_dev_lib_v1 import DEV_IDS, canonical_hash
from scripts.evidence_qa_dev_v3_3_lib import output_budget, safe_model_input
from scripts.evidence_qa_dev_v3_4_lib import build_freeze, write_visible_id_audit
from scripts.freeze_stage13_16_checkpoint_v1 import build_failure_freeze
from scripts.run_evidence_qa_dev_v3_4 import run_one, verify_request

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/evaluation"


class _Response:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _Client:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = 0

    def post(self, *_args, **_kwargs) -> _Response:
        self.calls += 1
        return _Response(self.payload)


def test_protocol_identity_manifest_configuration_and_budget() -> None:
    freeze = build_freeze()
    assert freeze["evaluation_version"] == "evidence-qa-dev-v3.4"
    assert freeze["question_ids"] == DEV_IDS
    assert freeze["question_count"] == 10
    assert freeze["answerable_questions"] == 9
    assert freeze["required_claims"] == 27
    assert freeze["q005_required_claims"] == 0
    assert freeze["prompt_version"] == DEV_V3_4_PROMPT_VERSION
    assert freeze["model_payload_schema_version"] == "required-claim-model-payload-v2"
    assert freeze["local_envelope_schema_version"] == "required-claim-local-envelope-v2"
    assert freeze["collection"] == "papers_jina_eval34_v2__20260713152149"
    assert freeze["embedding_dimensions"] == 1024
    assert freeze["provider"] == "siliconflow"
    assert freeze["model"] == "Qwen/Qwen3-8B"
    assert freeze["retry_policy"] == {"provider": 0, "json": 0, "citation": 0}
    assert freeze["frozen_before_live"] is True
    assert freeze["historical_backward_gate_effect"] is False
    assert output_budget(0)["calculated_max_output_tokens"] == 256
    assert output_budget(3)["calculated_max_output_tokens"] == 640
    assert output_budget(30)["calculated_max_output_tokens"] == 3072


def test_freeze_signature_is_canonical_and_stable() -> None:
    first = build_freeze()
    second = build_freeze()
    signature = first.pop("protocol_freeze_signature")
    assert signature == canonical_hash(first)
    assert second["protocol_freeze_signature"] == signature


def test_prompt_and_model_inputs_do_not_expose_forbidden_namespaces() -> None:
    prompt = dev_v3_4_system_prompt().lower()
    for token in (
        "qa-required-claims-citation-id-v3",
        "citation_ids",
        "evidence_id",
        "block_id",
        "paper_id",
        "relation_id",
        "gold_",
        "human_label",
    ):
        assert token not in prompt
    for question_id in DEV_IDS:
        safe = safe_model_input(question_id)[0]
        encoded = json.dumps(safe, ensure_ascii=False).lower()
        assert "citation_id" not in encoded
        assert "evidence_id" not in encoded
        assert "block_id" not in encoded
        assert "paper_id" not in encoded
        assert "gold_" not in encoded
        assert "human_label" not in encoded
    assert write_visible_id_audit()["gate"] == "PASSED"


def test_delivered_request_hash_is_exact_and_budget_bound() -> None:
    freeze = build_freeze()
    safe = safe_model_input("q001")[0]
    messages = [
        {"role": "system", "content": dev_v3_4_system_prompt()},
        {"role": "user", "content": json.dumps(safe, ensure_ascii=False)},
    ]
    body = {
        "model": "Qwen/Qwen3-8B",
        "messages": messages,
        "temperature": 0,
        "max_tokens": 640,
        "stream": False,
        "enable_thinking": False,
        "response_format": {"type": "json_object"},
    }
    hashes = verify_request(safe, messages, body, freeze)
    assert hashes["delivered_messages_hash"] == canonical_hash(messages)
    assert hashes["exact_delivered_request_body_hash"] == canonical_hash(body)


def test_contract_v2_canonicalization_and_q005_refusal() -> None:
    answerable = {
        "answerable": True,
        "required_claim_results": [
            {
                "required_claim_id": "c1",
                "status": "answered",
                "claim_text": "Supported claim.",
                "omission_reason": None,
            }
        ],
        "refusal_reason": "",
    }
    canonical = canonicalize_model_payload_v2(
        json.dumps(answerable), expected_claim_ids=["c1"]
    )
    assert canonical.canonicalization_rule == REFUSAL_CANONICALIZATION_VERSION
    assert canonical.canonicalization_applied is True
    assert canonical.changed_paths == ["$.refusal_reason"]
    assert canonical.canonical_payload.refusal_reason is None
    q005 = canonicalize_model_payload_v2(
        json.dumps(
            {
                "answerable": False,
                "required_claim_results": [],
                "refusal_reason": "The requested energy total is not reported.",
            }
        ),
        expected_claim_ids=[],
    )
    assert q005.canonicalization_applied is False
    assert q005.canonical_payload.answerable is False


def test_q005_single_request_run_is_closed_isolated_and_complete(
    tmp_path: Path, monkeypatch
) -> None:
    import scripts.run_evidence_qa_dev_v3_4 as runner

    payload = {
        "answerable": False,
        "required_claim_results": [],
        "refusal_reason": "The supplied evidence does not report the requested total.",
    }
    response = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": json.dumps(payload)},
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    }
    monkeypatch.setattr(runner, "RUN_ROOT", tmp_path)
    client = _Client(response)
    result = run_one("q005", Settings(), client, build_freeze())
    assert client.calls == 1
    assert result["status"] == "completed"
    assert result["request_attempt_count"] == 1
    assert result["provider_completed_request_count"] == 1
    assert result["usage_record_count"] == 1
    assert result["settled_reservation_count"] == 1
    assert result["active_reserved_tokens"] == 0
    assert result["retries"] == 0
    assert result["reranker_called"] is False
    assert result["final_answer"]["answerable"] is False
    assert result["final_answer"]["required_claim_results"] == []
    run_dir = next(tmp_path.iterdir())
    required = {
        "required-claims-input.json",
        "model-payload-schema.json",
        "local-envelope-schema.json",
        "canonicalization-policy.json",
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
        "structural-validation.json",
        "canonicalization-trace.json",
        "canonical-payload.json",
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
    assert required <= {path.name for path in run_dir.iterdir()}
    ledger = [
        json.loads(line)
        for line in (run_dir / "request-ledger.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert sum(row["event"] == "request_started" for row in ledger) == 1
    assert sum(row["event"] == "provider_usage_recorded" for row in ledger) == 1
    assert sum(row["event"] == "reservation_settled" for row in ledger) == 1


def test_historical_negative_gates_and_payload_replay_remain_preserved() -> None:
    stage14 = json.loads(
        (DATA / "evidence-qa-dev-v3-3-final-audit.json").read_text(encoding="utf-8")
    )
    stage15 = json.loads(
        (DATA / "dev-v3-3-payload-contract-v2-final-audit.json").read_text(encoding="utf-8")
    )
    assert stage14["dev_v3_3_engineering_gate"] == "FAILED"
    assert stage14["ready_for_full_qa"] is False
    assert stage15["payload_contract_v2_engineering_gate"] == "PASSED"
    assert stage15["payload_contract_v2_ready"] is True


def test_controlled_live_batch_is_single_attempt_and_accounting_closed() -> None:
    root = DATA / "evidence-qa-dev-v3-4/runs"
    results = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in root.glob("live-dev-v3-4-*/final-result.json")
    ]
    assert len(results) == 10
    assert sorted(row["question_id"] for row in results) == sorted(DEV_IDS)
    assert sum(row["request_attempt_count"] for row in results) == 10
    assert sum(row["provider_completed_request_count"] for row in results) == 10
    assert sum(row["provider_failure_count"] for row in results) == 0
    assert sum(row["usage_record_count"] for row in results) == 10
    assert sum(row["settled_reservation_count"] for row in results) == 10
    assert sum(row["active_reserved_tokens"] for row in results) == 0
    assert all(row["retries"] == 0 for row in results)
    assert all(row["reranker_called"] is False for row in results)
    assert all(row["template_fallback"] is False for row in results)


def test_formal_negative_result_is_preserved_and_conservatively_scored() -> None:
    summary = json.loads(
        (DATA / "evidence-qa-dev-v3-4.json").read_text(encoding="utf-8")
    )
    audit = json.loads(
        (DATA / "evidence-qa-dev-v3-4-final-audit.json").read_text(encoding="utf-8")
    )
    assert summary["raw_payload_layer"]["provider_completed"] == 10
    assert summary["raw_payload_layer"]["raw_json_valid"] == 10
    assert summary["raw_payload_layer"]["structural_payload_success"] == 1
    assert summary["raw_payload_layer"]["canonical_payload_success"] == 1
    assert summary["final_policy_layer"]["final_schema_success"] == 1
    assert summary["final_policy_layer"]["final_slot_success"] == 0
    assert summary["all_manifest_conservative"]["validation_failures"] == {
        "ValidationError": 9
    }
    assert summary["all_manifest_conservative"]["total_tokens"] == 20078
    assert summary["all_manifest_conservative"]["effective_active_reservations"] == 0
    assert audit["dev_v3_4_raw_payload_gate"] == "FAILED"
    assert audit["dev_v3_4_final_policy_engineering_gate"] == "FAILED"
    assert audit["dev_v3_4_engineering_gate"] == "FAILED"
    assert audit["dev_v3_4_automated_quality_gate"] == "FAILED"
    assert audit["dev_v3_4_human_support_gate"] == "FAILED"
    assert audit["ready_for_full_qa"] is False


def test_all_live_artifacts_exist_and_are_secret_free() -> None:
    required = {
        "required-claims-input.json",
        "model-payload-schema.json",
        "local-envelope-schema.json",
        "canonicalization-policy.json",
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
        "structural-validation.json",
        "canonicalization-trace.json",
        "canonical-payload.json",
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
    for run_dir in (DATA / "evidence-qa-dev-v3-4/runs").glob("live-dev-v3-4-*"):
        assert required <= {path.name for path in run_dir.iterdir()}
        combined = "\n".join(
            path.read_text(encoding="utf-8", errors="ignore")
            for path in run_dir.iterdir()
            if path.is_file()
        ).lower()
        assert "authorization: bearer" not in combined
        assert "llm_api_key" not in combined


def test_stage13_16_historical_protection_passes() -> None:
    protection = json.loads(
        (DATA / "stage13-16-historical-protection-v1.json").read_text(encoding="utf-8")
    )
    assert protection["baseline_commit"] == "70b2401f29cfec0cb8b3764fc945bc75dcfda96f"
    assert protection["changed_count"] == 0
    assert protection["gate"] == "PASSED"
    assert protection["stage13_14_gate"] == "FAILED_AND_PRESERVED"


def test_stage13_16_checkpoint_audit_and_failure_freeze_are_stable() -> None:
    file_audit = json.loads(
        (DATA / "stage13-16-checkpoint-file-audit-v1.json").read_text(
            encoding="utf-8"
        )
    )
    frozen = json.loads(
        (DATA / "stage13-16-dev-v3-4-failure-freeze-v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert file_audit["gate"] == "PASSED"
    assert file_audit["uncertain_count"] == 0
    assert file_audit["raw_run_count"] == 10
    assert file_audit["raw_runs_local_only"] is True
    assert file_audit["review_zip_local_only_count"] == 2
    assert file_audit["secret_hits"] == []
    assert build_failure_freeze() == frozen
    assert frozen["immutable"] is True
    assert frozen["run_count"] == 10
    assert frozen["failure_taxonomy"]["raw_status_distribution"] == {
        "answerable": 3,
        "supported": 23,
        "unsupported": 1,
    }
    assert frozen["failure_taxonomy"]["missing_top_level_refusal_reason"] == 1
    assert frozen["failure_taxonomy"]["strict_structural_pass"] == 1
    assert frozen["failure_taxonomy"]["final_slots"] == 0
    assert frozen["gate_results"]["engineering"] == "FAILED"
