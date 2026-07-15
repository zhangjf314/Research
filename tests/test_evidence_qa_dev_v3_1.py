# ruff: noqa: E501
from __future__ import annotations

import json

import pytest

from paper_research.evaluation.canonical_hash import hash_with_metadata
from paper_research.generation.required_claim_output import (
    RequiredClaimValidationError,
    parse_and_validate_required_claim_response_v31,
)
from scripts import run_evidence_qa_dev_v3_1 as runner
from scripts.evidence_qa_dev_lib_v1 import DATA, canonical_hash, read_jsonl
from scripts.evidence_qa_dev_v3_1_lib import (
    CAPABILITY_HASH,
    PROMPT_HASH,
    SCHEMA_HASH,
    SOURCE_MANIFEST_HASH,
    build_manifest,
    build_required_claim_input,
)
from scripts.run_evidence_qa_dev_v3_1 import assert_live_authorized, preflight


def material(question_id: str = "q001"):
    payload, registry, _, _ = build_required_claim_input(question_id)
    claim_ids = [row["required_claim_id"] for row in payload["required_claims"]]
    allowed = {row["required_claim_id"]: set(row["allowed_citation_ids"]) for row in payload["required_claims"]}
    return payload, registry, claim_ids, allowed


def validate(raw: dict, question_id: str = "q001"):
    _, registry, claim_ids, allowed = material(question_id)
    return parse_and_validate_required_claim_response_v31(
        json.dumps(raw), expected_question_id=question_id, expected_claim_ids=claim_ids,
        registry=registry, allowed_by_claim=allowed, expected_registry_hash=registry.registry_hash,
    )


def unsupported_response(question_id: str = "q001") -> dict:
    payload, _, _, _ = material(question_id)
    return {
        "question_id": question_id, "answerable": True,
        "required_claim_results": [
            {"required_claim_id": row["required_claim_id"], "status": "unsupported", "claim_text": None, "citation_ids": [], "omission_reason": "Evidence is insufficient."}
            for row in payload["required_claims"]
        ],
        "refusal_reason": None, "prompt_version": "qa-required-claims-citation-id-v3.1", "citation_protocol": "citation-id-v2",
    }


def test_manifest_and_preflight_are_frozen() -> None:
    body = build_manifest()
    assert body["manifest_hash"] == SOURCE_MANIFEST_HASH
    assert body["question_count"] == 10
    assert body["total_required_claims"] == 27
    assert body["configuration"]["prompt_hash"] == PROMPT_HASH
    assert body["configuration"]["schema_hash"] == SCHEMA_HASH
    assert body["configuration"]["provider_capability_snapshot_hash"] == CAPABILITY_HASH
    assert preflight()["protocol_checks"]["response_format"] is True


def test_valid_required_claim_slots_and_q005_refusal() -> None:
    assert len(validate(unsupported_response()).required_claim_results) == 3
    q005 = {
        "question_id": "q005", "answerable": False, "required_claim_results": [],
        "refusal_reason": "The corpus does not report exact total energy consumption.",
        "prompt_version": "qa-required-claims-citation-id-v3.1", "citation_protocol": "citation-id-v2",
    }
    assert validate(q005, "q005").answerable is False


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda raw: {"q001": raw}, "question_wrapper_rejected"),
        (lambda raw: {"claims": []}, "legacy_schema_rejected"),
        (lambda raw: {raw["required_claim_results"][0]["required_claim_id"]: {}}, "claim_map_rejected"),
        (lambda raw: {**raw, "paper_id": "1706.03762"}, "valid_json_wrong_schema"),
        (lambda raw: {**raw, "required_claim_results": raw["required_claim_results"][:-1]}, "missing_slot"),
        (lambda raw: {**raw, "required_claim_results": raw["required_claim_results"] + [raw["required_claim_results"][0]]}, "duplicate_slot"),
        (lambda raw: {**raw, "required_claim_results": raw["required_claim_results"] + [{"required_claim_id": "cl-extra", "status": "unsupported", "claim_text": None, "citation_ids": [], "omission_reason": "none"}]}, "extra_slot"),
    ],
)
def test_strict_schema_rejections(mutate, code: str) -> None:
    with pytest.raises(RequiredClaimValidationError) as caught:
        validate(mutate(unsupported_response()))
    assert caught.value.code == code


def test_missing_live_authorization_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEV_V3_1_LIVE_AUTHORIZED", raising=False)
    with pytest.raises(RuntimeError, match="NOT_AUTHORIZED"):
        assert_live_authorized()


def test_mock_run_persists_ordered_artifacts(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def __init__(self) -> None:
            content = {
                "question_id": "q005", "answerable": False,
                "required_claim_results": [], "refusal_reason": "No reported total energy.",
                "prompt_version": "qa-required-claims-citation-id-v3.1",
                "citation_protocol": "citation-id-v2",
            }
            self.content = json.dumps({
                "model": "Qwen/Qwen3-8B", "choices": [{"message": {"content": json.dumps(content)}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140},
            }).encode()

        def raise_for_status(self) -> None:
            return None

    class Client:
        def post(self, *args, **kwargs):
            assert kwargs["json"]["response_format"] == {"type": "json_object"}
            assert "tools" not in kwargs["json"] and "functions" not in kwargs["json"]
            return Response()

    monkeypatch.setattr(runner, "RUN_ROOT", tmp_path)
    result = runner.run_one("q005", runner.Settings(), Client())
    assert result["status"] == "completed"
    assert result["usage"]["total_tokens"] == 140
    run_dir = tmp_path / result["run_id"]
    assert all((run_dir / name).exists() for name in (
        "required-claims-input.json", "exact-json-schema.json", "raw-provider-response.json",
        "provider-response-envelope.json", "result.json", "result.csv", "request-ledger.jsonl",
    ))
    events = [json.loads(line)["event"] for line in (run_dir / "request-ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events.index("provider_usage_recorded") < events.index("response_parsing_started")
    assert events[-1] == "completed"


def test_canonical_citation_audit_is_pending_and_hash_stable() -> None:
    audit = read_jsonl(DATA / "evidence-qa-dev-v3-1-citation-audit-v1.jsonl")
    evidence_path = DATA / "evidence-corpus-v1.jsonl"
    evidence = {
        (row["paper_id"], int(row["page"]), row["block_id"]): row
        for row in read_jsonl(evidence_path)
    }
    source = hash_with_metadata(evidence_path, "canonical_jsonl_v1")
    assert len(audit) == len({row["sample_id"] for row in audit}) == 33
    for row in audit:
        assert row["human_review_status"] == "pending"
        assert row["human_label"] is None
        assert row["reviewer"] is row["reviewed_at"] is row["review_notes"] is None
        assert row["source_canonical_sha256"] == source["value"]
        triple = row["citation_triple"]
        unit = evidence[(triple["paper_id"], int(triple["page"]), triple["block_id"])]
        assert row["source_record_hash"] == canonical_hash(unit)
        immutable = {
            key: value
            for key, value in row.items()
            if key
            not in {
                "human_review_status", "human_label", "reviewer", "reviewed_at",
                "review_notes", "immutable_record_hash",
            }
        }
        assert row["immutable_record_hash"] == canonical_hash(immutable)


def test_checkpoint_secret_and_file_audits_are_closed() -> None:
    file_audit = json.loads(
        (DATA / "stage13-8-checkpoint-file-audit-v1.json").read_text(encoding="utf-8")
    )
    secret = json.loads(
        (DATA / "stage13-8-checkpoint-secret-scan-v1.json").read_text(encoding="utf-8")
    )
    assert file_audit["uncertain_count"] == 0
    assert file_audit["raw_run_local_only"]["commit_candidate"] is False
    assert file_audit["env_modified"] is False
    assert secret["actual_secret_hits"] == 0
    assert secret["safe_to_commit"] is True
