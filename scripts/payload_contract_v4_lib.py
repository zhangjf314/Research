"""Payload Contract v4 protocol, immutable inputs, and diagnostic projection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from paper_research.generation.citation_selection import (
    CITATION_SELECTION_VERSION,
    COMPARISON_VALIDATION_VERSION,
    NUMERIC_VALIDATION_VERSION,
    OBLIGATION_POLICY_VERSION,
)
from paper_research.generation.schema_reliability import (
    DEV_V3_6_CANDIDATE_PROMPT_VERSION,
    LOCAL_ENVELOPE_V4_VERSION,
    MODEL_PAYLOAD_V4_VERSION,
    PAYLOAD_V4_ADAPTER,
    SCHEMA_RELIABILITY_V4_CANDIDATE,
    SLOT_STATUS_DERIVATION_V2_VERSION,
    LocalEnvelopeV4,
    dev_v3_6_candidate_system_prompt,
)

try:
    from scripts.evidence_qa_dev_lib_v1 import DATA, DOCS, canonical_hash
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import DATA, DOCS, canonical_hash  # type: ignore[no-redef]

ROOT = DATA.parents[1]
RUN_ROOT = DATA / "evidence-qa-dev-v3-4/runs"
PREFLIGHT = DATA / "payload-contract-v4-preflight-inputs-v1.json"
PREFLIGHT_DOC = DOCS / "payload-contract-v4-preflight-inputs-v1.md"
PROTOCOL = DATA / "payload-contract-v4-protocol.json"
PROTOCOL_DOC = DOCS / "payload-contract-v4-protocol.md"
REPLAY = DATA / "dev-v3-4-payload-contract-v4-replay.json"
REPLAY_CSV = DATA / "dev-v3-4-payload-contract-v4-replay.csv"
REPLAY_DOC = DOCS / "dev-v3-4-payload-contract-v4-replay.md"
FINAL_AUDIT = DATA / "dev-v3-4-payload-contract-v4-final-audit.json"
SAFETY = DATA / "payload-v4-slot-shape-safety-audit-v1.json"
SAFETY_DOC = DOCS / "payload-v4-slot-shape-safety-audit-v1.md"
PLACEHOLDER_REMOVAL_VERSION = "empty-placeholder-field-removal-v1"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_preflight() -> dict[str, Any]:
    v3_protocol = json.loads(
        (DATA / "payload-contract-v3-protocol.json").read_text(encoding="utf-8")
    )
    failure_freeze = json.loads(
        (DATA / "stage13-16-dev-v3-4-failure-freeze-v1.json").read_text(
            encoding="utf-8"
        )
    )
    body = {
        "schema_version": "payload-contract-v4-preflight-inputs-v1",
        "immutable": True,
        "stage13_17_head": "1cbeb094a4c3a2466cb4c9079fb328114e670d4f",
        "inputs": {
            "payload_contract_v3_protocol": {
                "path": "data/evaluation/payload-contract-v3-protocol.json",
                "sha256": file_hash(DATA / "payload-contract-v3-protocol.json"),
                "protocol_signature": v3_protocol["protocol_signature"],
            },
            "payload_v3_prompt": {
                "version": v3_protocol["prompt_version"],
                "hash": v3_protocol["prompt_hash"],
            },
            "payload_v3_schema": {
                "version": v3_protocol["model_payload_version"],
                "hash": v3_protocol["model_payload_schema_hash"],
            },
            "local_envelope_v3": {
                "version": v3_protocol["local_envelope_version"],
                "hash": v3_protocol["local_envelope_schema_hash"],
            },
            "status_derivation_v1": v3_protocol["status_derivation_version"],
            "stage13_16_raw_response_hashes": {
                row["question_id"]: row["raw_response_sha256"]
                for row in failure_freeze["runs"]
            },
            "stage13_16_failure_freeze": {
                "path": "data/evaluation/stage13-16-dev-v3-4-failure-freeze-v1.json",
                "sha256": file_hash(
                    DATA / "stage13-16-dev-v3-4-failure-freeze-v1.json"
                ),
                "signature": failure_freeze["failure_freeze_signature"],
            },
            "stage13_17_replay": {
                "path": "data/evaluation/dev-v3-4-payload-contract-v3-replay.json",
                "sha256": file_hash(
                    DATA / "dev-v3-4-payload-contract-v3-replay.json"
                ),
            },
            "stage13_17_safety": {
                "path": "data/evaluation/slot-status-derivation-safety-audit-v1.json",
                "sha256": file_hash(
                    DATA / "slot-status-derivation-safety-audit-v1.json"
                ),
            },
            "policy_versions": {
                "citation_selection": CITATION_SELECTION_VERSION,
                "obligation": OBLIGATION_POLICY_VERSION,
                "numeric": NUMERIC_VALIDATION_VERSION,
                "comparison": COMPARISON_VALIDATION_VERSION,
            },
        },
    }
    body["preflight_signature"] = canonical_hash(body)
    return body


def write_preflight() -> dict[str, Any]:
    body = build_preflight()
    if PREFLIGHT.exists():
        existing = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
        if existing != body:
            raise RuntimeError("PAYLOAD_CONTRACT_V4_PREFLIGHT_DRIFT")
    else:
        PREFLIGHT.write_text(
            json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        PREFLIGHT_DOC.write_text(
            "# Payload Contract v4 Preflight Inputs\n\n"
            f"- Signature: `{body['preflight_signature']}`\n"
            "- Stage 13.16 raw hashes, failure freeze, Payload v3 protocol/replay/"
            "safety, and local citation policies are immutable inputs.\n",
            encoding="utf-8",
        )
    return body


def build_protocol() -> dict[str, Any]:
    body = {
        "schema_version": "payload-contract-v4-protocol-v1",
        "candidate": SCHEMA_RELIABILITY_V4_CANDIDATE,
        "model_payload_version": MODEL_PAYLOAD_V4_VERSION,
        "model_payload_schema_hash": canonical_hash(PAYLOAD_V4_ADAPTER.json_schema()),
        "local_envelope_version": LOCAL_ENVELOPE_V4_VERSION,
        "local_envelope_schema_hash": canonical_hash(
            LocalEnvelopeV4.model_json_schema()
        ),
        "prompt_version": DEV_V3_6_CANDIDATE_PROMPT_VERSION,
        "prompt_hash": canonical_hash(dev_v3_6_candidate_system_prompt()),
        "status_derivation_version": SLOT_STATUS_DERIVATION_V2_VERSION,
        "placeholder_removal_version": PLACEHOLDER_REMOVAL_VERSION,
        "canonicalization": "none",
        "answerable_top_level_fields": [
            "answerable",
            "required_claim_results",
        ],
        "unanswerable_top_level_fields": [
            "answerable",
            "required_claim_results",
            "refusal_reason",
        ],
        "answered_slot_fields": ["required_claim_id", "claim_text"],
        "unsupported_slot_fields": [
            "required_claim_id",
            "omission_reason",
        ],
        "model_outputs_status": False,
        "model_outputs_citation": False,
        "null_sentinel_required": False,
        "empty_sentinel_required": False,
        "provider_transport": "response_format=json_object",
        "provider_json_schema_sent": False,
        "next_live_authorized": False,
    }
    body["protocol_signature"] = canonical_hash(body)
    return body


def write_protocol() -> dict[str, Any]:
    body = build_protocol()
    if PROTOCOL.exists():
        existing = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        if existing != body:
            raise RuntimeError("PAYLOAD_CONTRACT_V4_PROTOCOL_DRIFT")
    else:
        PROTOCOL.write_text(
            json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        PROTOCOL_DOC.write_text(
            "# Payload Contract v4 Protocol\n\n"
            f"- Signature: `{body['protocol_signature']}`\n"
            f"- Prompt: `{body['prompt_version']}` / `{body['prompt_hash']}`\n"
            "- Claim slots use mutually exclusive field-presence shapes. The model "
            "does not output status, citations, null sentinels, or empty sentinels.\n"
            f"- Diagnostic placeholder policy: `{PLACEHOLDER_REMOVAL_VERSION}`\n"
            "- Canonicalization: none. Next live authorization: false.\n",
            encoding="utf-8",
        )
    return body


def _failure(
    reason: str,
    *,
    operations: list[dict[str, Any]],
    semantic_conflict: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "field_projection_completed": False,
        "failure": reason,
        "semantic_conflict": semantic_conflict,
        "operations": operations,
        "semantic_field_modifications": 0,
        **extra,
    }


def project_raw_payload_v4(raw: dict[str, Any]) -> dict[str, Any]:
    """Delete only deprecated or exact-empty opposite-side placeholder fields."""
    if not isinstance(raw, dict):
        return _failure("top_level_not_object", operations=[])
    allowed_top = {"answerable", "required_claim_results", "refusal_reason"}
    extra_top = sorted(set(raw) - allowed_top)
    if extra_top:
        return _failure(
            "unknown_top_level_extra_field",
            operations=[],
            extra_fields=extra_top,
        )
    projected = json.loads(json.dumps(raw, ensure_ascii=False))
    operations: list[dict[str, Any]] = []
    if projected.get("answerable") is True and "refusal_reason" in projected:
        old = projected["refusal_reason"]
        if old not in {None, ""}:
            return _failure(
                "nonempty_answerable_refusal_conflict",
                operations=operations,
                semantic_conflict=True,
            )
        del projected["refusal_reason"]
        operations.append(
            {
                "version": PLACEHOLDER_REMOVAL_VERSION,
                "operation": "remove_deprecated_field",
                "path": "$.refusal_reason",
                "old_value_type": type(old).__name__,
                "old_value": old,
            }
        )
    slots = projected.get("required_claim_results")
    if not isinstance(slots, list):
        return _failure("required_claim_results_not_array", operations=operations)
    semantic_conflict_indices = []
    allowed_slot = {
        "required_claim_id",
        "status",
        "claim_text",
        "omission_reason",
    }
    for index, slot in enumerate(slots):
        if not isinstance(slot, dict):
            return _failure(
                "slot_not_object",
                operations=operations,
                slot_index=index,
            )
        extra = sorted(set(slot) - allowed_slot)
        if extra:
            return _failure(
                "unknown_slot_extra_field",
                operations=operations,
                slot_index=index,
                extra_fields=extra,
            )
        if "status" in slot:
            old = slot.pop("status")
            operations.append(
                {
                    "version": PLACEHOLDER_REMOVAL_VERSION,
                    "operation": "remove_deprecated_field",
                    "path": f"$.required_claim_results[{index}].status",
                    "old_value_type": type(old).__name__,
                    "old_value": old,
                }
            )
        has_claim = "claim_text" in slot
        has_omission = "omission_reason" in slot
        claim = slot.get("claim_text")
        omission = slot.get("omission_reason")
        claim_nonempty = isinstance(claim, str) and bool(claim.strip())
        omission_nonempty = isinstance(omission, str) and bool(omission.strip())
        if claim_nonempty and omission == "":
            del slot["omission_reason"]
            operations.append(
                {
                    "version": PLACEHOLDER_REMOVAL_VERSION,
                    "operation": "remove_empty_opposite_field",
                    "path": f"$.required_claim_results[{index}].omission_reason",
                    "old_value_type": "str",
                    "old_value": "",
                }
            )
        elif omission_nonempty and claim == "":
            del slot["claim_text"]
            operations.append(
                {
                    "version": PLACEHOLDER_REMOVAL_VERSION,
                    "operation": "remove_empty_opposite_field",
                    "path": f"$.required_claim_results[{index}].claim_text",
                    "old_value_type": "str",
                    "old_value": "",
                }
            )
        elif claim_nonempty and omission_nonempty:
            semantic_conflict_indices.append(index)
        elif has_claim and claim is None:
            return _failure(
                "null_claim_sentinel_not_removable",
                operations=operations,
                slot_index=index,
            )
        elif has_omission and omission is None:
            return _failure(
                "null_omission_sentinel_not_removable",
                operations=operations,
                slot_index=index,
            )
        elif has_claim and has_omission:
            return _failure(
                "dual_empty_or_invalid_fields",
                operations=operations,
                slot_index=index,
            )
        elif not has_claim and not has_omission:
            return _failure(
                "missing_both_content_fields",
                operations=operations,
                slot_index=index,
            )
    if semantic_conflict_indices:
        return _failure(
            "unprojectable_semantic_conflict",
            operations=operations,
            semantic_conflict=True,
            semantic_conflict_count=len(semantic_conflict_indices),
            semantic_conflict_slot_indices=semantic_conflict_indices,
        )
    return {
        "field_projection_completed": True,
        "projected_payload": projected,
        "raw_payload_hash": canonical_hash(raw),
        "projected_payload_hash": canonical_hash(projected),
        "operations": operations,
        "placeholder_fields_removed": sum(
            row["operation"] == "remove_empty_opposite_field"
            for row in operations
        ),
        "semantic_conflict": False,
        "semantic_field_modifications": 0,
        "nonempty_claim_text_modifications": 0,
        "nonempty_omission_text_modifications": 0,
        "answerability_modifications": 0,
        "slot_count_modifications": 0,
    }
