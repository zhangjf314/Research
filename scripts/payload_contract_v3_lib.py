"""Offline-only Payload Contract v3 protocol and diagnostic projection."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from paper_research.generation.schema_reliability import (
    DEV_V3_5_CANDIDATE_PROMPT_VERSION,
    LOCAL_ENVELOPE_V3_VERSION,
    MODEL_PAYLOAD_V3_VERSION,
    PAYLOAD_V3_ADAPTER,
    SCHEMA_RELIABILITY_V3_CANDIDATE,
    SLOT_STATUS_DERIVATION_VERSION,
    LocalEnvelopeV3,
    dev_v3_5_candidate_system_prompt,
)

try:
    from scripts.evidence_qa_dev_lib_v1 import DATA, DOCS, canonical_hash
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import DATA, DOCS, canonical_hash  # type: ignore[no-redef]

ROOT = DATA.parents[1]
RUN_ROOT = DATA / "evidence-qa-dev-v3-4/runs"
PROTOCOL = DATA / "payload-contract-v3-protocol.json"
PROTOCOL_DOC = DOCS / "payload-contract-v3-protocol.md"
FORENSICS = DATA / "dev-v3-4-status-field-forensics-v1.json"
REPLAY = DATA / "dev-v3-4-payload-contract-v3-replay.json"
REPLAY_CSV = DATA / "dev-v3-4-payload-contract-v3-replay.csv"
REPLAY_DOC = DOCS / "dev-v3-4-payload-contract-v3-replay.md"
FINAL_AUDIT = DATA / "dev-v3-4-payload-contract-v3-final-audit.json"
SAFETY = DATA / "slot-status-derivation-safety-audit-v1.json"
SAFETY_DOC = DOCS / "slot-status-derivation-safety-audit-v1.md"


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build_protocol() -> dict[str, Any]:
    body = {
        "schema_version": "payload-contract-v3-protocol-v1",
        "candidate": SCHEMA_RELIABILITY_V3_CANDIDATE,
        "model_payload_version": MODEL_PAYLOAD_V3_VERSION,
        "model_payload_schema_hash": canonical_hash(PAYLOAD_V3_ADAPTER.json_schema()),
        "local_envelope_version": LOCAL_ENVELOPE_V3_VERSION,
        "local_envelope_schema_hash": canonical_hash(
            LocalEnvelopeV3.model_json_schema()
        ),
        "prompt_version": DEV_V3_5_CANDIDATE_PROMPT_VERSION,
        "prompt_hash": canonical_hash(dev_v3_5_candidate_system_prompt()),
        "status_derivation_version": SLOT_STATUS_DERIVATION_VERSION,
        "canonicalization": "none",
        "answerable_branch_fields": [
            "answerable",
            "required_claim_results",
        ],
        "unanswerable_branch_fields": [
            "answerable",
            "required_claim_results",
            "refusal_reason",
        ],
        "model_slot_fields": [
            "required_claim_id",
            "claim_text",
            "omission_reason",
        ],
        "model_outputs_status": False,
        "model_outputs_citation": False,
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
            raise RuntimeError("PAYLOAD_CONTRACT_V3_PROTOCOL_DRIFT")
    else:
        PROTOCOL.write_text(
            json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        PROTOCOL_DOC.write_text(
            "# Payload Contract v3 Protocol\n\n"
            f"- Signature: `{body['protocol_signature']}`\n"
            f"- Model payload: `{MODEL_PAYLOAD_V3_VERSION}`\n"
            f"- Local envelope: `{LOCAL_ENVELOPE_V3_VERSION}`\n"
            f"- Prompt: `{DEV_V3_5_CANDIDATE_PROMPT_VERSION}` / "
            f"`{body['prompt_hash']}`\n"
            "- The model does not output slot status or citations. Local status is "
            "derived from the unique claim/omission content shape.\n"
            "- Canonicalization: none. Next live authorization: false.\n",
            encoding="utf-8",
        )
    return body


def project_raw_payload_v3(raw: dict[str, Any]) -> dict[str, Any]:
    """Remove only fields deprecated by v3; never repair semantic content."""
    if not isinstance(raw, dict):
        return {
            "projectable": False,
            "failure": "top_level_not_object",
            "operations": [],
            "semantic_modifications": 0,
        }
    allowed_top = {"answerable", "required_claim_results", "refusal_reason"}
    extra_top = sorted(set(raw) - allowed_top)
    if extra_top:
        return {
            "projectable": False,
            "failure": "unknown_top_level_extra_field",
            "extra_fields": extra_top,
            "operations": [],
            "semantic_modifications": 0,
        }
    projected = json.loads(json.dumps(raw, ensure_ascii=False))
    operations = []
    if projected.get("answerable") is True and "refusal_reason" in projected:
        old = projected.pop("refusal_reason")
        operations.append(
            {
                "operation": "remove_deprecated_field",
                "path": "$.refusal_reason",
                "old_value": old,
            }
        )
    slots = projected.get("required_claim_results")
    if not isinstance(slots, list):
        return {
            "projectable": False,
            "failure": "required_claim_results_not_array",
            "operations": operations,
            "semantic_modifications": 0,
        }
    allowed_slot = {
        "required_claim_id",
        "status",
        "claim_text",
        "omission_reason",
    }
    for index, slot in enumerate(slots):
        if not isinstance(slot, dict):
            return {
                "projectable": False,
                "failure": "slot_not_object",
                "slot_index": index,
                "operations": operations,
                "semantic_modifications": 0,
            }
        extra = sorted(set(slot) - allowed_slot)
        if extra:
            return {
                "projectable": False,
                "failure": "unknown_slot_extra_field",
                "slot_index": index,
                "extra_fields": extra,
                "operations": operations,
                "semantic_modifications": 0,
            }
        if "status" in slot:
            old = slot.pop("status")
            operations.append(
                {
                    "operation": "remove_deprecated_field",
                    "path": f"$.required_claim_results[{index}].status",
                    "old_value": old,
                }
            )
    return {
        "projectable": True,
        "projected_payload": projected,
        "raw_payload_hash": canonical_hash(raw),
        "projected_payload_hash": canonical_hash(projected),
        "operations": operations,
        "removed_deprecated_fields": len(operations),
        "derived_local_fields": [],
        "semantic_modifications": 0,
        "claim_text_modified": False,
        "omission_reason_modified": False,
        "answerability_modified": False,
        "slot_count_modified": False,
        "claim_ids_modified": False,
    }
