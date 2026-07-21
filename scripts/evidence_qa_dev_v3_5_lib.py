"""Frozen Stage 13.19 Dev v3.5 live protocol and paths."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from paper_research.generation.citation_selection import (
    CITATION_BUDGET_VERSION,
    CITATION_SELECTION_VERSION,
    COMPARISON_VALIDATION_VERSION,
    EVIDENCE_ORIGIN_POLICY_VERSION,
    NUMERIC_VALIDATION_VERSION,
    OBLIGATION_POLICY_VERSION,
)
from paper_research.generation.schema_reliability import (
    DEV_V3_6_CANDIDATE_PROMPT_VERSION,
    LOCAL_ENVELOPE_V4_VERSION,
    MODEL_PAYLOAD_V4_VERSION,
    PAYLOAD_V4_ADAPTER,
    LocalEnvelopeV4,
    dev_v3_6_candidate_system_prompt,
)

try:
    from scripts.evidence_qa_dev_lib_v1 import DATA, DEV_IDS, DOCS, canonical_hash
    from scripts.evidence_qa_dev_v3_1_lib import CAPABILITY_HASH, SOURCE_MANIFEST_HASH
    from scripts.evidence_qa_dev_v3_3_lib import safe_model_input
    from scripts.payload_contract_v4_lib import PROTOCOL as PAYLOAD_V4_PROTOCOL
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import DATA, DEV_IDS, DOCS, canonical_hash  # type: ignore[no-redef]
    from evidence_qa_dev_v3_1_lib import (  # type: ignore[no-redef]
        CAPABILITY_HASH,
        SOURCE_MANIFEST_HASH,
    )
    from evidence_qa_dev_v3_3_lib import safe_model_input  # type: ignore[no-redef]
    from payload_contract_v4_lib import PROTOCOL as PAYLOAD_V4_PROTOCOL  # type: ignore[no-redef]

EVALUATION_VERSION = "evidence-qa-dev-v3.5"
RUN_ROOT = DATA / "evidence-qa-dev-v3-5/runs"
PROTOCOL_FREEZE = DATA / "evidence-qa-dev-v3-5-protocol-freeze-v1.json"
PROTOCOL_FREEZE_DOC = DOCS / "evidence-qa-dev-v3-5-protocol-freeze-v1.md"
PROMPT_DELIVERY_FREEZE = DATA / "evidence-qa-dev-v3-5-prompt-delivery-freeze-v1.json"
PROMPT_DELIVERY_FREEZE_DOC = DOCS / "evidence-qa-dev-v3-5-prompt-delivery-freeze-v1.md"
HEALTH = DATA / "provider-health-v1.json"
OUTPUT = DATA / "evidence-qa-dev-v3-5.json"
OUTPUT_CSV = DATA / "evidence-qa-dev-v3-5.csv"
OUTPUT_DOC = DOCS / "evidence-qa-dev-v3-5.md"
FINAL_AUDIT = DATA / "evidence-qa-dev-v3-5-final-audit.json"
FAILURE_FREEZE = DATA / "dev-v3-5-failure-freeze-v1.json"
FAILURE_FREEZE_DOC = DOCS / "dev-v3-5-failure-freeze-v1.md"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_protocol_freeze() -> dict[str, Any]:
    safe_inputs = {question_id: safe_model_input(question_id)[0] for question_id in DEV_IDS}
    payload_v4 = json.loads(PAYLOAD_V4_PROTOCOL.read_text(encoding="utf-8"))
    versions = {
        "citation_selection": CITATION_SELECTION_VERSION,
        "obligation_policy": OBLIGATION_POLICY_VERSION,
        "numeric_validator": NUMERIC_VALIDATION_VERSION,
        "comparison_validator": COMPARISON_VALIDATION_VERSION,
        "citation_budget": CITATION_BUDGET_VERSION,
        "evidence_origin": EVIDENCE_ORIGIN_POLICY_VERSION,
    }
    body = {
        "schema_version": "evidence-qa-dev-v3-5-protocol-freeze-v1",
        "evaluation_version": EVALUATION_VERSION,
        "fixed_manifest_hash": SOURCE_MANIFEST_HASH,
        "question_ids": DEV_IDS,
        "question_count": 10,
        "answerable_questions": 9,
        "required_claims": 27,
        "q005_required_claims": 0,
        "required_claim_input_hash": canonical_hash(safe_inputs),
        "payload_contract_v4_protocol_signature": payload_v4["protocol_signature"],
        "prompt_version": DEV_V3_6_CANDIDATE_PROMPT_VERSION,
        "prompt_template_hash": canonical_hash(dev_v3_6_candidate_system_prompt()),
        "model_payload_schema_version": MODEL_PAYLOAD_V4_VERSION,
        "model_payload_schema_hash": canonical_hash(PAYLOAD_V4_ADAPTER.json_schema()),
        "local_envelope_schema_version": LOCAL_ENVELOPE_V4_VERSION,
        "local_envelope_schema_hash": canonical_hash(LocalEnvelopeV4.model_json_schema()),
        "transport": {
            "response_format": {"type": "json_object"},
            "json_schema": False,
            "tool_calling": False,
            "stream": False,
        },
        "provider": "siliconflow",
        "model": "Qwen/Qwen3-8B",
        "temperature": 0,
        "retry_policy": {"provider": 0, "json": 0, "citation": 0},
        "canonicalization": "none",
        "normalization": "none",
        "repair": "none",
        "fallback": "none",
        "reranker_enabled": False,
        "gold_used_online": False,
        "human_labels_used_online": False,
        "policy_versions": versions,
        "policy_hashes": {key: canonical_hash(value) for key, value in versions.items()},
        "provider_capability_snapshot_hash": CAPABILITY_HASH,
        "billing": "explicit_free_provider",
        "monetary_cost_usd": "0",
        "frozen_before_live": True,
        "next_live_authorized": False,
    }
    body["protocol_freeze_signature"] = canonical_hash(body)
    return body


def write_protocol_freeze() -> dict[str, Any]:
    body = build_protocol_freeze()
    if PROTOCOL_FREEZE.exists():
        existing = json.loads(PROTOCOL_FREEZE.read_text(encoding="utf-8"))
        if existing != body:
            raise RuntimeError("DEV_V3_5_PROTOCOL_FREEZE_DRIFT")
    else:
        PROTOCOL_FREEZE.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
        PROTOCOL_FREEZE_DOC.write_text(
            "# Evidence QA Dev v3.5 Protocol Freeze\n\n"
            f"- Signature: `{body['protocol_freeze_signature']}`\n"
            f"- Payload v4 protocol: `{body['payload_contract_v4_protocol_signature']}`\n"
            f"- Prompt: `{body['prompt_version']}` / `{body['prompt_template_hash']}`\n"
            f"- Payload/envelope schema hashes: `{body['model_payload_schema_hash']}` / "
            f"`{body['local_envelope_schema_hash']}`\n"
            "- No normalization, JSON repair, retry, citation repair, fallback, Reranker, "
            "Gold injection, or human-label injection.\n",
            encoding="utf-8",
        )
    return body


def build_prompt_delivery_freeze() -> dict[str, Any]:
    protocol = write_protocol_freeze()
    rows = []
    for question_id in DEV_IDS:
        safe = safe_model_input(question_id)[0]
        system = dev_v3_6_candidate_system_prompt()
        user = json.dumps(safe, ensure_ascii=False)
        request_body = {
            "model": protocol["model"],
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_tokens": safe["output_budget"]["calculated_max_output_tokens"],
            "stream": False,
            "enable_thinking": False,
            "response_format": {"type": "json_object"},
        }
        rows.append(
            {
                "question_id": question_id,
                "delivered_system_prompt_hash": canonical_hash(system),
                "delivered_user_payload_hash": canonical_hash(safe),
                "delivered_messages_hash": canonical_hash(request_body["messages"]),
                "delivered_request_body_hash": canonical_hash(request_body),
                "schema_hash": protocol["model_payload_schema_hash"],
                "protocol_signature": protocol["protocol_freeze_signature"],
                "payload_contract_v4_protocol_signature": protocol[
                    "payload_contract_v4_protocol_signature"
                ],
            }
        )
    body = {
        "schema_version": "evidence-qa-dev-v3-5-prompt-delivery-freeze-v1",
        "evaluation_version": EVALUATION_VERSION,
        "prompt_version": protocol["prompt_version"],
        "prompt_hash": protocol["prompt_template_hash"],
        "schema_hash": protocol["model_payload_schema_hash"],
        "protocol_signature": protocol["protocol_freeze_signature"],
        "payload_contract_v4_protocol_signature": protocol[
            "payload_contract_v4_protocol_signature"
        ],
        "question_count": len(rows),
        "questions": rows,
        "old_prompt_mixed_in": False,
        "payload_schema_mismatch": False,
    }
    body["prompt_delivery_signature"] = canonical_hash(body)
    return body


def write_prompt_delivery_freeze() -> dict[str, Any]:
    body = build_prompt_delivery_freeze()
    if PROMPT_DELIVERY_FREEZE.exists():
        existing = json.loads(PROMPT_DELIVERY_FREEZE.read_text(encoding="utf-8"))
        if existing != body:
            raise RuntimeError("DEV_V3_5_PROMPT_DELIVERY_FREEZE_DRIFT")
    else:
        PROMPT_DELIVERY_FREEZE.write_text(
            json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        PROMPT_DELIVERY_FREEZE_DOC.write_text(
            "# Evidence QA Dev v3.5 Prompt Delivery Freeze\n\n"
            f"- Signature: `{body['prompt_delivery_signature']}`\n"
            f"- Prompt hash: `{body['prompt_hash']}`\n"
            f"- Schema hash: `{body['schema_hash']}`\n"
            f"- Protocol signature: `{body['protocol_signature']}`\n"
            "- v3.1/v3.2 prompt mix-in: false\n"
            "- Payload schema mismatch: false\n",
            encoding="utf-8",
        )
    return body
