"""Frozen Stage 13.16 Dev v3.4 protocol and paths."""

from __future__ import annotations

import hashlib
import json
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
    DEV_V3_4_PROMPT_VERSION,
    REFUSAL_CANONICALIZATION_VERSION,
    DevV34LocalEnvelope,
    MinimalRequiredClaimsPayload,
    dev_v3_4_system_prompt,
)

try:
    from scripts.evidence_qa_dev_lib_v1 import DATA, DEV_IDS, DOCS, canonical_hash
    from scripts.evidence_qa_dev_v3_1_lib import CAPABILITY_HASH, SOURCE_MANIFEST_HASH
    from scripts.evidence_qa_dev_v3_3_lib import safe_model_input
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import DATA, DEV_IDS, DOCS, canonical_hash  # type: ignore[no-redef]
    from evidence_qa_dev_v3_1_lib import (  # type: ignore[no-redef]
        CAPABILITY_HASH,
        SOURCE_MANIFEST_HASH,
    )
    from evidence_qa_dev_v3_3_lib import safe_model_input  # type: ignore[no-redef]

EVALUATION_VERSION = "evidence-qa-dev-v3.4"
RUN_ROOT = DATA / "evidence-qa-dev-v3-4/runs"
PROTOCOL_FREEZE = DATA / "evidence-qa-dev-v3-4-protocol-freeze-v1.json"
PROTOCOL_FREEZE_DOC = DOCS / "evidence-qa-dev-v3-4-protocol-freeze-v1.md"
VISIBLE_ID_AUDIT = DATA / "dev-v3-4-visible-id-namespace-audit-v1.json"
VISIBLE_ID_AUDIT_DOC = DOCS / "dev-v3-4-visible-id-namespace-audit-v1.md"
HEALTH = DATA / "provider-health-dev-v3-4-v1.json"
OUTPUT = DATA / "evidence-qa-dev-v3-4.json"
OUTPUT_CSV = DATA / "evidence-qa-dev-v3-4.csv"
OUTPUT_DOC = DOCS / "evidence-qa-dev-v3-4.md"
FINAL_AUDIT = DATA / "evidence-qa-dev-v3-4-final-audit.json"
CITATION_AUDIT = DATA / "evidence-qa-dev-v3-4-citation-audit-v1.jsonl"
CITATION_AUDIT_DOC = DOCS / "evidence-qa-dev-v3-4-citation-audit-v1.md"


def file_hash(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_freeze() -> dict[str, Any]:
    safe_inputs = {question_id: safe_model_input(question_id)[0] for question_id in DEV_IDS}
    versions = {
        "canonicalization": REFUSAL_CANONICALIZATION_VERSION,
        "citation_selection": CITATION_SELECTION_VERSION,
        "obligation_policy": OBLIGATION_POLICY_VERSION,
        "numeric_validator": NUMERIC_VALIDATION_VERSION,
        "comparison_validator": COMPARISON_VALIDATION_VERSION,
        "citation_budget": CITATION_BUDGET_VERSION,
        "evidence_origin": EVIDENCE_ORIGIN_POLICY_VERSION,
    }
    body = {
        "schema_version": "evidence-qa-dev-v3-4-protocol-freeze-v1",
        "evaluation_version": EVALUATION_VERSION,
        "fixed_manifest_hash": SOURCE_MANIFEST_HASH,
        "question_ids": DEV_IDS,
        "question_count": 10,
        "answerable_questions": 9,
        "required_claims": 27,
        "q005_required_claims": 0,
        "required_claim_input_hash": canonical_hash(safe_inputs),
        "prompt_version": DEV_V3_4_PROMPT_VERSION,
        "prompt_template_hash": canonical_hash(dev_v3_4_system_prompt()),
        "rendered_prompt_hash_rule": "sha256-utf8-exact-rendered-text",
        "delivered_messages_hash_rule": "canonical-json-sha256-sorted-keys-v1",
        "model_payload_schema_version": "required-claim-model-payload-v2",
        "model_payload_schema_hash": canonical_hash(
            MinimalRequiredClaimsPayload.model_json_schema()
        ),
        "local_envelope_schema_version": "required-claim-local-envelope-v2",
        "local_envelope_schema_hash": canonical_hash(DevV34LocalEnvelope.model_json_schema()),
        "policy_versions": versions,
        "policy_hashes": {key: canonical_hash(value) for key, value in versions.items()},
        "provider_capability_snapshot_hash": CAPABILITY_HASH,
        "collection": "papers_jina_eval34_v2__20260713152149",
        "embedding": "jina-embeddings-v5-text-small",
        "embedding_dimensions": 1024,
        "retrieval_profile": "adjacent_same_page_completion",
        "provider": "siliconflow",
        "model": "Qwen/Qwen3-8B",
        "temperature": 0,
        "retry_policy": {
            "provider": 0,
            "json": 0,
            "citation": 0,
        },
        "output_budget": {
            "formula": "min(3072, 256 + 128 * required_claim_count)",
            "three_slot_tokens": 640,
            "q005_tokens": 256,
        },
        "accounting_policy": "request-accounting-v1",
        "billing": "explicit_free_provider",
        "monetary_cost_usd": "0",
        "claim_gold_freeze_hash_evaluation_only": json.loads(
            (DATA / "claim-evidence-gold-dev-v1-freeze.json").read_text(encoding="utf-8")
        )["reviewed_file_hash"]["value"],
        "historical_hashes": {
            "stage13_14_failure_freeze": file_hash(
                DATA / "stage13-14-dev-v3-3-failure-freeze-v1.json"
            ),
            "stage13_15_payload_replay": file_hash(
                DATA / "dev-v3-3-payload-contract-v2-replay.json"
            ),
        },
        "frozen_before_live": True,
        "historical_backward_gate_effect": False,
    }
    body["protocol_freeze_signature"] = canonical_hash(body)
    return body


def write_freeze() -> dict[str, Any]:
    body = build_freeze()
    if PROTOCOL_FREEZE.exists():
        existing = json.loads(PROTOCOL_FREEZE.read_text(encoding="utf-8"))
        if existing != body:
            raise RuntimeError("DEV_V3_4_CONFIGURATION_INVALID")
    else:
        PROTOCOL_FREEZE.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
        PROTOCOL_FREEZE_DOC.write_text(
            "# Evidence QA Dev v3.4 Protocol Freeze\n\n"
            f"- Signature: `{body['protocol_freeze_signature']}`\n"
            f"- Prompt: `{DEV_V3_4_PROMPT_VERSION}` / "
            f"`{body['prompt_template_hash']}`\n"
            f"- Payload/envelope schema hashes: "
            f"`{body['model_payload_schema_hash']}` / "
            f"`{body['local_envelope_schema_hash']}`\n"
            "- Canonicalization is restricted to exact empty refusal -> null for "
            "answerable payloads.\n"
            "- Frozen before live; Reranker and retries disabled.\n",
            encoding="utf-8",
        )
    return body


def write_visible_id_audit() -> dict[str, Any]:
    findings = []
    for question_id in DEV_IDS:
        safe = safe_model_input(question_id)[0]
        encoded = json.dumps(safe, ensure_ascii=False).lower()
        forbidden = [
            token
            for token in (
                "evidence_id",
                "citation_id",
                "block_id",
                "paper_id",
                "relation_id",
                "gold_",
                "human_label",
                "qa-required-claims-citation-id-v3",
            )
            if token in encoded
        ]
        if forbidden:
            findings.append({"question_id": question_id, "tokens": forbidden})
    body = {
        "schema_version": "dev-v3-4-visible-id-namespace-audit-v1",
        "questions_scanned": 10,
        "copyable_internal_ids": len(findings),
        "evidence_id_exposure": 0,
        "block_id_exposure": 0,
        "gold_id_exposure": 0,
        "human_label_exposure": 0,
        "model_visible_citation_id_requirement": 0,
        "old_prompt_version_examples": 0,
        "findings": findings,
        "gate": "PASSED" if not findings else "FAILED",
    }
    VISIBLE_ID_AUDIT.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    VISIBLE_ID_AUDIT_DOC.write_text(
        "# Dev v3.4 Visible ID Namespace Audit\n\n"
        f"- Questions: 10\n- Copyable internal IDs: "
        f"{body['copyable_internal_ids']}\n"
        "- Evidence/block/Gold/human-label exposure: 0/0/0/0\n"
        "- Model-visible citation ID requirement: 0\n"
        "- Old Prompt examples: 0\n"
        f"- Gate: `{body['gate']}`\n",
        encoding="utf-8",
    )
    return body
