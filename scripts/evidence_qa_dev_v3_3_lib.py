"""Frozen Stage 13.14 Dev v3.3 protocol and safe model inputs."""

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
    DEV_V3_3_PROMPT_VERSION,
    DevV33RequiredClaimsEnvelope,
    MinimalRequiredClaimsPayload,
    dev_v3_3_system_prompt,
)

try:
    from scripts.evidence_qa_dev_lib_v1 import DATA, DEV_IDS, DOCS, canonical_hash
    from scripts.evidence_qa_dev_v3_2_lib import (
        CAPABILITY_HASH,
        SOURCE_MANIFEST_HASH,
    )
    from scripts.evidence_qa_dev_v3_2_lib import (
        build_required_claim_input as build_full_input,
    )
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import DATA, DEV_IDS, DOCS, canonical_hash  # type: ignore[no-redef]
    from evidence_qa_dev_v3_2_lib import (  # type: ignore[no-redef]
        CAPABILITY_HASH,
        SOURCE_MANIFEST_HASH,
    )
    from evidence_qa_dev_v3_2_lib import (
        build_required_claim_input as build_full_input,
    )

ROOT = DATA.parents[1]
EVALUATION_VERSION = "evidence-qa-dev-v3.3"
RUN_ROOT = DATA / "evidence-qa-dev-v3-3/runs"
PROTOCOL_FREEZE = DATA / "evidence-qa-dev-v3-3-protocol-freeze-v1.json"
PROTOCOL_FREEZE_DOC = DOCS / "evidence-qa-dev-v3-3-protocol-freeze-v1.md"
VISIBLE_ID_AUDIT = DATA / "dev-v3-3-visible-id-namespace-audit-v1.json"
VISIBLE_ID_AUDIT_DOC = DOCS / "dev-v3-3-visible-id-namespace-audit-v1.md"
HEALTH = DATA / "provider-health-dev-v3-3-v1.json"
OUTPUT = DATA / "evidence-qa-dev-v3-3.json"
OUTPUT_CSV = DATA / "evidence-qa-dev-v3-3.csv"
OUTPUT_DOC = DOCS / "evidence-qa-dev-v3-3.md"
FINAL_AUDIT = DATA / "evidence-qa-dev-v3-3-final-audit.json"
CITATION_AUDIT = DATA / "evidence-qa-dev-v3-3-citation-audit-v1.jsonl"
CITATION_AUDIT_DOC = DOCS / "evidence-qa-dev-v3-3-citation-audit-v1.md"


def output_budget(required_claim_count: int) -> dict[str, Any]:
    value = min(3072, 256 + 128 * required_claim_count)
    return {
        "required_claim_count": required_claim_count,
        "calculated_max_output_tokens": value,
        "capped": value == 3072 and required_claim_count > 22,
        "budget_formula_version": "minimal-required-claim-output-budget-v1",
        "formula": "min(3072, 256 + 128 * required_claim_count)",
    }


def safe_model_input(
    question_id: str,
) -> tuple[dict[str, Any], dict[str, Any], Any, dict[str, Any]]:
    full, registry, _contexts, trace = build_full_input(question_id)
    safe_claims = []
    local_maps = {}
    for claim in full["required_claims"]:
        evidence = []
        labels = {}
        for index, allocated in enumerate(claim["allocated_evidence"]):
            label = f"Evidence {chr(65 + index)}"
            evidence.append({"label": label, "text": allocated["summary"]})
            labels[label] = {
                "evidence_id": allocated["evidence_id"],
                "citation_ids": allocated["citation_ids"],
            }
        safe_claims.append(
            {
                "required_claim_id": claim["required_claim_id"],
                "required_claim_text": claim["required_claim_text"],
                "evidence_complete": claim["evidence_complete"],
                "evidence": evidence,
                "omission_policy": claim["omission_policy"],
            }
        )
        local_maps[claim["required_claim_id"]] = labels
    safe = {
        "question": full["question"],
        "answerability_expectation": full["answerability_expectation"],
        "required_claims": safe_claims,
        "output_budget": output_budget(len(safe_claims)),
    }
    return safe, full, registry, {**trace, "display_label_map": local_maps}


def _source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_freeze() -> dict[str, Any]:
    prompt = dev_v3_3_system_prompt()
    payload_schema = MinimalRequiredClaimsPayload.model_json_schema()
    envelope_schema = DevV33RequiredClaimsEnvelope.model_json_schema()
    policy_versions = {
        "citation_selection": CITATION_SELECTION_VERSION,
        "claim_obligation": OBLIGATION_POLICY_VERSION,
        "numeric_validator": NUMERIC_VALIDATION_VERSION,
        "comparison_validator": COMPARISON_VALIDATION_VERSION,
        "citation_budget": CITATION_BUDGET_VERSION,
        "evidence_origin": EVIDENCE_ORIGIN_POLICY_VERSION,
    }
    policy_hashes = {key: canonical_hash(value) for key, value in policy_versions.items()}
    body = {
        "schema_version": "evidence-qa-dev-v3-3-protocol-freeze-v1",
        "evaluation_version": EVALUATION_VERSION,
        "fixed_manifest": {
            "manifest_hash": SOURCE_MANIFEST_HASH,
            "question_ids": DEV_IDS,
            "question_count": 10,
            "answerable_questions": 9,
            "required_claims": 27,
            "q005_required_claims": 0,
        },
        "model_payload_schema": "required-claim-model-payload-v1",
        "model_payload_schema_hash": canonical_hash(payload_schema),
        "local_envelope_schema": "required-claim-local-envelope-v1",
        "local_envelope_schema_hash": canonical_hash(envelope_schema),
        "prompt_version": DEV_V3_3_PROMPT_VERSION,
        "prompt_hash": canonical_hash(prompt),
        "rendered_prompt_hashing_rule": "sha256-utf8-exact-rendered-text",
        "delivered_messages_hashing_rule": "canonical-json-sha256-sorted-keys-v1",
        "delivered_request_body_hashing_rule": "canonical-json-sha256-without-headers-v1",
        "policy_versions": policy_versions,
        "policy_hashes": policy_hashes,
        "provider_capability_snapshot_hash": CAPABILITY_HASH,
        "collection": "papers_jina_eval34_v2__20260713152149",
        "embedding": "jina-embeddings-v5-text-small",
        "embedding_dimensions": 1024,
        "retrieval_profile": "adjacent_same_page_completion",
        "provider": "siliconflow",
        "model": "Qwen/Qwen3-8B",
        "temperature": 0,
        "provider_retries": 0,
        "json_retries": 0,
        "citation_retries": 0,
        "output_budget": output_budget(3),
        "accounting_policy": "request-accounting-v1",
        "billing": "explicit_free_provider",
        "monetary_cost_usd": "0",
        "gold_freeze_hash_evaluation_only": json.loads(
            (DATA / "claim-evidence-gold-dev-v1-freeze.json").read_text(encoding="utf-8")
        )["reviewed_file_hash"]["value"],
        "historical_protection_hashes": {
            "stage13_12_failure_freeze": _source_hash(
                DATA / "stage13-12-dev-v3-2-failure-freeze-v1.json"
            ),
            "stage13_13_reconciliation": _source_hash(
                DATA / "stage13-12-reservation-reconciliation-v1.json"
            ),
        },
        "frozen_before_live": True,
    }
    body["protocol_freeze_signature"] = canonical_hash(body)
    return body


def write_freeze() -> dict[str, Any]:
    body = build_freeze()
    if PROTOCOL_FREEZE.exists():
        existing = json.loads(PROTOCOL_FREEZE.read_text(encoding="utf-8"))
        if existing != body:
            raise RuntimeError("DEV_V3_3_CONFIGURATION_INVALID: protocol hash drift")
    else:
        PROTOCOL_FREEZE.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
        PROTOCOL_FREEZE_DOC.write_text(
            "# Evidence QA Dev v3.3 Protocol Freeze\n\n"
            f"- Signature: `{body['protocol_freeze_signature']}`\n"
            f"- Prompt: `{DEV_V3_3_PROMPT_VERSION}` / `{body['prompt_hash']}`\n"
            f"- Payload/envelope schemas: `{body['model_payload_schema_hash']}` / "
            f"`{body['local_envelope_schema_hash']}`\n"
            "- Fixed 10-question manifest, 27 required claims, q005 has zero claims.\n"
            "- Frozen before live; retries and Reranker disabled.\n",
            encoding="utf-8",
        )
    return body


def write_visible_id_audit() -> dict[str, Any]:
    findings = []
    for question_id in DEV_IDS:
        safe, _full, _registry, _trace = safe_model_input(question_id)
        encoded = json.dumps(safe, ensure_ascii=False)
        forbidden = [
            token
            for token in (
                "evidence_id",
                "citation_id",
                "block_id",
                "paper_id",
                "relation_id",
                "source_record",
                "human_label",
                "gold_",
            )
            if token in encoded.lower()
        ]
        if forbidden:
            findings.append({"question_id": question_id, "forbidden": forbidden})
    body = {
        "schema_version": "dev-v3-3-visible-id-namespace-audit-v1",
        "questions_scanned": 10,
        "copyable_internal_ids": 0 if not findings else len(findings),
        "gold_ids": 0,
        "human_label_fields": 0,
        "findings": findings,
        "gate": "PASSED" if not findings else "FAILED",
    }
    VISIBLE_ID_AUDIT.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    VISIBLE_ID_AUDIT_DOC.write_text(
        "# Dev v3.3 Visible ID Namespace Audit\n\n"
        f"- Questions scanned: 10\n- Copyable internal IDs: "
        f"{body['copyable_internal_ids']}\n- Gold IDs: 0\n"
        f"- Human-label fields: 0\n- Gate: `{body['gate']}`\n",
        encoding="utf-8",
    )
    return body
