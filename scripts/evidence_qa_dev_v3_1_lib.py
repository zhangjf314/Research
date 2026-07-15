"""Frozen Dev v3.1 manifest and input helpers."""

from __future__ import annotations

import json
from typing import Any

from paper_research.generation.prompts import QA_REQUIRED_CLAIMS_CITATION_ID_V3_1

try:
    from scripts.evidence_qa_dev_lib_v1 import DATA, DEV_IDS, DOCS
    from scripts.evidence_qa_dev_v3_lib import (
        SOURCE_MANIFEST_HASH,
    )
    from scripts.evidence_qa_dev_v3_lib import (
        build_required_claim_input as build_v3_input,
    )
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import DATA, DEV_IDS, DOCS  # type: ignore[no-redef]
    from evidence_qa_dev_v3_lib import (  # type: ignore[no-redef]
        SOURCE_MANIFEST_HASH,
    )
    from evidence_qa_dev_v3_lib import (
        build_required_claim_input as build_v3_input,
    )

EVALUATION_VERSION = "evidence-qa-dev-v3.1"
PROMPT_HASH = "071098990460a8fdffb9c9a168b13ebedcf3eced8bcea5ed3ac4bcd2db0f11bf"
SCHEMA_HASH = "158504a7bd7e0fe7e32f53c81aa57e30240f3a07429447797c3ccd56d5c57a34"
CAPABILITY_HASH = "000bf54a298e21db53f42419123a53f16a05c279f002d8336146175ac95e7a03"
SOURCE_PROTOCOL_HASH = "ba7a8d5a132ca6c201e835ed2f66583f91598a0d2c1c6c1c0920185714502552"
MANIFEST = DATA / "evidence-qa-dev-v3-1-manifest.json"
MANIFEST_DOC = DOCS / "evidence-qa-dev-v3-1-manifest.md"
RUN_ROOT = DATA / "evidence-qa-dev-v3-1/runs"
HEALTH = DATA / "provider-health-dev-v3-1-v1.json"
OUTPUT = DATA / "evidence-qa-dev-v3-1.json"
OUTPUT_CSV = DATA / "evidence-qa-dev-v3-1.csv"
OUTPUT_DOC = DOCS / "evidence-qa-dev-v3-1.md"
CITATION_AUDIT = DATA / "evidence-qa-dev-v3-1-citation-audit-v1.jsonl"
CITATION_AUDIT_DOC = DOCS / "evidence-qa-dev-v3-1-citation-audit-v1.md"
FINAL_AUDIT = DATA / "evidence-qa-dev-v3-1-final-audit.json"


def build_required_claim_input(question_id: str):
    payload, registry, contexts, trace = build_v3_input(question_id)
    payload["prompt_version"] = QA_REQUIRED_CLAIMS_CITATION_ID_V3_1
    payload["required_claim_protocol"] = "required-claim-slots-v1.1"
    return payload, registry, contexts, trace


def required_claim_counts() -> dict[str, int]:
    return {
        question_id: len(build_required_claim_input(question_id)[0]["required_claims"])
        for question_id in DEV_IDS
    }


def build_manifest() -> dict[str, Any]:
    counts = required_claim_counts()
    return {
        "schema_version": "evidence-qa-dev-v3-1-manifest-v1",
        "evaluation_version": EVALUATION_VERSION,
        "manifest_hash": SOURCE_MANIFEST_HASH,
        "source_protocol_hash": SOURCE_PROTOCOL_HASH,
        "question_ids": DEV_IDS,
        "question_count": 10,
        "required_claim_counts": counts,
        "total_required_claims": sum(counts.values()),
        "configuration": {
            "collection": "papers_jina_eval34_v2__20260713152149",
            "embedding_model": "jina-embeddings-v5-text-small",
            "embedding_dimensions": 1024,
            "retrieval": "adjacent_same_page_completion",
            "prompt": QA_REQUIRED_CLAIMS_CITATION_ID_V3_1,
            "prompt_hash": PROMPT_HASH,
            "schema_hash": SCHEMA_HASH,
            "provider_capability_snapshot_hash": CAPABILITY_HASH,
            "citation_protocol": "citation-id-v2",
            "required_claim_protocol": "required-claim-slots-v1.1",
            "transport": "provider_json_object_plus_strict_local_schema",
            "response_format": "json_object",
            "formal_normalization_policy": "raw_schema_passed_only",
            "provider": "siliconflow",
            "model": "Qwen/Qwen3-8B",
            "temperature": 0,
            "reranker_enabled": False,
            "llm_max_retries": 0,
            "json_correction_retries": 0,
            "citation_correction_retries": 0,
            "billing_mode": "free",
            "monetary_cost_usd": "0",
        },
        "request_budgets": {
            "per_question_requests_max": 1,
            "per_question_tokens_max": 24000,
            "per_question_elapsed_seconds_max": 180,
            "global_requests_max": 10,
            "global_tokens_max": 240000,
            "global_elapsed_seconds_max": 1800,
        },
        "engineering_gate_frozen": {
            "provider_completed_min": 9,
            "raw_json_valid_rate_min": 0.9,
            "raw_schema_success_rate_min": 0.9,
            "slot_success_rate_min": 0.9,
            "silent_omission_rate": 0,
            "strict_hashes_and_ledger": True,
            "no_retries_or_leakage": True,
        },
        "quality_candidate_gate_frozen": {
            "required_claim_coverage_numerator_min": 17,
            "required_claim_coverage_strictly_greater_than": 0.592593,
            "exact_citation_precision_min": 0.181731,
            "citation_recall_min": 0.295833,
            "unsupported_claim_rate_strictly_less_than": 0.8,
            "refusal_accuracy": 1.0,
            "unknown_invalid_cross_claim_rate": 0,
            "silent_omission_rate": 0,
            "non_regressed_questions_min": 6,
            "improved_questions_greater_than_regressed": True,
            "claim_coverage_improved_questions_min": 3,
            "focus_questions_improved_min": 2,
            "focus_questions": ["q002", "q007", "q013", "q050"],
            "q019_independent_slots": True,
            "q005_correct_refusal": True,
        },
        "created_before_live_results": True,
        "questions_reselected": False,
        "gold_oracle_pilot_evidence_injected": False,
    }


def write_manifest() -> dict[str, Any]:
    body = build_manifest()
    if MANIFEST.exists() and json.loads(MANIFEST.read_text(encoding="utf-8")) != body:
        raise RuntimeError("frozen Dev v3.1 manifest changed")
    MANIFEST.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    MANIFEST_DOC.write_text(
        "# Evidence QA Dev v3.1 Manifest\n\n"
        "- Questions: `q001,q002,q004,q005,q007,q008,q013,q015,q019,q050`\n"
        f"- Manifest hash: `{SOURCE_MANIFEST_HASH}`\n"
        "- Required claims: 27\n"
        f"- Prompt/schema/capability hashes: `{PROMPT_HASH}` / `{SCHEMA_HASH}` / "
        f"`{CAPABILITY_HASH}`\n"
        "- Transport: `json_object` plus strict local v3.1 schema\n"
        "- Formal normalization: disabled (`raw_schema_passed_only`)\n"
        "- Reranker/retries: disabled / 0\n"
        "- Frozen before live results: true\n",
        encoding="utf-8",
    )
    return body
