"""Frozen Stage 13.12 Dev v3.2 manifest and paths."""

from __future__ import annotations

import json
from typing import Any

from paper_research.generation.prompts import (
    QA_REQUIRED_CLAIMS_CITATION_ID_V3_2_CANDIDATE,
)

try:
    from scripts.evidence_qa_dev_lib_v1 import DATA, DEV_IDS, DOCS
    from scripts.evidence_qa_dev_v3_1_lib import (
        CAPABILITY_HASH,
        SOURCE_MANIFEST_HASH,
    )
    from scripts.evidence_qa_dev_v3_1_lib import (
        build_required_claim_input as build_v31_input,
    )
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import DATA, DEV_IDS, DOCS  # type: ignore[no-redef]
    from evidence_qa_dev_v3_1_lib import (  # type: ignore[no-redef]
        CAPABILITY_HASH,
        SOURCE_MANIFEST_HASH,
    )
    from evidence_qa_dev_v3_1_lib import (
        build_required_claim_input as build_v31_input,
    )

EVALUATION_VERSION = "evidence-qa-dev-v3.2"
PROMPT_HASH = "73df617c1ae2943a6fcecb07f096fed480392951a0c9e271ff03624284bc992c"
SCHEMA_HASH = "158504a7bd7e0fe7e32f53c81aa57e30240f3a07429447797c3ccd56d5c57a34"
POLICY_HASH = "0a381a75ede9fbfc43d6d47fc2c6b392435cb37e28e285157187839fffa6cc1b"
RUN_ROOT = DATA / "evidence-qa-dev-v3-2/runs"
MANIFEST = DATA / "evidence-qa-dev-v3-2-manifest.json"
MANIFEST_DOC = DOCS / "evidence-qa-dev-v3-2-manifest.md"
HEALTH = DATA / "provider-health-dev-v3-2-v1.json"
OUTPUT = DATA / "evidence-qa-dev-v3-2.json"
OUTPUT_CSV = DATA / "evidence-qa-dev-v3-2.csv"
OUTPUT_DOC = DOCS / "evidence-qa-dev-v3-2.md"
FINAL_AUDIT = DATA / "evidence-qa-dev-v3-2-final-audit.json"
CITATION_AUDIT = DATA / "evidence-qa-dev-v3-2-citation-audit-v1.jsonl"
CITATION_AUDIT_DOC = DOCS / "evidence-qa-dev-v3-2-citation-audit-v1.md"
PROTOCOL = DATA / "dev-v3-2-protocol-candidate-v1.json"
CLAIM_GOLD_FREEZE = DATA / "claim-evidence-gold-dev-v1-freeze.json"


def build_required_claim_input(question_id: str):
    payload, registry, contexts, trace = build_v31_input(question_id)
    payload["prompt_version"] = QA_REQUIRED_CLAIMS_CITATION_ID_V3_2_CANDIDATE
    return payload, registry, contexts, trace


def build_manifest() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    freeze = json.loads(CLAIM_GOLD_FREEZE.read_text(encoding="utf-8"))
    return {
        "schema_version": "evidence-qa-dev-v3-2-manifest-v1",
        "evaluation_version": EVALUATION_VERSION,
        "manifest_hash": SOURCE_MANIFEST_HASH,
        "question_ids": DEV_IDS,
        "question_count": 10,
        "required_claim_counts": {
            qid: len(build_required_claim_input(qid)[0]["required_claims"])
            for qid in DEV_IDS
        },
        "total_required_claims": 27,
        "configuration": {
            "collection": "papers_jina_eval34_v2__20260713152149",
            "embedding_model": "jina-embeddings-v5-text-small",
            "embedding_dimensions": 1024,
            "retrieval": "adjacent_same_page_completion",
            "prompt": QA_REQUIRED_CLAIMS_CITATION_ID_V3_2_CANDIDATE,
            "prompt_hash": PROMPT_HASH,
            "schema": "required-claim-slots-v1.1",
            "schema_hash": SCHEMA_HASH,
            "combined_policy_hash": POLICY_HASH,
            "policy_versions": protocol["policy_versions"],
            "provider_capability_snapshot_hash": CAPABILITY_HASH,
            "provider": "siliconflow",
            "model": "Qwen/Qwen3-8B",
            "temperature": 0,
            "reranker_enabled": False,
            "llm_max_retries": 0,
            "json_correction_retries": 0,
            "citation_correction_retries": 0,
            "response_format": "json_object",
            "json_schema_sent": False,
            "tools_or_functions_sent": False,
            "formal_normalization_policy": "raw_schema_passed_only",
            "citation_budget": protocol["policy_versions"]["citation_budget"],
            "billing_mode": "free",
            "monetary_cost_usd": "0",
        },
        "claim_gold_freeze_hash_evaluation_only": freeze["reviewed_file_hash"]["value"],
        "created_before_live_results": True,
        "live_authorized_for_one_batch": True,
        "full_qa_authorized": False,
        "deep_research_authorized": False,
    }


def write_manifest() -> dict[str, Any]:
    body = build_manifest()
    if MANIFEST.exists() and json.loads(MANIFEST.read_text(encoding="utf-8")) != body:
        raise RuntimeError("DEV_V3_2_CONFIGURATION_INVALID: manifest changed")
    MANIFEST.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    MANIFEST_DOC.write_text(
        "# Evidence QA Dev v3.2 Manifest\n\n"
        "- Fixed questions: 10; required claims: 27; q005 claims: 0\n"
        f"- Prompt/schema/policy hashes: `{PROMPT_HASH}` / `{SCHEMA_HASH}` / `{POLICY_HASH}`\n"
        "- Provider: SiliconFlow `Qwen/Qwen3-8B`; temperature 0; retries 0\n"
        "- Reranker disabled; response format json_object; raw schema only\n"
        "- Claim Gold freeze is evaluation-only and is absent from request payloads\n"
        "- Created before live results: true\n",
        encoding="utf-8",
    )
    return body
