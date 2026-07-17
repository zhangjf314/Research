"""Frozen Stage 13.21 Dev v3.6 protocol and paths."""

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
    DEV_V3_7_CANDIDATE_PROMPT_VERSION,
    LOCAL_ENVELOPE_V4_VERSION,
    MODEL_PAYLOAD_V4_VERSION,
    PAYLOAD_V4_ADAPTER,
    LocalEnvelopeV4,
    dev_v3_7_candidate_system_prompt,
)

try:
    from scripts.evidence_presentation_v2_lib import (
        PRESENTATION_VERSION,
        SELECTED_FORMAT,
        rendered_messages,
    )
    from scripts.evidence_presentation_v2_lib import (
        PROTOCOL as PRESENTATION_PROTOCOL,
    )
    from scripts.evidence_presentation_v2_lib import (
        build_protocol as build_presentation_protocol,
    )
    from scripts.evidence_qa_dev_lib_v1 import DATA, DEV_IDS, DOCS, canonical_hash
    from scripts.evidence_qa_dev_v3_1_lib import CAPABILITY_HASH, SOURCE_MANIFEST_HASH
    from scripts.evidence_qa_dev_v3_3_lib import output_budget
except ModuleNotFoundError:
    from evidence_presentation_v2_lib import (  # type: ignore[no-redef]
        PRESENTATION_VERSION,
        SELECTED_FORMAT,
        rendered_messages,
    )
    from evidence_presentation_v2_lib import (
        PROTOCOL as PRESENTATION_PROTOCOL,
    )
    from evidence_presentation_v2_lib import (
        build_protocol as build_presentation_protocol,
    )
    from evidence_qa_dev_lib_v1 import DATA, DEV_IDS, DOCS, canonical_hash  # type: ignore[no-redef]
    from evidence_qa_dev_v3_1_lib import (  # type: ignore[no-redef]
        CAPABILITY_HASH,
        SOURCE_MANIFEST_HASH,
    )
    from evidence_qa_dev_v3_3_lib import output_budget  # type: ignore[no-redef]

EVALUATION_VERSION = "evidence-qa-dev-v3.6"
RUN_ROOT = DATA / "evidence-qa-dev-v3-6/runs"
PROTOCOL_FREEZE = DATA / "evidence-qa-dev-v3-6-protocol-freeze-v1.json"
PROTOCOL_FREEZE_DOC = DOCS / "evidence-qa-dev-v3-6-protocol-freeze-v1.md"
HEALTH = DATA / "provider-health-dev-v3-6-v1.json"
OUTPUT = DATA / "evidence-qa-dev-v3-6.json"
OUTPUT_CSV = DATA / "evidence-qa-dev-v3-6.csv"
OUTPUT_DOC = DOCS / "evidence-qa-dev-v3-6.md"
FINAL_AUDIT = DATA / "evidence-qa-dev-v3-6-final-audit.json"
CITATION_AUDIT = DATA / "evidence-qa-dev-v3-6-citation-audit-v1.jsonl"
CITATION_AUDIT_DOC = DOCS / "evidence-qa-dev-v3-6-citation-audit-v1.md"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_protocol_freeze() -> dict[str, Any]:
    presentation = build_presentation_protocol()
    if PRESENTATION_PROTOCOL.exists():
        existing = json.loads(PRESENTATION_PROTOCOL.read_text(encoding="utf-8"))
        if existing != presentation:
            raise RuntimeError("DEV_V3_6_CONFIGURATION_INVALID: presentation drift")
    safe_inputs = {question_id: rendered_messages(question_id)[1] for question_id in DEV_IDS}
    versions = {
        "citation_selection": CITATION_SELECTION_VERSION,
        "obligation_policy": OBLIGATION_POLICY_VERSION,
        "numeric_validator": NUMERIC_VALIDATION_VERSION,
        "comparison_validator": COMPARISON_VALIDATION_VERSION,
        "citation_budget": CITATION_BUDGET_VERSION,
        "evidence_origin": EVIDENCE_ORIGIN_POLICY_VERSION,
    }
    body = {
        "schema_version": "evidence-qa-dev-v3-6-protocol-freeze-v1",
        "evaluation_version": EVALUATION_VERSION,
        "fixed_manifest_hash": SOURCE_MANIFEST_HASH,
        "question_ids": DEV_IDS,
        "question_count": 10,
        "answerable_questions": 9,
        "required_claims": 27,
        "q005_required_claims": 0,
        "required_claim_input_hash": canonical_hash(safe_inputs),
        "prompt_version": DEV_V3_7_CANDIDATE_PROMPT_VERSION,
        "prompt_hash": canonical_hash(dev_v3_7_candidate_system_prompt()),
        "payload_v4_version": MODEL_PAYLOAD_V4_VERSION,
        "payload_v4_hash": canonical_hash(PAYLOAD_V4_ADAPTER.json_schema()),
        "envelope_v4_version": LOCAL_ENVELOPE_V4_VERSION,
        "envelope_v4_hash": canonical_hash(LocalEnvelopeV4.model_json_schema()),
        "evidence_presentation_version": PRESENTATION_VERSION,
        "evidence_presentation_hash": presentation["protocol_signature"],
        "selected_rendering_format": SELECTED_FORMAT,
        "rendered_prompt_hash_rule": "canonical-json-sha256-sorted-keys-v1",
        "exact_delivered_request_hash_rule": "canonical-json-sha256-sorted-keys-v1",
        "citation_registry_hash_rule": "registry.registry_hash",
        "candidate_evidence_hash_rule": "canonical-json-sha256-sorted-keys-v1",
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
            "repair": 0,
        },
        "output_budget": {
            "formula": "min(3072, 256 + 128 * required_claim_count)",
            "three_slot_tokens": output_budget(3)["calculated_max_output_tokens"],
            "q005_tokens": output_budget(0)["calculated_max_output_tokens"],
        },
        "accounting_policy": "request-accounting-v1",
        "billing": "explicit_free_provider",
        "monetary_cost_usd": "0",
        "claim_gold_freeze_hash_evaluation_only": json.loads(
            (DATA / "claim-evidence-gold-dev-v1-freeze.json").read_text(encoding="utf-8")
        )["reviewed_file_hash"]["value"],
        "frozen_before_live": True,
        "historical_results_immutable": True,
    }
    body["protocol_freeze_signature"] = canonical_hash(body)
    return body


def write_protocol_freeze() -> dict[str, Any]:
    first = build_protocol_freeze()
    second = build_protocol_freeze()
    if first != second:
        raise RuntimeError("DEV_V3_6_CONFIGURATION_INVALID: nondeterministic freeze")
    if PROTOCOL_FREEZE.exists():
        existing = json.loads(PROTOCOL_FREEZE.read_text(encoding="utf-8"))
        if existing != first:
            raise RuntimeError("DEV_V3_6_CONFIGURATION_INVALID: freeze drift")
    else:
        PROTOCOL_FREEZE.write_text(
            json.dumps(first, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        PROTOCOL_FREEZE_DOC.write_text(
            "# Evidence QA Dev v3.6 Protocol Freeze\n\n"
            f"- Signature: `{first['protocol_freeze_signature']}`\n"
            f"- Prompt: `{first['prompt_version']}` / `{first['prompt_hash']}`\n"
            f"- Payload v4: `{first['payload_v4_hash']}`\n"
            f"- Envelope v4: `{first['envelope_v4_hash']}`\n"
            f"- Evidence presentation: `{first['evidence_presentation_version']}` / "
            f"`{first['evidence_presentation_hash']}`\n"
            "- Frozen before live; historical results immutable.\n",
            encoding="utf-8",
        )
    return first
