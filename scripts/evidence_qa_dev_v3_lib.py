# ruff: noqa: E501
"""Frozen Dev v3 protocol and Gold-independent context allocation helpers."""

from __future__ import annotations

import json
from typing import Any

from paper_research.generation.citation_registry import CitationRegistry
from paper_research.generation.required_claim_output import (
    RequiredClaimInput,
    required_claim_output_token_budget,
)

try:
    from scripts.evidence_qa_dev_lib_v1 import (
        DATA,
        DEV_IDS,
        DOCS,
        canonical_hash,
        overlap,
        read_jsonl,
    )
    from scripts.run_evidence_qa_dev_v1 import load_contexts
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import (  # type: ignore[no-redef]
        DATA,
        DEV_IDS,
        DOCS,
        canonical_hash,
        overlap,
        read_jsonl,
    )
    from run_evidence_qa_dev_v1 import load_contexts  # type: ignore[no-redef]

SOURCE_MANIFEST_HASH = "fcb59b71fc68549479c24f6475f7d18ad9e382aace93e70e93594ee355ffb988"
MANIFEST = DATA / "evidence-qa-dev-v3-manifest.json"
MANIFEST_DOC = DOCS / "evidence-qa-dev-v3-manifest.md"
RUN_ROOT = DATA / "evidence-qa-dev-v3/runs"
FIXTURE_SUMMARY = DATA / "evidence-qa-dev-v3-fixture-summary.json"
READINESS = DATA / "evidence-qa-dev-v3-readiness-v1.json"
READINESS_DOC = DOCS / "evidence-qa-dev-v3-readiness-v1.md"


def build_manifest() -> dict[str, Any]:
    body = {"schema_version": "evidence-qa-dev-v3-manifest-v1", "evaluation_version": "evidence-qa-dev-v3", "manifest_hash": SOURCE_MANIFEST_HASH, "question_ids": DEV_IDS, "question_count": 10, "configuration": {"retrieval": "adjacent_same_page_completion", "prompt": "qa-required-claims-citation-id-v3", "citation_protocol": "citation-id-v2", "required_claim_protocol": "required-claim-slots-v1", "provider": "siliconflow", "model": "Qwen/Qwen3-8B", "temperature": 0, "reranker_enabled": False, "retries": 0, "citation_retries": 0, "billing_mode": "free"}, "selection": {"same_fixed_stage13_dev_questions": True, "questions_reselected": False, "gold_used_for_evidence_selection": False, "oracle_used_for_evidence_selection": False, "human_pilot_used_for_evidence_selection": False}, "engineering_gate_frozen": {"run_directories": "10/10", "schema_success_min": 0.90, "provider_client_completion_min": 0.90, "usage_persisted_before_parse": True, "required_claim_slots_complete": True, "silent_omission_rate": 0, "strict_citation_validation": True}, "quality_candidate_gate_frozen": {"required_claim_coverage_strictly_greater_than": 0.592593, "exact_citation_precision_min": 0.181731, "citation_recall_min": 0.295833, "unsupported_claim_rate_strictly_less_than": 0.8, "refusal_accuracy": 1.0, "unknown_citation_id_rate": 0, "invalid_citation_rate": 0, "silent_omission_rate": 0, "non_regressed_questions_min": 6, "improved_questions_greater_than_regressed": True, "claim_coverage_improved_questions_min": 3, "gain_focus_questions_improved_min": 2, "gain_focus_questions": ["q002", "q007", "q013", "q050"], "no_unsupported_label_gaming": True, "latency_and_tokens_bounded": True, "not_single_question_driven": True}}
    body["protocol_hash"] = canonical_hash(body)
    return body


def write_manifest() -> dict[str, Any]:
    body = build_manifest()
    if MANIFEST.exists() and json.loads(MANIFEST.read_text(encoding="utf-8")) != body:
        raise RuntimeError("frozen Dev v3 manifest changed")
    MANIFEST.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    MANIFEST_DOC.write_text("# Evidence QA Dev v3 Manifest\n\n- Fixed questions: `q001,q002,q004,q005,q007,q008,q013,q015,q019,q050`\n- Source manifest hash: `fcb59b71fc68549479c24f6475f7d18ad9e382aace93e70e93594ee355ffb988`\n- Prompt: `qa-required-claims-citation-id-v3`\n- Required claim protocol: `required-claim-slots-v1`\n- Citation protocol: `citation-id-v2`\n- Reranker/retries/citation retries: disabled/0/0\n- Live execution: **not authorized**\n\nQuality thresholds are frozen before any Dev v3 live result, including required claim coverage strictly greater than `0.592593`.\n", encoding="utf-8")
    return body


def claim_units() -> dict[str, list[dict[str, Any]]]:
    output = {qid: [] for qid in DEV_IDS}
    for row in read_jsonl(DATA / "claim-units-v1.jsonl"):
        if row["question_id"] in output and row.get("expected_answerability"):
            output[row["question_id"]].append(row)
    return output


def build_required_claim_input(question_id: str) -> tuple[dict[str, Any], CitationRegistry, list, dict[str, Any]]:
    contexts, trace = load_contexts(question_id)
    units = claim_units()[question_id]
    allocations: dict[str, list[str]] = {}
    for unit in units:
        ranked = sorted(((overlap(unit["claim_text"], item.evidence), item.chunk_id) for item in contexts), reverse=True)
        allocations[unit["claim_id"]] = [evidence_id for score, evidence_id in ranked[:3] if score > 0]
    registry = CitationRegistry.from_context(contexts, claim_allocations=allocations)
    entries_by_evidence: dict[str, list] = {}
    for entry in registry.entries:
        entries_by_evidence.setdefault(entry.evidence_id, []).append(entry)
    required = []
    for unit in units:
        allowed = [entry.citation_id for entry in registry.entries if unit["claim_id"] in entry.claim_ids]
        summaries = []
        for evidence_id in allocations[unit["claim_id"]]:
            context = next(item for item in contexts if item.chunk_id == evidence_id)
            summaries.append({"evidence_id": evidence_id, "citation_ids": [entry.citation_id for entry in entries_by_evidence[evidence_id] if unit["claim_id"] in entry.claim_ids], "summary": context.evidence[:1000]})
        required.append(RequiredClaimInput(required_claim_id=unit["claim_id"], required_claim_text=unit["claim_text"], evidence_complete=bool(allowed), allowed_citation_ids=allowed, allocated_evidence=summaries, omission_policy="Return answered only with claim-local evidence; otherwise return unsupported with an explicit reason.").model_dump())
    gold = next(row for row in read_jsonl(DATA / "gold-set-v1.jsonl") if row["question_id"] == question_id)
    budget = required_claim_output_token_budget(len(required)).model_dump()
    payload = {"question_id": question_id, "question": gold["question"], "answerability_expectation": gold["answerable"], "required_claims": required, "prompt_version": "qa-required-claims-citation-id-v3", "citation_protocol": "citation-id-v2", "required_claim_protocol": "required-claim-slots-v1", "output_budget": budget, "allocation_policy": "deterministic_lexical_context_only_v1", "gold_evidence_used_for_allocation": False, "oracle_used_for_allocation": False, "human_pilot_used_for_allocation": False}
    return payload, registry, contexts, trace
