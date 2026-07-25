from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from paper_research.agents.deep_research_graph import DeepResearchGraph
from paper_research.agents.providers import HybridLocalResearchProvider
from paper_research.agents.research_synthesis_provider import (
    EXPECTED_SECTIONS,
    DeepSeekResearchSynthesisProvider,
    _research_repair_prompt,
    _research_system_prompt,
    _research_user_prompt,
)
from paper_research.agents.state import ResearchBudget, initial_state
from paper_research.config import get_settings

QUERY = "RAG 方法的主要技术路线、实验结果和局限分别是什么？"
CONTRACT_JSON = Path("data/evaluation/research-synthesis-current-contract-v1.json")
CONTRACT_DOC = Path("docs/research-synthesis-current-contract-v1.md")
PREFLIGHT_JSON = Path("data/evaluation/research-section-allowlist-preflight-v1.json")


def main() -> None:
    settings = get_settings()
    contract = _build_contract_snapshot(settings)
    preflight = _build_allowlist_preflight(settings)
    contract["section_allowlist_preflight"] = {
        "path": str(PREFLIGHT_JSON),
        "mapping_invariant_errors": preflight["mapping_invariant_errors"],
        "unassigned_evidence_ids": preflight["unassigned_evidence_ids"],
        "global_evidence_count": preflight["global_evidence_count"],
    }
    CONTRACT_JSON.parent.mkdir(parents=True, exist_ok=True)
    CONTRACT_DOC.parent.mkdir(parents=True, exist_ok=True)
    CONTRACT_JSON.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", "utf-8")
    PREFLIGHT_JSON.write_text(json.dumps(preflight, ensure_ascii=False, indent=2) + "\n", "utf-8")
    CONTRACT_DOC.write_text(_render_contract_doc(contract, preflight), "utf-8")
    print(json.dumps({"contract": contract, "preflight": preflight}, ensure_ascii=False, indent=2))
    if preflight["mapping_invariant_errors"] or preflight["unassigned_evidence_ids"]:
        raise SystemExit(1)


def _build_contract_snapshot(settings: Any) -> dict[str, Any]:
    sample_section_ids = {
        "background": ["E01", "E02"],
        "methods": ["E03"],
        "results": ["E04"],
        "limitations": ["E05"],
    }
    sample_evidence = {
        citation_id: {
            "citation_id": citation_id,
            "evidence_id": citation_id.lower(),
            "paper_id": "redacted-paper",
            "section_path": ["redacted"],
            "page_start": 1,
            "page_end": 1,
            "text": "redacted evidence text for prompt contract hashing only",
            "retrieval_score": 1.0,
            "retrieval_sources": ["contract-fixture"],
            "target_sections": [
                section_id
                for section_id, ids in sample_section_ids.items()
                if citation_id in ids
            ],
        }
        for citation_id in ["E01", "E02", "E03", "E04", "E05"]
    }
    system_prompt = _research_system_prompt()
    user_prompt = _research_user_prompt(
        question=QUERY,
        section_queries={section_id: f"{QUERY} {section_id}" for section_id in EXPECTED_SECTIONS},
        evidence_catalog=sample_evidence,
        section_evidence_ids=sample_section_ids,
        contradictions=[],
    )
    repair_prompt = _research_repair_prompt(
        previous_payload={},
        validation_error=ValueError("contract snapshot validation error"),
        allowed_citation_ids=set(sample_evidence),
        section_evidence_ids=sample_section_ids,
    )
    prompt_checks = _prompt_checks(user_prompt, repair_prompt)
    return {
        "schema_version": "research-synthesis-current-contract-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "prompt_version": DeepSeekResearchSynthesisProvider.schema_name,
        "repair_prompt_version": f"{DeepSeekResearchSynthesisProvider.schema_name}:repair",
        "research_gap_shape": "object",
        "required_section_ids": list(EXPECTED_SECTIONS),
        "citation_key_format": "E[0-9]{2,3}",
        "section_allowlist_enabled": True,
        "global_allowlist_enabled": True,
        "max_attempts": 2,
        "template_fallback": False,
        "provider": getattr(settings, "llm_provider_name", None) or settings.llm_provider,
        "model": settings.llm_model,
        "response_format": settings.llm_response_format,
        "delivered_system_prompt_hash": _sha256(system_prompt),
        "delivered_user_payload_hash": _sha256(user_prompt),
        "repair_prompt_hash": _sha256(repair_prompt),
        "protocol_signature": _sha256(
            "|".join(
                [
                    DeepSeekResearchSynthesisProvider.schema_name,
                    "research_gap_shape=object",
                    "citation_key_format=E[0-9]{2,3}",
                    "section_allowlist=true",
                    "global_allowlist=true",
                ]
            )
        ),
        "prompt_snapshot_tests": prompt_checks,
    }


def _prompt_checks(user_prompt: str, repair_prompt: str) -> dict[str, bool]:
    required_lines = [
        "SECTION: background",
        "SECTION: methods",
        "SECTION: results",
        "SECTION: limitations",
        "ALLOWED_CITATION_IDS:",
        "A claim inside one section may only cite IDs listed for that section.",
        "An ID being present in the global evidence catalog does not make it",
        '"text": "string"',
        '"citation_ids": ["E01"]',
        '"is_inference": false',
    ]
    return {
        "user_prompt_contains_required_section_allowlists": all(
            item in user_prompt for item in required_lines[:7]
        ),
        "repair_prompt_contains_required_section_allowlists": all(
            item in repair_prompt for item in required_lines[:7]
        ),
        "user_prompt_contains_research_gap_object_skeleton": all(
            item in user_prompt for item in required_lines[7:]
        ),
        "repair_prompt_contains_research_gap_object_skeleton": all(
            item in repair_prompt for item in required_lines[7:]
        ),
    }


def _build_allowlist_preflight(settings: Any) -> dict[str, Any]:
    graph = DeepResearchGraph(HybridLocalResearchProvider(settings))
    state = initial_state(
        QUERY,
        ResearchBudget(
            max_iterations=1,
            max_external_searches=0,
            max_papers=1,
            max_evidence_items=40,
            max_estimated_tokens=20_000,
            max_no_new_evidence_rounds=1,
        ),
        None,
    )
    for step in (graph._understand, graph._plan, graph._local_search):  # noqa: SLF001
        state.update(step(state))
    catalog = state["evidence_catalog"]
    section_evidence_ids = state["section_evidence_ids"]
    invariant_errors = _mapping_invariant_errors(catalog, section_evidence_ids)
    assigned = {
        citation_id
        for citation_ids in section_evidence_ids.values()
        for citation_id in citation_ids
    }
    key_errors = [
        citation_id
        for citation_id in catalog
        if not re.fullmatch(r"E\d{2,3}", citation_id)
    ]
    if key_errors:
        invariant_errors.append(f"invalid_citation_key_format:{key_errors}")
    return {
        "schema_version": "research-section-allowlist-preflight-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "query": QUERY,
        "global_evidence_count": len(catalog),
        "background_allowed_ids": section_evidence_ids.get("background", []),
        "methods_allowed_ids": section_evidence_ids.get("methods", []),
        "results_allowed_ids": section_evidence_ids.get("results", []),
        "limitations_allowed_ids": section_evidence_ids.get("limitations", []),
        "multi_section_evidence_ids": [
            citation_id
            for citation_id, item in catalog.items()
            if len(item.get("target_sections", [])) > 1
        ],
        "unassigned_evidence_ids": sorted(set(catalog) - assigned),
        "mapping_invariant_errors": invariant_errors,
        "model_visible_key_format": "E01",
        "full_evidence_text_persisted": False,
        "llm_called": False,
        "reranker_enabled": bool(settings.rerank_enabled),
    }


def _mapping_invariant_errors(
    catalog: dict[str, dict[str, Any]],
    section_evidence_ids: dict[str, list[str]],
) -> list[str]:
    errors: list[str] = []
    for section_id, citation_ids in section_evidence_ids.items():
        for citation_id in citation_ids:
            if citation_id not in catalog:
                errors.append(f"{section_id}:{citation_id}:missing_from_catalog")
                continue
            if section_id not in catalog[citation_id].get("target_sections", []):
                errors.append(f"{section_id}:{citation_id}:missing_target_section")
    for citation_id, item in catalog.items():
        for section_id in item.get("target_sections", []):
            if citation_id not in section_evidence_ids.get(section_id, []):
                errors.append(f"{section_id}:{citation_id}:missing_from_section_allowlist")
    return errors


def _render_contract_doc(contract: dict[str, Any], preflight: dict[str, Any]) -> str:
    return (
        "# Research synthesis current contract v1\n\n"
        "This snapshot contains prompt/schema hashes and citation allowlist metadata only. "
        "It does not persist full paper evidence text.\n\n"
        "```json\n"
        + json.dumps(contract, ensure_ascii=False, indent=2)
        + "\n```\n\n"
        "## Section allowlist preflight\n\n"
        "```json\n"
        + json.dumps(preflight, ensure_ascii=False, indent=2)
        + "\n```\n"
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
