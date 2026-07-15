# ruff: noqa: E501
from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_research.config import Settings
from paper_research.evidence.schema import EvidenceUnit
from paper_research.retrieval.context_completion import complete_with_adjacent_same_page
from scripts.evidence_qa_dev_lib_v1 import DEV_IDS, build_manifest, canonical_hash
from scripts.run_evidence_qa_dev_v1 import safe_preflight

ROOT = Path(__file__).resolve().parents[1]


def test_manifest_is_fixed_stable_and_stratified() -> None:
    first = build_manifest()
    second = build_manifest()
    assert first == second
    assert first["question_count"] == 10
    assert first["question_ids"] == DEV_IDS
    assert first["manifest_hash"] == canonical_hash({k: v for k, v in first.items() if k != "manifest_hash"})
    assert {row["difficulty"] for row in first["questions"]} == {"easy", "medium", "hard"}
    assert any(not row["answerable"] for row in first["questions"])
    assert any(row["multi_paper"] for row in first["questions"])
    assert {"q002", "q007", "q013", "q050"}.issubset(first["question_ids"])


def test_variant_isolation_and_known_selector_block() -> None:
    manifest = build_manifest()
    assert manifest["variants"]["historical_stage11c"]["live_requests"] == 0
    assert manifest["variants"]["retrieval_only"]["prompt_version"] == "qa-production-v1"
    assert manifest["variants"]["evidence_centric"]["status"] == "blocked_by_known_selector_defect"


def test_preflight_fail_closed_for_reranker_and_wrong_variant() -> None:
    base = dict(
        app_profile="production",
        embedding_provider="jina",
        embedding_model="jina-embeddings-v5-text-small",
        embedding_dimensions=1024,
        embedding_api_key="configured",
        llm_provider="siliconflow",
        llm_model="Qwen/Qwen3-8B",
        llm_api_key="configured",
        llm_temperature=0,
        llm_max_retries=0,
        llm_billing_mode="free",
        rerank_enabled=False,
    )
    assert safe_preflight(Settings(**base), "retrieval_only")["api_key_configured"]
    with pytest.raises(RuntimeError, match="known_selector_defect"):
        safe_preflight(Settings(**base), "evidence_centric")
    with pytest.raises(RuntimeError, match="rerank_enabled"):
        safe_preflight(Settings(**{**base, "rerank_enabled": True}), "retrieval_only")


def test_adjacent_completion_is_same_page_and_filters_metadata() -> None:
    rows = [json.loads(line) for line in (ROOT / "data/evaluation/evidence-corpus-v1.jsonl").read_text(encoding="utf-8").splitlines()]
    units = [EvidenceUnit.model_validate(row) for row in rows]
    by_id = {unit.evidence_id: unit for unit in units}
    seed = next(unit for unit in units if unit.next_block_id and unit.block_type == "paragraph")
    # The production helper is generic and neither accepts nor reads question/Gold state.
    from paper_research.retrieval.evidence_retriever import (
        EvidenceCandidate,
        EvidenceScoreComponents,
    )

    candidate = EvidenceCandidate(
        claim_id="test-claim",
        evidence=seed,
        total_score=1,
        score_components=EvidenceScoreComponents(
            query_relevance=1,
            claim_term_coverage=0,
            evidence_role_compatibility=0,
            section_compatibility=0,
            paper_filter_validity=1,
            numeric_fact_compatibility=0,
            comparison_dimension_coverage=0,
            answerability_compatibility=0,
            metadata_penalty=0,
            citation_only_penalty=0,
            duplication_penalty=0,
        ),
    )
    completed = complete_with_adjacent_same_page([candidate], units, seed_limit=1, window=1)
    assert all(item.evidence.paper_id == seed.paper_id for item in completed)
    assert all(item.evidence.page == seed.page for item in completed)
    assert all("metadata" not in item.evidence.evidence_roles for item in completed)
    assert by_id[seed.evidence_id].evidence_id == seed.evidence_id


def test_no_api_key_or_header_in_generated_protocol_sources() -> None:
    for path in [
        ROOT / "scripts/evidence_qa_dev_lib_v1.py",
        ROOT / "scripts/summarize_evidence_qa_dev_v1.py",
        ROOT / "scripts/audit_evidence_qa_dev_v1.py",
    ]:
        text = path.read_text(encoding="utf-8").lower()
        assert "authorization" not in text
        assert "bearer " not in text
