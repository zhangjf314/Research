from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_research.evidence.claims import ClaimUnit, build_claim_units
from paper_research.evidence.schema import (
    EvidenceUnit,
    build_evidence_unit,
    sentence_spans,
    stable_evidence_id,
)
from paper_research.generation.claim_evidence_selector import (
    ClaimEvidenceAllocation,
    ClaimFirstEvidenceSelector,
)
from paper_research.generation.prompts import qa_system_prompt
from paper_research.parsing.types import BoundingBox, PaperBlock
from paper_research.retrieval.evidence_context_builder import EvidenceContextBuilder
from paper_research.retrieval.evidence_retriever import EvidenceCandidate, EvidenceRetriever
from paper_research.retrieval.filters import RetrievalFilter
from paper_research.retrieval.query_router import route_query

ROOT = Path(__file__).resolve().parents[1]


def block(
    block_id: str,
    text: str,
    *,
    paper_page: int = 2,
    block_type: str = "paragraph",
    previous: str | None = None,
    following: str | None = None,
) -> PaperBlock:
    return PaperBlock(
        block_id=block_id,
        block_type=block_type,
        section_path=["Methods"],
        page_start=paper_page,
        page_end=paper_page,
        block_index=int(block_id.removeprefix("b")),
        text=text,
        bbox=BoundingBox(x0=0, y0=0, x1=1, y1=1),
        previous_block_id=previous,
        next_block_id=following,
    )


def evidence(
    paper_id: str,
    source: PaperBlock,
    source_version: str = "test-source-v1",
) -> EvidenceUnit:
    return build_evidence_unit(
        paper_id,
        source,
        source_chunk_id=f"chunk-{source.block_id}",
        source_version=source_version,
    )


def claim(
    claim_id: str = "cl-1",
    *,
    role: str = "explain_method",
    papers: list[str] | None = None,
    answerable: bool = True,
) -> ClaimUnit:
    return ClaimUnit(
        claim_id=claim_id,
        question_id="q-test",
        question_type="method",
        claim_text="The method uses attention training.",
        normalized_claim="the method uses attention training.",
        claim_role=role,
        target_paper_ids=papers or ["p1"],
        target_terms=["method", "attention", "training"],
        expected_answerability=answerable,
        required_evidence_roles=["method"],
    )


def candidate(unit: EvidenceUnit, score: float, claim_id: str = "cl-1") -> EvidenceCandidate:
    decision = route_query(
        "method attention training", [claim(claim_id)], RetrievalFilter(paper_ids=[unit.paper_id])
    )
    return (
        EvidenceRetriever([unit])
        .score_candidates(claim(claim_id, papers=[unit.paper_id]), decision)[0]
        .model_copy(update={"total_score": score})
    )


def test_evidence_id_and_trace_are_stable() -> None:
    source = block("b1", "Our method uses an attention architecture.")
    first = evidence("p1", source)
    second = evidence("p1", source)
    assert first.evidence_id == second.evidence_id
    assert first.evidence_id == stable_evidence_id("p1", 2, "b1", "test-source-v1")
    assert first.citation_triple == ("p1", 2, "b1")


def test_metadata_reference_and_sentence_boundaries() -> None:
    title = evidence("p1", block("b1", "Paper title", block_type="title"))
    reference = evidence("p1", block("b2", "[1] Author. Reference title.", block_type="reference"))
    spans = sentence_spans("First sentence. Second sentence!")
    assert not title.eligible_for_final_context
    assert not reference.eligible_for_final_context
    assert [item.text for item in spans] == ["First sentence.", "Second sentence!"]
    assert spans[0].end <= spans[1].start


def test_previous_next_links_remain_with_source_paper() -> None:
    first = evidence("p1", block("b1", "A method paragraph.", following="b2"))
    second = evidence("p1", block("b2", "A result paragraph.", previous="b1"))
    other = evidence("p2", block("b1", "Another paper method."))
    assert first.next_block_id == second.block_id
    assert second.previous_block_id == first.block_id
    assert first.paper_id != other.paper_id


def test_required_claim_text_is_not_modified_and_unanswerable_is_explicit() -> None:
    gold = [
        {
            "question_id": "q1",
            "question": "How does it work?",
            "category": "method",
            "answerable": True,
            "required_claims": ["Original claim text."],
            "gold_paper_ids": ["p1"],
            "gold_block_ids": ["b1"],
            "gold_pages": [2],
        },
        {
            "question_id": "q2",
            "question": "What absent fact is reported?",
            "category": "unanswerable",
            "answerable": False,
            "required_claims": [],
            "gold_paper_ids": [],
            "gold_block_ids": [],
            "gold_pages": [],
        },
    ]
    retrieval = {
        "q1": {"retrieval_filter": {"paper_ids": ["p1"]}},
        "q2": {"retrieval_filter": {"paper_ids": ["p1"]}},
    }
    units = build_claim_units(gold, retrieval)
    assert units[0].claim_text == "Original claim text."
    assert units[1].claim_role == "verify_absence"
    assert units[1].negative_constraints


def test_multi_paper_and_unknown_routing_are_safe_and_deterministic() -> None:
    multi = claim(papers=["p1", "p2"]).model_copy(update={"question_type": "multi_paper"})
    filter_value = RetrievalFilter(paper_ids=["p1", "p2"])
    first = route_query("compare papers", [multi], filter_value)
    second = route_query("compare papers", [multi], filter_value)
    assert first == second
    assert first.target_paper_ids == ["p1", "p2"]
    assert first.profile.paper_diversity_minimum == 2
    unknown = claim(role="unknown").model_copy(update={"question_type": "unknown"})
    fallback = route_query("topic", [unknown], RetrievalFilter(paper_ids=["p1"]))
    assert fallback.fallback_used
    assert fallback.retrieval_filter.paper_ids == ["p1"]
    assert "metadata" in fallback.profile.exclude_roles


def test_evidence_scoring_is_explainable_and_penalizes_non_evidence() -> None:
    method = evidence("p1", block("b1", "Our method uses attention during training."))
    metadata = evidence("p1", block("b2", "Paper title", block_type="title"))
    decision = route_query(
        "method attention training", [claim()], RetrievalFilter(paper_ids=["p1"])
    )
    rows = EvidenceRetriever([method, metadata]).score_candidates(claim(), decision)
    assert rows[0].evidence.block_id == "b1"
    assert rows[0].score_components.claim_term_coverage > 0
    meta = next(row for row in rows if row.evidence.block_id == "b2")
    assert meta.score_components.metadata_penalty == 1
    assert "default_non_evidence_filter" in meta.rejection_reasons
    assert not hasattr(rows[0].score_components, "gold_score")


def test_duplicate_penalty_and_paper_filter() -> None:
    one = evidence("p1", block("b1", "Our method uses attention during training."))
    duplicate = evidence("p1", block("b2", "Our method uses attention during training."))
    other = evidence("p2", block("b1", "Our method uses attention during training."))
    decision = route_query("method", [claim()], RetrievalFilter(paper_ids=["p1"]))
    rows = EvidenceRetriever([one, duplicate, other]).score_candidates(claim(), decision)
    assert all(row.evidence.paper_id == "p1" for row in rows)
    assert any(row.score_components.duplication_penalty == 1 for row in rows)


def test_claim_first_selector_does_not_force_unsupported_or_unanswerable() -> None:
    unit = evidence("p1", block("b1", "A weak unrelated paragraph."))
    selector = ClaimFirstEvidenceSelector(minimum_score=0.9)
    allocation = selector.select(claim(), [candidate(unit, 0.1)])
    assert allocation.unsupported_before_generation
    assert not allocation.selected_evidence
    refusal = selector.select(claim(answerable=False), [candidate(unit, 1.0)])
    assert not refusal.selected_evidence


def test_selector_supports_explicit_multi_block_minimum() -> None:
    first = candidate(evidence("p1", block("b1", "Our method uses attention.")), 0.8)
    second = candidate(evidence("p1", block("b2", "Training follows two stages.")), 0.7)
    multi = claim().model_copy(update={"multi_block_required": True})
    allocation = ClaimFirstEvidenceSelector(max_per_claim=2).select(multi, [first, second])
    assert allocation.evidence_complete
    assert len(allocation.selected_evidence) == 2


def test_selector_enforces_multi_paper_evidence_quota() -> None:
    first = candidate(
        evidence("p1", block("b1", "Our method uses attention.")), 0.8
    )
    second = candidate(
        evidence("p2", block("b1", "The comparison baseline scores 90 percent.")),
        0.7,
    )
    multi = claim(papers=["p1", "p2"]).model_copy(
        update={"question_type": "multi_paper"}
    )
    allocation = ClaimFirstEvidenceSelector(max_per_claim=2).select(
        multi, [first, second]
    )
    assert allocation.evidence_complete
    assert {item.evidence.paper_id for item in allocation.selected_evidence} == {"p1", "p2"}


def test_context_builder_preserves_allocation_dedup_and_budget_trace() -> None:
    shared = candidate(evidence("p1", block("b1", "Our method uses attention.")), 0.8)
    extra = candidate(evidence("p2", block("b2", "Comparison result is 90 percent.")), 0.7, "cl-2")
    allocations = [
        ClaimEvidenceAllocation(
            claim_id="cl-1", selected_evidence=[shared], evidence_complete=True
        ),
        ClaimEvidenceAllocation(
            claim_id="cl-2", selected_evidence=[shared, extra], evidence_complete=True
        ),
    ]
    result = EvidenceContextBuilder(max_tokens=200).build(allocations)
    assert len({item.evidence_id for item in result.context}) == len(result.context)
    assert {item.paper_id for item in result.context} == {"p1", "p2"}
    shared_item = next(item for item in result.context if item.paper_id == "p1")
    assert shared_item.claim_ids == ["cl-1", "cl-2"]
    assert all(
        item.evidence_id in {row.evidence_id for row in result.context}
        for item in result.trace
        if not item.truncated
    )


def test_context_builder_records_truncation_and_never_adds_unselected() -> None:
    selected = candidate(evidence("p1", block("b1", "word " * 200)), 0.8)
    unselected = candidate(evidence("p1", block("b2", "not selected evidence")), 0.7)
    allocation = ClaimEvidenceAllocation(
        claim_id="cl-1",
        selected_evidence=[selected],
        rejected_evidence=[unselected],
        evidence_complete=True,
    )
    result = EvidenceContextBuilder(max_tokens=5).build([allocation])
    assert not result.context
    assert result.trace[0].truncated
    assert unselected.evidence.evidence_id not in {item.evidence_id for item in result.context}


def test_evidence_prompt_enforces_claim_specific_triples_and_refusal() -> None:
    prompt = qa_system_prompt("qa-evidence-centric-v1")
    assert "allocated to another claim" in prompt
    assert "paper_id, page, block_id" in prompt
    assert "claims=[]" in prompt
    with pytest.raises(ValueError):
        qa_system_prompt("qa-unknown")


def test_offline_artifacts_keep_human_review_pending_and_block_live_qa() -> None:
    retrieval = json.loads(
        (ROOT / "data/evaluation/evidence-retrieval-v1.json").read_text(encoding="utf-8")
    )
    qa = json.loads((ROOT / "data/evaluation/evidence-qa-v1.json").read_text(encoding="utf-8"))
    calibration = [
        json.loads(line)
        for line in (ROOT / "data/evaluation/citation-calibration-v1.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert retrieval["rerank_enabled"] is False
    assert retrieval["llm_called"] is False
    assert retrieval["deep_research_called"] is False
    assert retrieval["oracle_used_for_selection"] is False
    assert retrieval["gold_required_claims_used_for_selection"] is False
    assert retrieval["allow_dev_qa"] is False
    assert qa["llm_called"] is False and qa["full_run"] is False
    assert len(calibration) == 60
    assert all(row["human_review_status"] == "pending" for row in calibration)
    assert all(row["human_label"] is None for row in calibration)


def test_no_api_key_in_stage13_outputs_and_windows_paths_work() -> None:
    paths = [
        ROOT / "data/evaluation/evidence-retrieval-v1.json",
        ROOT / "data/evaluation/evidence-qa-v1.json",
        ROOT / "data/evaluation/evidence-corpus-v1-manifest.json",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert "api_key" not in text.casefold()
    windows = Path(r"D:\Agents\Codex\research\data\evaluation")
    assert windows.parts[-2:] == ("data", "evaluation")


def test_evidence_corpus_manifest_and_links_match_signed_boundary() -> None:
    manifest = json.loads(
        (ROOT / "data/evaluation/evidence-corpus-v1-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    rows = [
        json.loads(line)
        for line in (ROOT / "data/evaluation/evidence-corpus-v1.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert manifest["production_document_count"] == 34
    assert manifest["excluded_ocr_fixture_count"] == 2
    assert manifest["evidence_unit_count"] == len(rows) == 18484
    assert len({row["evidence_id"] for row in rows}) == len(rows)
    ids_by_paper = {}
    for row in rows:
        ids_by_paper.setdefault(row["paper_id"], set()).add(row["block_id"])
    for row in rows:
        for linked in (row["previous_block_id"], row["next_block_id"]):
            if linked:
                assert linked in ids_by_paper[row["paper_id"]]


def test_calibration_strata_and_runtime_defaults_are_frozen() -> None:
    rows = [
        json.loads(line)
        for line in (ROOT / "data/evaluation/citation-calibration-v1.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    strata = {}
    for row in rows:
        strata[row["stratum"]] = strata.get(row["stratum"], 0) + 1
    assert set(strata.values()) == {15}
    assert len({row["sample_id"] for row in rows}) == 60
    config = (ROOT / "src/paper_research/config.py").read_text(encoding="utf-8")
    assert "rerank_enabled: bool = False" in config
    qa = json.loads((ROOT / "data/evaluation/evidence-qa-v1.json").read_text())
    assert qa["status"] == "BLOCKED_BY_OFFLINE_RETRIEVAL_GATE"
    assert qa["metrics"]["tokens"] == 0
    assert qa["metrics"]["monetary_cost_usd"] == "0"
