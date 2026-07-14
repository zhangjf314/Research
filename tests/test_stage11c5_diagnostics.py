# ruff: noqa: E501,E702

from pathlib import Path

import scripts.run_qa_context_diagnostics_v1 as diag
from paper_research.parsing.types import BoundingBox, PaperBlock


def block(identifier: str, page: int, index: int) -> PaperBlock:
    return PaperBlock(
        block_id=identifier, block_type="paragraph", page_start=page, page_end=page,
        block_index=index, text=f"attention evidence {identifier}",
        bbox=BoundingBox(x0=0, y0=0, x1=1, y1=1),
    )


def source() -> dict:
    return {
        "question_id": "q001", "gold": {"answerable": True, "gold_paper_ids": ["p1"], "gold_block_ids": ["b000010"], "gold_pages": [2], "required_claims": ["attention evidence"]},
        "context": [{"chunk_id": "r1", "paper_id": "p1", "block_ids": ["b000020"], "section_path": [], "page_start": 5, "page_end": 5, "evidence": "distractor", "score": 1.0}],
    }


def test_context_modes_keep_oracle_separate_and_deduplicate() -> None:
    blocks = {("p1", "b000010"): block("b000010", 2, 10)}
    gold, audit = diag.build_context(source(), "oracle_gold_only", blocks)
    assert [item.block_ids for item in gold] == [["b000010"]]
    assert audit["oracle"] is True and audit["production_metric"] is False
    mixed, audit = diag.build_context(source(), "oracle_gold_plus_distractors", blocks)
    assert mixed[0].block_ids == ["b000010"] and audit["distractor_count"] == 0
    plus, _ = diag.build_context(source(), "retrieved_plus_missing_gold", blocks)
    assert {item.block_ids[0] for item in plus} == {"b000010", "b000020"}
    repeated = source(); repeated["context"][0]["block_ids"] = ["b000010"]
    plus, _ = diag.build_context(repeated, "retrieved_plus_missing_gold", blocks)
    assert len(plus) == 1


def test_exact_page_adjacent_semantic_and_invalid_classification() -> None:
    context = [diag.block_context("p1", block("b000010", 2, 10)), diag.block_context("p1", block("b000011", 3, 11)), diag.block_context("p1", block("b000020", 4, 20))]
    gold = source()["gold"]
    assert diag.classify_citation({"paper_id": "p1", "page": 2, "block_id": "b000010"}, "x", context, gold)["classification"] == "exact_gold_block"
    assert diag.classify_citation({"paper_id": "p1", "page": 2, "block_id": "b000020"}, "x", [context[2].model_copy(update={"page_start": 2, "page_end": 2})], gold)["classification"] == "same_gold_page"
    assert diag.classify_citation({"paper_id": "p1", "page": 3, "block_id": "b000011"}, "x", context, gold)["classification"] == "adjacent_to_gold_block"
    assert diag.classify_citation({"paper_id": "p1", "page": 4, "block_id": "b000020"}, "attention evidence", context, gold)["classification"] == "semantic_support_non_gold"
    assert diag.classify_citation({"paper_id": "p1", "page": 99, "block_id": "missing"}, "x", context, gold)["classification"] == "invalid"


def test_protocol_guards_and_original_artifact_path() -> None:
    text = Path(diag.__file__).read_text(encoding="utf-8")
    assert "if settings.rerank_enabled" in text
    assert '"deep_research_called": False' in text
    assert diag.PRODUCTION.name == "qa-production-v1.json"
    assert diag.DEFAULT_OUTPUT.name == "qa-context-diagnostics-v1.json"
    assert "llm_model != \"Qwen/Qwen3-8B\"" in text
    assert "prompt_version != \"qa-production-v1\"" in text
    assert "api_key" not in text.lower()
