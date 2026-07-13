import json
import re
from pathlib import Path

import pytest

import scripts.build_retrieval_protocol_v2 as protocol_builder
import scripts.run_retrieval_ablation_v2 as ablation
from paper_research.indexing.embedding import HashEmbeddingProvider

ROOT = Path(__file__).parents[1]
CORPUS = ROOT / "data/evaluation/production-corpus-v1.json"
GOLD_V1 = ROOT / "data/evaluation/gold-set-v1.jsonl"
GOLD_V2 = ROOT / "data/evaluation/retrieval-gold-v2.jsonl"
INDEX_V2 = ROOT / "data/evaluation/retrieval-index-v2.json"


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_production_corpus_excludes_only_two_ocr_fixtures() -> None:
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    included = [paper for paper in corpus["papers"] if paper["included_in_production"]]
    excluded = [paper for paper in corpus["papers"] if not paper["included_in_production"]]
    assert len(included) == 34
    assert len(excluded) == 2
    assert {paper["title"] for paper in excluded} == {
        "mixed-native-scanned",
        "fully-scanned",
    }
    assert all(paper["corpus_role"] == "ocr_fixture" for paper in excluded)
    assert all(paper["exclusion_reason"] for paper in excluded)


def test_protocol_scope_and_filter_invariants() -> None:
    records = jsonl(GOLD_V2)
    assert len(records) == 50
    assert {scope: sum(row["retrieval_scope"] == scope for row in records) for scope in (
        "global",
        "paper",
        "multi_paper",
        "unanswerable",
    )} == {"global": 0, "paper": 46, "multi_paper": 2, "unanswerable": 2}
    for record in records:
        assert record["query_revision_reason"]
        if record["retrieval_scope"] == "global":
            lowered = record["retrieval_query"].lower()
            assert "the target paper" not in lowered
            assert not re.search(r"\b\d{4}\.\d{4,5}\b", record["retrieval_query"])
        elif record["retrieval_scope"] == "paper":
            assert len(record["retrieval_filter"]["paper_ids"]) == 1
        elif record["retrieval_scope"] == "multi_paper":
            assert len(record["retrieval_filter"]["paper_ids"]) >= 2
        else:
            assert not record["gold_paper_ids"]
            assert not record["gold_block_ids"]
            assert not record["gold_pages"]


def test_original_questions_are_not_overwritten() -> None:
    original = {row["question_id"]: row["question"] for row in jsonl(GOLD_V1)}
    revised = {row["question_id"]: row for row in jsonl(GOLD_V2)}
    assert set(original) == set(revised)
    assert all(
        row["original_question"] == original[question_id]
        for question_id, row in revised.items()
    )
    changed = {
        question_id
        for question_id, row in revised.items()
        if row["retrieval_query"] != row["original_question"]
    }
    assert changed == {"q005", "q030"}
    assert revised["q005"]["retrieval_query"] != revised["q030"]["retrieval_query"]
    for question_id in ("q005", "q030"):
        assert revised[question_id]["query_revision_review_status"] == "approved"
        assert revised[question_id]["query_revision_reviewer"] == "zjf"
        assert revised[question_id]["query_revision_reviewed_at"] == "2026-07-13"
        assert revised[question_id]["query_revision_review_notes"]
        assert not revised[question_id]["gold_paper_ids"]
        assert not revised[question_id]["gold_pages"]
        assert not revised[question_id]["gold_block_ids"]


def test_query_review_is_preserved_only_when_identity_is_unchanged() -> None:
    approved = next(row for row in jsonl(GOLD_V2) if row["question_id"] == "q005")
    generated = protocol_builder.retrieval_record(
        next(row for row in jsonl(GOLD_V1) if row["question_id"] == "q005")
    )
    preserved = protocol_builder.preserve_query_review(generated.copy(), approved)
    assert preserved["query_revision_review_status"] == "approved"
    assert preserved["query_revision_reviewer"] == "zjf"
    assert preserved["query_revision_review_notes"] == approved["query_revision_review_notes"]


@pytest.mark.parametrize("field", protocol_builder.REVIEW_IDENTITY_FIELDS)
def test_query_review_is_invalidated_when_identity_changes(field: str) -> None:
    approved = next(row for row in jsonl(GOLD_V2) if row["question_id"] == "q005")
    generated = protocol_builder.retrieval_record(
        next(row for row in jsonl(GOLD_V1) if row["question_id"] == "q005")
    )
    if field == "retrieval_filter":
        generated[field] = {"paper_ids": ["changed"]}
    else:
        generated[field] = f"{generated[field]}-changed"
    invalidated = protocol_builder.preserve_query_review(generated, approved)
    assert invalidated["query_revision_review_status"] == "pending_human_review"
    assert "query_revision_reviewer" not in invalidated
    assert "query_revision_reviewed_at" not in invalidated
    assert "query_revision_review_notes" not in invalidated


def test_q049_q050_have_distinct_intents_and_same_two_paper_filter() -> None:
    records = {row["question_id"]: row for row in jsonl(GOLD_V2)}
    assert records["q049"]["retrieval_query"] != records["q050"]["retrieval_query"]
    assert records["q049"]["retrieval_filter"] == records["q050"]["retrieval_filter"]
    assert len(records["q049"]["gold_paper_ids"]) == 2


def test_evaluation_applies_paper_filter_and_keeps_scopes_separate(monkeypatch) -> None:
    captured = []

    class FakeDense:
        def __init__(self, _provider, _store) -> None:
            pass

        def retrieve(self, _query, *, retrieval_filter, top_k):
            captured.append((retrieval_filter.paper_ids, top_k))
            return []

    monkeypatch.setattr(ablation, "DenseRetriever", FakeDense)
    protocol = [
        {
            "question_id": "paper-test",
            "original_question": "question",
            "retrieval_query": "question",
            "retrieval_scope": "paper",
            "retrieval_filter": {"paper_ids": ["paper-public"]},
            "gold_paper_ids": ["paper-public"],
            "gold_pages": [1],
            "gold_block_ids": ["block-1"],
            "category": "method",
            "difficulty": "easy",
            "review_status": "approved",
            "query_revision_reason": "test",
            "query_revision_version": "retrieval-query-v2",
            "query_revision_author": "test",
            "query_revision_review_status": "not_required_scope_only",
        },
        {
            "question_id": "unanswerable-test",
            "original_question": "question",
            "retrieval_query": "question",
            "retrieval_scope": "unanswerable",
            "retrieval_filter": {"paper_ids": ["paper-public"]},
            "gold_paper_ids": [],
            "gold_pages": [],
            "gold_block_ids": [],
            "category": "unanswerable",
            "difficulty": "easy",
            "review_status": "approved",
            "query_revision_reason": "test",
            "query_revision_version": "retrieval-query-v2",
            "query_revision_author": "test",
            "query_revision_review_status": "pending_human_review",
        },
    ]
    result = ablation.evaluate_variant(
        name="test",
        provider=HashEmbeddingProvider(384),
        retriever_type="dense",
        collection="unused",
        chunks=[],
        protocol=protocol,
        public_to_raw={"paper-public": "paper-raw"},
        raw_to_public={"paper-raw": "paper-public"},
        production_raw_ids=["paper-raw"],
        client=object(),
    )
    assert captured == [(["paper-raw"], ablation.RECALL_K)] * 2
    assert result["metrics"]["paper"]["query_count"] == 1
    assert result["metrics"]["unanswerable"]["query_count"] == 1
    assert result["configuration"]["rerank_enabled"] is False
    assert result["configuration"]["llm_called"] is False


def test_failed_query_is_counted(monkeypatch) -> None:
    class FailingDense:
        def __init__(self, _provider, _store) -> None:
            pass

        def retrieve(self, _query, *, retrieval_filter, top_k):
            raise TimeoutError("simulated")

    monkeypatch.setattr(ablation, "DenseRetriever", FailingDense)
    record = jsonl(GOLD_V2)[0]
    result = ablation.evaluate_variant(
        name="failure-test",
        provider=HashEmbeddingProvider(384),
        retriever_type="dense",
        collection="unused",
        chunks=[],
        protocol=[record],
        public_to_raw={"1706.03762": "paper-raw"},
        raw_to_public={"paper-raw": "1706.03762"},
        production_raw_ids=["paper-raw"],
        client=object(),
    )
    assert result["metrics"]["latency"]["failure_count"] == 1
    assert result["queries"][0]["failure_reason"] == "TimeoutError: simulated"


def test_hash_and_jina_eval_indexes_share_corpus_and_chunks() -> None:
    if not INDEX_V2.exists():
        pytest.skip("evaluation collections have not been materialized yet")
    manifest = json.loads(INDEX_V2.read_text(encoding="utf-8"))
    hash_index = manifest["collections"]["hash"]
    jina_index = manifest["collections"]["jina"]
    assert hash_index["paper_count"] == jina_index["paper_count"] == 34
    assert hash_index["point_count"] == jina_index["point_count"]
    assert hash_index["chunk_signature"] == jina_index["chunk_signature"]
    assert hash_index["dimension"] == 384
    assert jina_index["dimension"] == 1024
