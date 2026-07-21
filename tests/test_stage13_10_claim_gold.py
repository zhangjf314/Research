from __future__ import annotations

import copy
import inspect

import pytest

import scripts.build_claim_evidence_gold_dev_v1 as builder
from scripts.build_claim_evidence_gold_dev_v1 import (
    MAX_CANDIDATES,
    build_rows,
    immutable_payload,
)
from scripts.evidence_qa_dev_lib_v1 import canonical_hash
from scripts.review_claim_evidence_gold_dev_v1 import validate


def test_builder_creates_frozen_pending_claim_set() -> None:
    rows, stats = build_rows()
    assert len(rows) == 27
    assert len({row["required_claim_id"] for row in rows}) == 27
    assert all(row["adjudication_status"] == "pending" for row in rows)
    assert all(row["reviewer"] is None for row in rows)
    assert stats["historical_gold_candidates"] == 99
    assert all(
        len(row["candidate_evidence_relations"])
        <= max(MAX_CANDIDATES, len(row["inherited_question_gold_blocks"]))
        for row in rows
    )


def test_candidates_are_unique_and_historical_gold_is_retained() -> None:
    rows, _ = build_rows()
    for row in rows:
        relations = row["candidate_evidence_relations"]
        triples = {
            (relation["paper_id"], relation["page"], relation["block_id"])
            for relation in relations
        }
        assert len(triples) == len(relations)
        assert set(row["inherited_question_gold_blocks"]) <= {
            relation["block_id"] for relation in relations if relation["source_question_gold"]
        }
        assert all(relation["adjudication_label"] is None for relation in relations)


def test_cited_and_human_supported_candidates_are_included() -> None:
    rows, stats = build_rows()
    assert stats["dev_v2_cited_candidates"] > 0
    assert stats["dev_v3_1_cited_candidates"] > 0
    assert stats["human_supported_candidates"] > 0
    assert any(
        relation["cited_in_dev_v2"] or relation["cited_in_dev_v3_1"]
        for row in rows
        for relation in row["candidate_evidence_relations"]
    )


def test_immutable_hash_excludes_only_human_adjudication() -> None:
    rows, _ = build_rows()
    row = rows[0]
    assert canonical_hash(immutable_payload(row)) == row["immutable_record_hash"]
    changed = copy.deepcopy(row)
    changed["review_notes"] = "human note"
    assert canonical_hash(immutable_payload(changed)) == row["immutable_record_hash"]
    changed["required_claim_text"] += " changed"
    assert canonical_hash(immutable_payload(changed)) != row["immutable_record_hash"]


def test_review_validator_rejects_unknown_duplicate_and_conflicting_relations() -> None:
    rows, _ = build_rows()
    row = rows[0]
    relation_id = row["candidate_evidence_relations"][0]["relation_id"]
    approved = copy.deepcopy(rows)
    target = approved[0]
    target.update(
        approved_core_relations=[{
            "core_set_id": "cs-test",
            "required_relations": [relation_id],
            "minimum_complete_set": True,
        }],
        adjudication_status="approved",
        reviewer="human",
        reviewed_at="2026-07-16T00:00:00+00:00",
        review_notes="reviewed",
    )
    validate(approved)
    target["rejected_relations"] = [relation_id]
    with pytest.raises(RuntimeError, match="multiple roles"):
        validate(approved)
    target["rejected_relations"] = []
    target["approved_supporting_relations"] = ["missing"]
    with pytest.raises(RuntimeError, match="unknown adjudicated relation"):
        validate(approved)


def test_no_valid_evidence_is_mutually_exclusive() -> None:
    rows, _ = build_rows()
    target = rows[0]
    relation_id = target["candidate_evidence_relations"][0]["relation_id"]
    target.update(
        approved_core_relations=[{
            "core_set_id": "cs-test",
            "required_relations": [relation_id],
            "minimum_complete_set": True,
        }],
        no_valid_gold_evidence=True,
        adjudication_status="approved",
        reviewer="human",
        reviewed_at="2026-07-16T00:00:00+00:00",
        review_notes="reviewed",
    )
    with pytest.raises(RuntimeError, match="conflicts"):
        validate(rows)


def test_no_online_model_or_reranker_dependencies() -> None:
    source = inspect.getsource(builder)
    assert "httpx" not in source
    assert "Embedding" not in source
    assert "rerank" not in source.lower()
