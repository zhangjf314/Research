"""RQ9 §18 tests: provenance, traces, hashing, E0/E1/E2, isolation, pairing."""

from __future__ import annotations

import pytest

from paper_research.evaluation.rag_v2_live import (
    EXPERIMENT_COLLECTION_PREFIX,
    DenseProvenance,
    LiveDenseTrace,
    build_embedding_text,
    collection_isolation_guard,
    dense_relation,
    derive_paper_title,
    embedding_config_hash,
    paired_comparison,
)


def _trace(provenance: DenseProvenance, depth: int = 25) -> LiveDenseTrace:
    return LiveDenseTrace(
        query_id="q",
        claim_id="c",
        query_text="t",
        query_type="method",
        gold_block_ids=["p|1|b"],
        dense_candidates=[
            {
                "block_id": f"p|1|b{i}",
                "chunk_rank": i + 1,
                "block_rank": i + 1,
                "score": 0.5,
                "paper_id": "p",
                "page": 1,
                "section_path": ["Intro"],
            }
            for i in range(depth)
        ],
        bm25_ranking=["x"],
        equal_rrf_ranking=["x"],
        final_context_ranking=["x"],
        provenance=provenance,
    )


def test_proxy_dense_is_not_valid_live_evidence() -> None:
    problems = _trace(DenseProvenance.PROXY_DENSE).validate()
    assert any("PROXY_DENSE" in p for p in problems)


def test_live_trace_requires_minimum_depth() -> None:
    problems = _trace(DenseProvenance.PRODUCTION_JINA_DENSE, depth=19).validate()
    assert any("LIVE_DENSE_DEPTH_INSUFFICIENT" in p for p in problems)
    assert _trace(DenseProvenance.PRODUCTION_JINA_DENSE, depth=20).validate() == []


def test_live_trace_serialization_carries_provenance() -> None:
    record = _trace(DenseProvenance.PRODUCTION_JINA_DENSE).to_record()
    assert record["dense_provenance"] == "PRODUCTION_JINA_DENSE"
    assert record["dense_depth"] == 25
    assert record["schema_version"] == "rag-v2-live-dense-v1"
    assert isinstance(record["dense_candidates"][0]["score"], float)


def test_embedding_config_hash_is_deterministic_and_sensitive() -> None:
    base = dict(
        provider="jina",
        model="m",
        dimensions=1024,
        distance="Cosine",
        query_task="retrieval.query",
        passage_task="retrieval.passage",
        chunk_max_tokens=400,
        chunk_overlap_tokens=60,
        representation="E0",
    )
    assert embedding_config_hash(**base) == embedding_config_hash(**base)
    changed = base | {"representation": "E1"}
    assert embedding_config_hash(**base) != embedding_config_hash(**changed)


def test_embedding_text_builders_and_citation_invariance() -> None:
    chunk_text = "The encoder maps tokens to vectors."
    e0_emb, e0_cite = build_embedding_text("E0", chunk_text)
    assert e0_emb == chunk_text and e0_cite == chunk_text
    e1_emb, e1_cite = build_embedding_text(
        "E1", chunk_text, paper_title="Attention", section_path=["Encoder", "Layers"]
    )
    assert e1_cite == chunk_text  # citation invariance
    assert "Attention" in e1_emb and "Encoder / Layers" in e1_emb
    e2_emb, e2_cite = build_embedding_text(
        "E2", chunk_text, paper_title="Attention", section_path=["Encoder", "Layers", "Sub"]
    )
    assert e2_cite == chunk_text
    assert "Sub" not in e2_emb  # subsection cap at 2 levels
    with pytest.raises(KeyError):
        build_embedding_text("E9", chunk_text)


def test_flat_section_structure_makes_e1_equal_e2() -> None:
    # Documents the observed corpus property: 280/284 chunks have 1 section
    # level, so E1 and E2 embedding texts coincide there.
    chunk_text = "text"
    e1, _ = build_embedding_text("E1", chunk_text, paper_title="T", section_path=["S"])
    e2, _ = build_embedding_text("E2", chunk_text, paper_title="T", section_path=["S"])
    assert e1 == e2


def test_collection_isolation_guard() -> None:
    collection_isolation_guard(f"{EXPERIMENT_COLLECTION_PREFIX}diag_e0_x")
    with pytest.raises(ValueError):
        collection_isolation_guard("papers_jina_eval34_v2__20260713152149")
    with pytest.raises(ValueError):
        collection_isolation_guard("paper_chunks")
    with pytest.raises(ValueError):
        collection_isolation_guard("random_name")


def test_paired_comparison_rules() -> None:
    assert paired_comparison(1, 3) == "dense_wins"
    assert paired_comparison(3, 1) == "bm25_wins"
    assert paired_comparison(2, 2) == "ties"
    assert paired_comparison(0, 4) == "bm25_wins"  # miss penalized
    assert paired_comparison(4, 0) == "dense_wins"


def test_dense_relation_rules() -> None:
    assert dense_relation(1, 3, 2) == "DENSE_HELPED"
    assert dense_relation(5, 2, 4) == "DENSE_HURT"  # fusion worsened vs bm25
    assert dense_relation(4, 4, 4) == "DENSE_NEUTRAL"


def test_derive_paper_title_fallbacks() -> None:
    blocks = [
        {"block_type": "paragraph", "text": "body"},
        {"block_type": "heading", "text": "First Heading"},
    ]
    assert derive_paper_title(blocks) == "First Heading"
    assert derive_paper_title([{"block_type": "title", "text": "T"}]) == "T"
    assert derive_paper_title([{"block_type": "paragraph", "text": "x"}]) == ""
