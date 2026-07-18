from __future__ import annotations

import importlib.util
from pathlib import Path

from paper_research.retrieval.local_lexical_index import LexicalDocument, LocalLexicalIndex
from paper_research.retrieval.reciprocal_rank_fusion import FROZEN_RRF_GRID, reciprocal_rank_fusion


def test_bm25_is_deterministic() -> None:
    index = LocalLexicalIndex(
        [
            LexicalDocument("d1", "p1", 1, "b1", "attention enables parallel modeling"),
            LexicalDocument("d2", "p1", 1, "b2", "optimizer warmup schedule"),
        ]
    )
    first = index.bm25("attention parallel", top_k=2)
    second = index.bm25("attention parallel", top_k=2)
    assert first == second
    assert first[0].doc_id == "d1"


def test_exact_numeric_match_and_paper_scope() -> None:
    index = LocalLexicalIndex(
        [
            LexicalDocument("d1", "p1", 1, "b1", "trained on 8 GPUs"),
            LexicalDocument("d2", "p2", 1, "b2", "trained on 8 GPUs"),
        ]
    )
    results = index.exact_numeric("8 GPUs", paper_ids={"p1"})
    assert [result.doc_id for result in results] == ["d1"]


def test_same_page_expansion() -> None:
    index = LocalLexicalIndex(
        [
            LexicalDocument("d1", "p1", 1, "b1", "anchor text"),
            LexicalDocument("d2", "p1", 1, "b2", "neighbor text"),
        ]
    )
    expanded = index.same_page_expand(index.bm25("anchor", top_k=1), top_k=3)
    assert {result.doc_id for result in expanded} == {"d1", "d2"}


def test_rrf_deterministic_and_tiebreak() -> None:
    class R:
        def __init__(self, doc_id: str, rank: int) -> None:
            self.doc_id = doc_id
            self.rank = rank

    fused = reciprocal_rank_fusion({"a": [R("d2", 1)], "b": [R("d1", 1)]}, rrf_k=60)
    assert [result.doc_id for result in fused] == ["d1", "d2"]


def test_frozen_parameter_grid() -> None:
    assert FROZEN_RRF_GRID["rrf_k"] == [20, 40, 60]
    assert FROZEN_RRF_GRID["dense_weight"] == [1.0]


def test_benchmark_builder_outputs_limited_sample_flag() -> None:
    path = Path("scripts/build_retrieval_recall_benchmark_v1.py")
    spec = importlib.util.spec_from_file_location("retrieval_benchmark_builder", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows, summary = module.build_samples()
    assert rows
    assert summary["BENCHMARK_SAMPLE_SIZE_LIMITED"] is True
    assert summary["RETRIEVAL_BENCHMARK_SAMPLE_SIZE_SUFFICIENT"] is False


def test_feature_leakage_audit_passes() -> None:
    path = Path("scripts/audit_retrieval_benchmark_feature_leakage.py")
    spec = importlib.util.spec_from_file_location("retrieval_benchmark_leakage", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    body = module.build()
    assert body["gate"] == "PASSED"


def test_split_builder_has_no_leakage() -> None:
    path = Path("scripts/build_retrieval_recall_benchmark_v1.py")
    spec = importlib.util.spec_from_file_location("retrieval_benchmark_builder_split", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows, _summary = module.build_samples()
    by_relation: dict[str, str] = {}
    for row in rows:
        for relation in row["positive_core_relations"] + row["positive_equivalent_relations"]:
            previous = by_relation.setdefault(relation, row["split"])
            assert previous == row["split"]
