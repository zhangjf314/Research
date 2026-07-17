from __future__ import annotations

import importlib.util
from pathlib import Path

from paper_research.generation.bounded_set_search import (
    BOUNDED_SET_SEARCH_VERSION,
    bounded_complementary_set_search,
    enumerate_candidate_sets,
    evaluate_set_sufficiency_v3,
)
from paper_research.generation.citation_selection import CitationCandidate
from paper_research.generation.claim_obligations import build_claim_obligation_set
from paper_research.retrieval.obligation_query_builder_v1 import build_obligation_queries


def cand(cid: str, text: str, *, score: float = 1.0) -> CitationCandidate:
    return CitationCandidate(
        citation_id=cid,
        paper_id="p",
        page=1,
        block_id=cid,
        text=text,
        retrieval_score=score,
    )


def test_exhaustive_combinations_size_1_2_3() -> None:
    candidates = tuple(cand(f"E{i}", f"evidence {i}") for i in range(1, 5))
    combos = enumerate_candidate_sets(candidates)
    assert len([combo for combo in combos if len(combo) == 1]) == 4
    assert len([combo for combo in combos if len(combo) == 2]) == 6
    assert len([combo for combo in combos if len(combo) == 3]) == 4


def test_baseline_plus_one_complement() -> None:
    claim = "The first model uses recurrence while the second uses attention."
    baseline = cand("E1", "The first model uses recurrence.")
    complement = cand("E2", "The second model uses attention in contrast to recurrence.")
    result = bounded_complementary_set_search(claim, (baseline, complement), ("E1",))
    assert "E1" in result.best_ids
    assert "E2" in result.best_ids
    assert result.optimized_matches_exhaustive


def test_baseline_plus_two_complements_and_cap_three() -> None:
    claim = "The model uses Adam and warmup and 8 GPUs."
    candidates = (
        cand("E1", "The model uses Adam optimizer."),
        cand("E2", "The schedule includes warmup steps."),
        cand("E3", "Training uses 8 GPUs."),
        cand("E4", "Unrelated keyword model."),
    )
    result = bounded_complementary_set_search(claim, candidates, ("E1",))
    assert len(result.best_ids) <= 3
    assert "E1" in result.best_ids


def test_set_sufficiency_numeric_endpoints_compose() -> None:
    claim = "Dataset size varies from 22M to 23B tokens."
    obligation_set = build_claim_obligation_set(claim)
    evidence = (
        cand("E1", "Dataset size starts at 22M tokens."),
        cand("E2", "Dataset size reaches 23B tokens."),
    )
    sufficiency = evaluate_set_sufficiency_v3(claim, obligation_set, evidence)
    assert sufficiency.numeric_complete


def test_query_builder_numeric_and_range_queries_are_deterministic() -> None:
    obligation_set = build_claim_obligation_set("Dataset size varies from 22M to 23B tokens.")
    first = build_obligation_queries(obligation_set)
    second = build_obligation_queries(obligation_set)
    assert [query.deterministic_hash for query in first] == [
        query.deterministic_hash for query in second
    ]
    assert any(query.query_type.value == "numeric_exact_query" for query in first)
    assert any(query.query_type.value == "range_endpoint_query" for query in first)


def test_query_builder_has_no_ids_in_query_text() -> None:
    obligation_set = build_claim_obligation_set("The method reports a limitation.")
    queries = build_obligation_queries(obligation_set)
    joined = " ".join(query.query_text.lower() for query in queries)
    assert "question_id" not in joined
    assert "required_claim_id" not in joined


def test_feature_leakage_audit_passes() -> None:
    path = Path("scripts/audit_stage13_26_feature_leakage.py")
    spec = importlib.util.spec_from_file_location("stage13_26_leakage", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    body = module.build()
    assert body["gate"] == "PASSED"


def test_version_constant() -> None:
    assert BOUNDED_SET_SEARCH_VERSION == "bounded-complementary-set-search-v1"
