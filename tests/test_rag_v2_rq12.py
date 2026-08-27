"""RQ12 §30 tests: obligations, merges, coverage packing, claim metrics, labels."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_research.evaluation.rag_v2_claim_metrics import (
    all_claims_covered_at_k,
    evidence_obligation_recall_at_k,
    multi_evidence_complete_rate_at_k,
    required_claim_coverage_at_k,
)
from paper_research.retrieval.obligation_aware import (
    MAX_OBLIGATIONS,
    coverage_aware_context,
    extract_obligations,
    merge_union_rrf,
)

ROOT = Path(__file__).resolve().parents[1]


def test_obligation_extraction_gold_free_contract() -> None:
    # §11: only question text may enter decomposition — gold-shaped inputs
    # must be rejected loudly.
    with pytest.raises(TypeError):
        extract_obligations({"question": "q", "required_claims": ["c"]}, "O1")
    with pytest.raises(TypeError):
        extract_obligations(["gold", "blocks"], "O1")


def test_obligation_extraction_methods_and_bounds() -> None:
    q = "What is the method, and how many parameters does it use, and which datasets are reported?"
    o0 = extract_obligations(q, "O0")
    assert o0 == [q.rstrip("?").strip(" ?.")]
    o1 = extract_obligations(q, "O1")
    assert o1[0].startswith("What is the method")
    assert 1 < len(o1) <= MAX_OBLIGATIONS
    o2 = extract_obligations("The model uses 175B parameters.", "O2")
    assert 1 <= len(o2) <= MAX_OBLIGATIONS
    with pytest.raises(KeyError):
        extract_obligations(q, "O9")


def test_union_rrf_merge_prefers_cross_query_agreement() -> None:
    merged = merge_union_rrf(
        [["a", "x", "y"], ["b", "x", "z"], ["c", "d", "x"]]
    )
    assert merged[0] == "x"  # appears in all three queries
    assert set(merged) == {"a", "b", "c", "d", "x", "y", "z"}


def test_coverage_packing_top5_invariant_and_text_dedup() -> None:
    ranked_lists = [["p|a1", "p|a2"], ["p|b1"], ["p|c1", "p|c2"]]
    global_ranked = ["p|a1", "p|a2", "p|b1", "p|c1", "p|d", "p|e"]
    packed = coverage_aware_context(
        ranked_lists, global_ranked, top_k=5,
        text_of=lambda d: {"p|a1": "alpha beta", "p|a2": "alpha beta",
                           "p|b1": "totally different", "p|c1": "third topic",
                           "p|c2": "third topic extra", "p|d": "fourth",
                           "p|e": "fifth"}[d],
    )
    assert len(packed) <= 5
    # one representative per obligation first
    assert packed[0] == "p|a1"
    assert "p|b1" in packed[:3]
    assert "p|c1" in packed[:4]
    # near-duplicate of a1 (identical text) is suppressed entirely
    assert "p|a2" not in packed


def test_claim_metrics_distinguish_hit_from_coverage() -> None:
    ranked = ["g1", "n1", "n2"]
    claims = [{"g1"}, {"g2"}]
    assert required_claim_coverage_at_k(ranked, claims, 5) == 0.5
    assert all_claims_covered_at_k(ranked, claims, 5) is False
    assert all_claims_covered_at_k(ranked + ["g2"], claims, 5) is True
    assert evidence_obligation_recall_at_k(ranked, claims, 5) == 0.5
    # one gold hit must NOT equal multi-evidence solved
    rows = multi_evidence_complete_rate_at_k([ranked], [claims], 5)
    assert rows == 0.0
    assert multi_evidence_complete_rate_at_k([], [], 5) == "METRIC_NOT_COMPUTABLE"


def test_rq11_relabelled_development_only() -> None:
    diag = json.loads(
        (
            ROOT / "artifacts" / "rag-quality-v2" / "rq12" / "failure-analysis"
            / "rq11-failure-summary-v1.json"
        )
        .read_text(encoding="utf-8")
    )
    assert "DEVELOPMENT_ONLY" in diag["dataset_status"]
    dev = json.loads(
        (
            ROOT / "data" / "evaluation" / "retrieval-multievidence-dev-v1.json"
        )
        .read_text(encoding="utf-8")
    )
    assert dev["dataset_purpose"] == "DEVELOPMENT_ONLY"
    assert dev["dataset_hash"]


def test_c_star_retired_not_patched() -> None:
    retired = json.loads(
        (
            ROOT / "artifacts" / "rag-quality-v2" / "rq12" / "candidate-freeze"
            / "C-STAR-RETIREMENT.json"
        )
        .read_text(encoding="utf-8")
    )
    assert retired["C_STAR_STATUS"] == "FAILED_BLIND_HOLDOUT"
    # v1 freeze manifest untouched
    v1 = json.loads(
        (
            ROOT / "artifacts" / "rag-quality-v2" / "candidate-freeze"
            / "candidate-freeze-C-star-e1-a3a-lexical-gated.json"
        )
        .read_text(encoding="utf-8")
    )
    assert v1["candidate_name"] == "C-star-e1-a3a-lexical-gated"
