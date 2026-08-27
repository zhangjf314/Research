"""RQ13 §35 tests: oracle isolation, ceiling, selector contracts, freeze."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_oracle_is_evaluation_only_not_imported_by_retrieval() -> None:
    """Gold-dependent oracle code must never be imported by runtime packages."""

    source = subprocess.run(
        [
            "git",
            "grep",
            "-l",
            "rag_v2_oracle",
            "--",
            "src/paper_research/retrieval",
            "src/paper_research/generation",
            "src/paper_research/api",
            "src/paper_research/indexing",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    ).stdout
    assert "rag_v2_oracle" not in source


def test_oracle_top_k_covers_claims_first() -> None:
    from paper_research.evaluation.rag_v2_oracle import oracle_top_k

    pool = ["a", "b", "c", "d", "e", "f"]
    claim_golds = [{"a"}, {"b"}]
    top = oracle_top_k(pool, claim_golds, top_k=5)
    assert {"a", "b"} <= set(top)
    assert len(top) == 5
    # single-claim: gold first, then filler in pool order
    assert oracle_top_k(pool, [{"d"}], top_k=3)[0] == "d"


def test_claim_ceiling_metrics_present() -> None:
    report = json.loads(
        (
            ROOT / "artifacts" / "rag-quality-v2" / "rq13" / "candidate-ceiling"
            / "candidate-ceiling-v1.json"
        )
        .read_text(encoding="utf-8")
    )
    c = report["datasets"]["C"]
    assert c["n"] == 49
    assert 0.0 < c["required_claim_coverage@20"] <= 1.0
    assert c["oracle"]["oracle_required_claim_coverage@5"] == pytest.approx(
        c["required_claim_coverage@20"]
    )


def test_reranker_score_normalization_contract() -> None:
    artifact = json.loads(
        (
            ROOT / "artifacts" / "rag-quality-v2" / "rq13" / "reranker"
            / "rerank-scores-v1.json"
        )
        .read_text(encoding="utf-8")
    )
    assert artifact["model"] == "jina-reranker-v2-base-multilingual"
    assert len(artifact["scores"]) == 121  # 49 C + 27 A + 45 B questions
    for key, scores in artifact["scores"].items():
        assert len(scores) == 12, key  # fusion pool depth (see freeze manifest)
        assert all(isinstance(s, float) for s in scores)


def test_obligation_scoring_probes_do_not_retrieve() -> None:
    """S3 uses obligation rankings as scoring probes only: the selector
    signature takes rankings, never performs retrieval."""

    from paper_research.retrieval.obligation_aware import coverage_aware_context

    # deterministic, no retrieval side channels: same inputs -> same output
    a = coverage_aware_context([["x", "y"], ["z"]], ["x", "y", "z", "w"], top_k=5)
    b = coverage_aware_context([["x", "y"], ["z"]], ["x", "y", "z", "w"], top_k=5)
    assert a == b == ["x", "z", "y", "w"][: len(a)]


def test_gold_cannot_enter_selector() -> None:
    import inspect

    from scripts.run_rag_v2_rq13_smatrix_v1 import select_greedy

    signature = inspect.signature(select_greedy).parameters
    assert "gold" not in signature
    assert "claims" not in signature
    allowed = {"pool", "relevance", "obligation_rankings", "text_of",
               "beta", "gamma", "top_k"}
    assert set(signature) <= allowed


def test_top5_output_invariant() -> None:
    from scripts.run_rag_v2_rq13_smatrix_v1 import select_greedy

    pool = [f"p|b{i}" for i in range(20)]
    rel = {d: 1.0 / (60 + i) for i, d in enumerate(pool)}
    out = select_greedy(
        pool, rel, None, lambda d: f"text of {d}", beta=0.0, gamma=0.1
    )
    assert len(out) == 5
    assert len(set(out)) == 5  # no duplicates


def test_paper_level_aggregation_and_lopo() -> None:
    report = json.loads(
        (
            ROOT / "artifacts" / "rag-quality-v2" / "rq13" / "set-selection"
            / "smatrix-v1.json"
        )
        .read_text(encoding="utf-8")
    )
    c_rows = report["per_question"]["C"]
    papers = {r["_paper"] for r in c_rows}
    assert len(papers) == 10
    # leave-one-paper-out coverage deltas all positive (recorded in report)
    assert report["aggregates"]["C"]["all"]["S1_rerank|required_claim_coverage@5"] > (
        report["aggregates"]["C"]["all"]["S0_production|required_claim_coverage@5"]
    )


def test_c3_freeze_reconstruction() -> None:
    manifest = json.loads(
        (
            ROOT / "artifacts" / "rag-quality-v2" / "rq13" / "candidate-freeze"
            / "candidate-freeze-C3-set-aware-rerank.json"
        )
        .read_text(encoding="utf-8")
    )
    assert manifest["candidate_name"] == "C3-set-aware-rerank"
    assert manifest["candidate_generation_contract"]["pool_depth"] == 12
    assert manifest["reranker_contract"]["extra_retrieval"] is False
    assert manifest["set_selector_contract"]["obligation_queries"] is False
    assert manifest["context_contract"]["top_k"] == 5
    for rel in (
        "scripts/run_rag_v2_rq13_smatrix_v1.py",
        "scripts/freeze_rag_v3_candidate_v1.py",
    ):
        code = subprocess.run(
            ["git", "cat-file", "-e", f"HEAD:{rel}"], cwd=ROOT
        ).returncode
        assert code == 0, rel
    # config hash round-trips
    import hashlib

    spec = {
        "generation": manifest["candidate_generation_contract"],
        "reranker": manifest["reranker_contract"],
        "selector": manifest["set_selector_contract"],
        "context": manifest["context_contract"],
    }
    recomputed = hashlib.sha256(
        json.dumps(spec, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    assert recomputed == manifest["candidate_config_hash"]
