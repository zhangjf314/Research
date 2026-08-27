"""RQ11 §30 tests: sealed-run invariants, one-shot marker, hedge accounting."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = (
    ROOT / "artifacts" / "rag-quality-v2" / "shadow-holdout" / "results"
    / "holdout-eval-v1.json"
)
SEALED = (
    ROOT / "artifacts" / "rag-quality-v2" / "shadow-holdout" / "frozen"
    / "shadow-holdout-sealed-v1.json"
)


def _report() -> dict:
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def test_first_run_marker_and_one_shot() -> None:
    report = _report()
    assert report["first_run"] is True
    assert report["tuning_lock"].startswith("one-shot")


def test_sealed_hash_matches_result_record() -> None:
    sealed = json.loads(SEALED.read_text(encoding="utf-8"))
    report = _report()
    assert report["sealed_dataset_sha256"] == sealed["holdout_dataset_hash"]


def test_arm_identity_and_h1_not_promotable() -> None:
    report = _report()
    arms = report["arms"]
    assert set(arms) == {
        "H0_production_equal_rrf",
        "H1_bm25_only",
        "H2_cstar_e1_a3a",
    }
    assert arms["H1_bm25_only"]["H1_PROMOTABLE"] is False
    assert arms["H2_cstar_e1_a3a"]["representation"] == "E1"
    assert arms["H2_cstar_e1_a3a"]["dense_weight_default"] == 0.25


def test_candidate_config_hash_from_frozen_manifest() -> None:
    manifest = json.loads(
        (
            ROOT / "artifacts" / "rag-quality-v2" / "candidate-freeze"
            / "candidate-freeze-C-star-e1-a3a-lexical-gated-v2.json"
        ).read_text(encoding="utf-8")
    )
    report = _report()
    assert report["candidate_config_hash"] == manifest["candidate_config_hash"]
    assert report["fusion_policy_hash"] == manifest["fusion_policy_hash"]


def test_dense_hedge_accounting_sums_to_answerable() -> None:
    report = _report()
    counts = report["dense_hedge"]["counts"]
    assert (
        counts["DENSE_HEDGE_HELPED"]
        + counts["DENSE_HEDGE_HURT"]
        + counts["DENSE_HEDGE_NEUTRAL"]
        == report["answerable_count"]
    )
    assert len(report["dense_hedge"]["helped_ids"]) == counts["DENSE_HEDGE_HELPED"]
    assert len(report["dense_hedge"]["hurt_ids"]) == counts["DENSE_HEDGE_HURT"]


def test_metric_naming_contract_in_results() -> None:
    from paper_research.evaluation.rag_v2_diagnosis import validate_metric_names

    report = _report()
    for arm, metrics in report["metrics"].items():
        assert validate_metric_names(
            {k: v for k, v in metrics.items() if isinstance(v, float)}
        ) == [], arm
        for k in metrics:
            assert not k.startswith("Recall@"), (arm, k)


def test_bootstrap_config_frozen() -> None:
    report = _report()
    assert report["bootstrap_config"] == {"resamples": 10_000, "seed": 20260822}
