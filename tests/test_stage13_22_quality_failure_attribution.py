"""Stage 13.22 offline attribution and replay safeguards."""

from __future__ import annotations

import json

from scripts.audit_dev_v3_6_evidence_funnel_v1 import METRICS_JSON, build_rows, summarize
from scripts.audit_evidence_selection_v2_feature_leakage import audit_source
from scripts.replay_dev_v3_6_evidence_selection_v2 import build_replay


def test_dev_v3_6_evidence_funnel_has_all_claims_and_no_unknown_root_causes() -> None:
    rows = build_rows()
    metrics = summarize(rows)

    assert len(rows) == 27
    assert metrics["total_required_claims"] == 27
    assert sum(metrics["primary_root_cause_distribution"].values()) == 27
    assert "UNKNOWN" not in metrics["primary_root_cause_distribution"]
    assert metrics["any_valid_final_recall"] == 0.25925925925925924
    assert metrics["can_selection_only_cross_0296296"] is True
    assert metrics["retrieval_completion_v2_required"] is False


def test_dev_v3_6_selection_v2_replay_is_deterministic_and_fail_closed() -> None:
    first = build_replay()
    second = build_replay()

    assert first["replay_hash"] == second["replay_hash"]
    assert first["selection_version"] == "evidence-selection-v2-candidate"
    assert first["offline_quality_preflight"] == "FAILED"
    assert first["gold_online_dependency"] == 0
    assert first["human_label_online_dependency"] == 0
    assert first["fixed_id_special_cases"] == 0
    assert first["modes"]["selection_v2_only"]["any_valid_recall"] >= 0.296296
    assert (
        first["modes"]["selection_v2_only"]["improvement"]
        < first["modes"]["selection_v2_only"]["regression"]
    )


def test_evidence_selection_v2_feature_leakage_gate_passes() -> None:
    audit = audit_source()

    assert audit["gate"] == "PASSED"
    assert audit["gold_online_leakage"] == 0
    assert audit["human_label_online_leakage"] == 0
    assert audit["fixed_id_special_cases"] == 0


def test_written_funnel_metrics_remain_diagnostic_only() -> None:
    metrics = json.loads(METRICS_JSON.read_text(encoding="utf-8"))

    assert metrics["diagnostic_only"] is True
    assert metrics["gold_used_for_offline_scoring_only"] is True
    assert metrics["total_required_claims"] == 27
