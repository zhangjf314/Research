from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_stage4c_workflow_agent_evaluation_v1 as stage4c


def test_stage4c_accepts_attempt4_only() -> None:
    _manifest, _protocol, _tasks, _rubrics, results = stage4c.validate_inputs()

    assert results["official_run_id"] == "stage4-official-v1-attempt4"
    assert results["attempt_4"]["status"] == "VALID_COMPLETE"
    assert results["attempt_4"]["stage4c_eligible"] is True
    for attempt in ("attempt_1", "attempt_2", "attempt_3"):
        assert results[attempt].get("stage4c_eligible") is False


def test_blind_package_has_no_direct_system_identity() -> None:
    text = stage4c.BLIND_PACKAGE.read_text(encoding="utf-8")

    assert '"system"' not in text
    assert "-agent" not in text
    assert "-workflow" not in text
    assert "SYSTEM_A" in text
    assert "SYSTEM_B" in text


def test_score_bundle_hash_is_deterministic(tmp_path: Path) -> None:
    one = tmp_path / "a.json"
    two = tmp_path / "b.json"
    one.write_text('{"x": 1}\n', encoding="utf-8")
    two.write_text('{"y": 2}\n', encoding="utf-8")

    first = stage4c.bundle_hash([two, one])
    second = stage4c.bundle_hash([one, two])

    assert first == second


def test_bootstrap_is_paired_and_reproducible() -> None:
    rows = [
        {
            "deltas": {
                metric: float(index)
                for metric in stage4c.PRIMARY_METRICS + ["tokens", "cost", "latency"]
            }
        }
        for index in range(1, 5)
    ]

    first = stage4c.bootstrap(rows, seed=41007, resamples=25)
    second = stage4c.bootstrap(rows, seed=41007, resamples=25)

    assert first == second
    assert first["paired_by_task"] is True


def test_failed_outputs_score_zero_coverage() -> None:
    pair = {
        "task_id": "rt-v1-test",
        "category": "multi_paper_synthesis",
        "difficulty": "easy",
        "rubric": {
            "required_claims": [{"claim_id": "c1"}],
            "required_evidence_sets": [{"evidence_set_id": "e1"}],
        },
    }
    task = {"required_dimensions": [{"dimension_id": "d1"}]}
    output = {
        "status": "FAILED",
        "failure_category": "SYSTEM_PROVIDER_FAILURE",
        "citation_ids_structurally_valid": True,
        "citation_structure_parseable": True,
        "trace_complete": True,
        "accounting_complete": True,
    }

    score = stage4c.score_output(pair, "output_x", output, task)

    assert score["task_success"] == 0.0
    assert score["required_claim_coverage"] == 0.0
    assert score["required_dimension_coverage"] == 0.0
    assert score["evidence_coverage"] == 0.0


def test_budget_comparable_remains_false_after_run_if_artifact_exists() -> None:
    path = stage4c.OUT_FINAL
    if not path.exists():
        pytest.skip("Stage4C final artifact has not been generated yet")

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["efficiency"]["budget_comparable"] is False
    assert payload["cost_separation"]["stage4c_judge_cost"] == 0.0


def test_private_label_map_is_not_public_artifact() -> None:
    tracked_candidates = [
        Path("data/evaluation/research-agent/benchmark/system-label-map.json"),
        Path("data/evaluation/research-agent/benchmark/stage4-system-label-map.json"),
        Path("docs/research-agent/benchmark/system-label-map.md"),
    ]

    assert not any(path.exists() for path in tracked_candidates)
