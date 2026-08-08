from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.finalize_research_agent_stage3d_v1 import stable_hash

ROOT = Path("data/evaluation/research-agent")


def _json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_protocol_amendment_preserves_negative_live_replan_history() -> None:
    amendment = _json("stage3-validation-protocol-amendment-v1.json")

    assert amendment["previous_gate"]["effective_live_replan_observed"] is True
    assert (
        amendment["revised_gate"]["effective_live_replan_observed"]
        == "NOT_REQUIRED_FOR_STAGE3_FREEZE"
    )
    assert amendment["effective_live_replan_observed"] is False
    assert amendment["known_limitation"] == "LIVE_EFFECTIVE_REPLAN_NOT_OBSERVED"
    assert amendment["additional_tasks_after_preregistered_set"] == 0
    assert amendment["behavior_changes_during_stage3c2"] == 0
    assert amendment["stage3c_result"]["effective_replan_observed"] is False
    assert amendment["stage3c2_result"]["effective_live_replan_observed"] is False
    assert amendment["stage3c2_result"]["task1_plan_version_2_is_not_effective_replan"]


def test_stage3_can_freeze_with_known_live_replan_limitation() -> None:
    final = _json("research-agent-stage3-final-v1.json")

    assert final["stage3_status"] == "COMPLETE_WITH_KNOWN_LIMITATION"
    assert final["stage3_complete"] is True
    assert final["stage3_complete_with_known_limitation"] is True
    assert final["stage4_ready"] is True
    assert final["replan_runtime_supported"] is True
    assert final["replan_deterministic_tests_pass"] is True
    assert final["effective_live_replan_observed"] is False
    assert final["effective_live_replan_gate"] == "NOT_REQUIRED_FOR_STAGE3_FREEZE"
    assert final["known_limitation"] == "LIVE_EFFECTIVE_REPLAN_NOT_OBSERVED"
    assert final["new_provider_requests"] == 0
    assert final["new_tokens"] == 0
    assert final["new_cost"] == 0


def test_stage3_agent_behavior_hash_is_stable_and_source_backed() -> None:
    lock = _json("stage3-agent-lock-v1.json")
    behavior = lock["behavior_hash_inputs"]

    assert stable_hash(behavior) == lock["stage3_agent_behavior_hash"]
    for path, recorded_hash in behavior["source_hashes"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == recorded_hash
    assert behavior["runtime_includes_post_stage3c1_fixes"] == [
        "PARTIAL_FAIL_REPLAN_TRANSITION",
        "FINISH_GUARD",
        "VERIFY_TRACE",
        "DECISION_CAUSALITY",
        "REPLAN_DELTA_HASH",
    ]


def test_stage4_workflow_and_comparability_locks_freeze_fairness() -> None:
    workflow = _json("stage4-workflow-control-lock-v1.json")
    comparability = _json("stage4-comparability-lock-v1.json")

    assert workflow["workflow_path"] == "CONTROL_GROUP_WORKFLOW"
    assert workflow["workflow_behavior_changed"] is False
    assert workflow["frozen_for_stage4_control_group"] is True
    assert comparability["same_corpus"] is True
    assert comparability["same_index"] is True
    assert comparability["same_embedding"] is True
    assert comparability["same_retrieval_backend"] is True
    assert comparability["same_reranker_state"] is True
    assert comparability["same_query_rewrite_state"] is True
    assert comparability["same_query_decomposition_state"] is True
    assert comparability["same_model_family"] is True
    assert "replan_task_count" in comparability["replan_metrics_preregistered"]
    assert "post_replan_claim_coverage_delta" in comparability[
        "replan_metrics_preregistered"
    ]


def test_stage4_task_exclusions_cover_stage3c_and_stage3c2_tasks() -> None:
    exclusions = _json("stage4-task-exclusions-v1.json")
    task_ids = {item["task_id"] for item in exclusions["tasks"]}

    assert exclusions["task_count"] == 6
    assert {
        "stage3c-smoke-1-straightforward",
        "stage3c-smoke-2-multi-evidence",
        "stage3c-smoke-3-insufficient-resume",
        "stage3-replan-v1-task-1-dataset-bridge",
        "stage3-replan-v1-task-2-limitation-bridge",
        "stage3-replan-v1-task-3-metric-comparability",
    } <= task_ids
    assert all(item["reason"] == "development_validation_exposure" for item in exclusions["tasks"])
    assert all(item["task_hash"] for item in exclusions["tasks"])


def test_existing_stage3c2_artifact_still_records_no_effective_replan() -> None:
    validation = _json("stage3-live-replan-validation-v1.json")
    task1 = validation["validation_results"][0]

    assert validation["effective_replan_observed"] is False
    assert validation["dynamic_tool_selection_observed"] is True
    assert validation["observation_driven_action_observed"] is True
    assert task1["task_id"] == "stage3-replan-v1-task-1-dataset-bridge"
    assert task1["plan_version"] == 2
    assert task1["effective_replan_observed"] is False
