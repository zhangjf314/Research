from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path("scripts/run_stage4_workflow_agent_benchmark_v1.py")


def _runner():
    spec = importlib.util.spec_from_file_location("stage4_runner", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_official_execution_unit_ids_include_benchmark_version() -> None:
    runner = _runner()
    unit = {"task_id": "rt-v1-001", "system": "workflow"}

    assert (
        runner.official_execution_unit_id(unit)
        == "research-benchmark-v1:rt-v1-001:workflow"
    )


def test_order_validation_requires_paired_task_units() -> None:
    runner = _runner()
    valid = [
        {"task_id": "rt-v1-001", "system": "agent"},
        {"task_id": "rt-v1-001", "system": "workflow"},
        {"task_id": "rt-v1-002", "system": "workflow"},
        {"task_id": "rt-v1-002", "system": "agent"},
    ]
    wrong_task_pair = [
        {"task_id": "rt-v1-001", "system": "agent"},
        {"task_id": "rt-v1-002", "system": "workflow"},
    ]
    wrong_system_pair = [
        {"task_id": "rt-v1-001", "system": "agent"},
        {"task_id": "rt-v1-001", "system": "agent"},
    ]

    assert runner.order_violations(valid) == 0
    assert runner.order_violations(wrong_task_pair) == 1
    assert runner.order_violations(wrong_system_pair) == 1


def test_integrity_counts_scope_do_not_mix_workflow_agent_or_pairs() -> None:
    runner = _runner()
    frozen = [
        {"task_id": "rt-v1-001", "system": "agent"},
        {"task_id": "rt-v1-001", "system": "workflow"},
        {"task_id": "rt-v1-002", "system": "workflow"},
        {"task_id": "rt-v1-002", "system": "agent"},
    ]
    states = [
        {
            "execution_unit_id": "research-benchmark-v1:rt-v1-001:agent",
            "task_id": "rt-v1-001",
            "system": "agent",
            "status": "COMPLETED",
            "provider_requests": 1,
        },
        {
            "execution_unit_id": "research-benchmark-v1:rt-v1-001:workflow",
            "task_id": "rt-v1-001",
            "system": "workflow",
            "status": "FAILED",
            "provider_requests": 1,
        },
        {
            "execution_unit_id": "research-benchmark-v1:rt-v1-002:workflow",
            "task_id": "rt-v1-002",
            "system": "workflow",
            "status": "PENDING",
            "provider_requests": 0,
        },
        {
            "execution_unit_id": "research-benchmark-v1:rt-v1-002:agent",
            "task_id": "rt-v1-002",
            "system": "agent",
            "status": "COMPLETED",
            "provider_requests": 1,
        },
    ]

    summary = runner.calculate_integrity(frozen, states)

    assert summary["official_workflow_runs"] == 1
    assert summary["official_agent_runs"] == 2
    assert summary["complete_pairs"] == 1
    assert summary["duplicate_logical_execution_count"] == 0
    assert summary["duplicate_completed_unit_count"] == 0
    assert summary["duplicate_provider_execution_count"] == 0


def test_global_caps_stop_before_next_unit() -> None:
    runner = _runner()
    state = {
        "global_totals": {
            "official_logical_runs": 120,
            "provider_requests": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
        }
    }

    assert runner.cap_exceeded(state) is True


def test_usage_extraction_handles_workflow_and_agent_shapes() -> None:
    runner = _runner()
    workflow = {
        "request_attempt_count": 2,
        "provider_completed_request_count": 1,
        "usage_record_count": 1,
        "active_reserved_tokens": 0,
        "model_usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "monetary_cost_usd": 0.01,
        },
    }
    agent = {
        "provider_call_count": 3,
        "token_usage": {"input_tokens": 7, "output_tokens": 4, "total_tokens": 11},
        "estimated_cost": 0.02,
        "active_reserved_tokens": 0,
    }

    assert runner.extract_usage(workflow)["provider_requests"] == 1
    assert runner.extract_usage(workflow)["provider_failures"] == 1
    assert runner.extract_usage(workflow)["total_tokens"] == 15
    assert runner.extract_usage(agent)["provider_requests"] == 3
    assert runner.extract_usage(agent)["total_tokens"] == 11


def test_sanitizer_redacts_secret_fields() -> None:
    runner = _runner()
    payload = {
        "Authorization": "Bearer secret",
        "nested": {"LLM_API_KEY": "abc123"},
        "normal": "visible",
    }

    sanitized = runner.sanitize_payload(payload)

    assert sanitized["normal"] == "visible"
    assert "secret" not in sanitized["Authorization"]
    assert "abc123" not in sanitized["nested"]["LLM_API_KEY"]


def test_preflight_status_is_case_insensitive_for_provider_script_output() -> None:
    runner = _runner()
    preflight = {
        "provider_health": {
            "payload": {
                "status": "PASSED",
                "minimal_completion_status": "passed",
                "safe_to_start_batch": True,
            }
        }
    }

    assert runner.preflight_passed(preflight) is True
