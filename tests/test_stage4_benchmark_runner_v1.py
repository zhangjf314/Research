from __future__ import annotations

import importlib.util
import json
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


def test_stage4b2_global_cost_cap_is_amended_once() -> None:
    runner = _runner()

    assert runner.GLOBAL_CAPS["max_benchmark_total_cost_usd"] == 4.00


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


def test_workflow_invocation_omits_budget_and_external_search_override() -> None:
    runner = _runner()
    payload = runner.build_request_payload(
        {"system": "workflow", "task_id": "rt-v1-001"},
        {
            "task_id": "rt-v1-001",
            "research_question": "Compare papers.",
            "target_paper_ids": ["p1", "p2"],
        },
    )

    assert "budget" not in payload
    contract = runner.workflow_invocation_contract(payload)
    assert contract["max_external_searches"] == "OMITTED"
    assert contract["budget_field_present"] is False


def test_agent_invocation_keeps_frozen_budget() -> None:
    runner = _runner()
    payload = runner.build_request_payload(
        {"system": "agent", "task_id": "rt-v1-001"},
        {
            "task_id": "rt-v1-001",
            "research_question": "Compare papers.",
            "target_paper_ids": ["p1", "p2"],
        },
    )

    contract = runner.agent_invocation_contract(payload)
    assert contract["max_steps"] == 12
    assert contract["max_tool_calls"] == 16
    assert contract["max_provider_requests"] == 12
    assert contract["external_search_capability"] is False


def test_official_attempt_task_ids_are_namespaced_by_run_id() -> None:
    runner = _runner()
    payload = runner.build_request_payload(
        {"system": "agent", "task_id": "rt-v1-002"},
        {
            "task_id": "rt-v1-002",
            "research_question": "Compare papers.",
            "target_paper_ids": ["p1", "p2"],
        },
        run_id="stage4-official-v1-attempt3",
    )

    assert payload["task_id"] == "stage4-official-v1-attempt3-rt-v1-002-agent"


def test_http_error_body_parsing_keeps_structured_detail() -> None:
    runner = _runner()
    detail = runner.parse_error_body(
        '{"detail":{"code":"SCHEMA_FAILURE","request_id":"req-1"}}',
        "application/json",
    )
    error = runner.BenchmarkHttpError(
        status=503,
        content_type="application/json",
        detail=detail,
    ).to_dict()

    assert error["http_status"] == 503
    assert error["structured_error_code"] == "SCHEMA_FAILURE"
    assert error["request_id"] == "req-1"


def test_attempt2_runtime_paths_are_isolated_from_legacy_attempt1() -> None:
    runner = _runner()

    state_path = str(runner.state_path_for_run("stage4-official-v1-attempt2"))
    normalized = state_path.replace("\\", "/")

    assert normalized.endswith(
        ".runtime/stage4/stage4-official-v1-attempt2/"
        "execution-state/stage4-execution-state-v1.json"
    )


def test_recompute_summary_uses_unit_records_for_failed_and_partial() -> None:
    runner = _runner()
    inputs = runner.BenchmarkInputs(
        manifest={"benchmark_version": "research-benchmark-v1"},
        order={
            "units": [
                {"task_id": "rt-v1-001", "system": "workflow"},
                {"task_id": "rt-v1-001", "system": "agent"},
            ]
        },
        tasks={},
        rubrics={},
    )
    state = {
        "units": {
            "w": {
                "execution_unit_id": "w",
                "task_id": "rt-v1-001",
                "system": "workflow",
                "status": "FAILED",
                "accounting_complete": True,
            },
            "a": {
                "execution_unit_id": "a",
                "task_id": "rt-v1-001",
                "system": "agent",
                "status": "PARTIAL",
                "accounting_complete": True,
            },
        }
    }

    runner.recompute_summary_from_units(inputs, state)

    assert state["terminal_units"] == 2
    assert state["official_workflow_runs"] == 1
    assert state["official_agent_runs"] == 1
    assert state["complete_pairs"] == 1
    assert state["failed_units"] == 1
    assert state["partial_units"] == 1


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


def test_public_results_payload_json_round_trips_with_docker_ps_output() -> None:
    runner = _runner()
    inputs = runner.BenchmarkInputs(
        manifest={"benchmark_version": "research-benchmark-v1", **runner.FROZEN_HASHES},
        order={"units": []},
        tasks={},
        rubrics={},
    )
    state = {
        "official_run_id": "stage4-official-v1-attempt2",
        "benchmark_status": "INVALID",
        "global_caps": runner.GLOBAL_CAPS,
        "global_totals": {
            "provider_requests": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
        },
        "preflight": {
            "docker_compose_ps": {
                "stdout_tail": 'research-api-1 "uvicorn paper_resea…" api\n',
            }
        },
        "units": {},
    }

    payload = runner.public_results_payload(inputs, state)
    round_tripped = json.loads(json.dumps(payload, ensure_ascii=False))

    assert round_tripped["attempt_2"]["status"] == "INVALIDATED_INFRASTRUCTURE"
    assert "uvicorn" in round_tripped["preflight"]["docker_compose_ps"]["stdout_tail"]


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


def test_agent_provider_failure_is_valid_system_failure() -> None:
    runner = _runner()
    response = {
        "status": "FAILED",
        "terminal": True,
        "stop_reason": "PROVIDER_FAILURE",
        "failure_code": "AGENT_DECISION_PROVIDER_ERROR",
        "provider_call_count": 1,
        "token_usage": {
            "input_tokens": 491,
            "output_tokens": 599,
            "total_tokens": 1090,
        },
        "estimated_cost": 0.00023646,
        "checkpoint_id": "stage4-rt-v1-002-agent-0003-PROVIDER_FAILURE",
        "verification_state": None,
    }

    unit = runner.summarize_unit_result(
        unit={
            "system": "agent",
            "task_id": "rt-v1-002",
            "blind_label": "SYSTEM_A",
            "execution_unit_id": "rt-v1-002-agent",
        },
        task={
            "task_id": "rt-v1-002",
            "category": "multi_paper_synthesis",
            "difficulty": "easy",
            "target_paper_ids": ["p1", "p2"],
        },
        response=response,
        http_status=200,
        error=None,
        http_error_detail=None,
        latency_seconds=1.0,
    )

    assert unit["status"] == "FAILED"
    assert unit["stop_reason"] == "PROVIDER_FAILURE"
    assert unit["failure_category"] == "SYSTEM_PROVIDER_FAILURE"
    assert unit["failure_validity"] == "valid_system_failure"
    assert unit["provider_requests"] == 1
    assert unit["total_tokens"] == 1090
    assert unit["trace_complete"] is True
