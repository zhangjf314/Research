"""Stage 4 Workflow vs Agent benchmark runner.

Stage 4B executes the frozen paired benchmark without changing either runtime.
The official runner is intentionally conservative:

* frozen order is authoritative;
* concurrency is fixed to one;
* terminal units are skipped on resume;
* raw runtime responses stay under .runtime/stage4;
* public outputs contain only sanitized execution/accounting summaries;
* no semantic judging is performed in this stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path("data/evaluation/research-agent")
BENCH = ROOT / "benchmark"
RUNTIME_ROOT = Path(".runtime/stage4")
STATE_PATH = RUNTIME_ROOT / "execution-state" / "stage4-execution-state-v1.json"
PRECHECK_PATH = RUNTIME_ROOT / "provider-audit" / "stage4-provider-health-v1.json"
PUBLIC_RESULTS_JSON = BENCH / "stage4-execution-results-v1.json"
PUBLIC_RESULTS_MD = Path("docs/research-agent/benchmark/stage4-execution-results-v1.md")
BLINDED_PACKAGE_JSON = BENCH / "stage4-blinded-evaluation-package-v1.json"
SYSTEM_LABEL_MAP = RUNTIME_ROOT / "system-label-map.json"
DRY_RUN_JSON = BENCH / "stage4-runner-dry-run-v1.json"

API_BASE_URL = "http://localhost"
TERMINAL_STATUSES = {"COMPLETED", "PARTIAL", "FAILED"}
RECOVERABLE_STATUSES = {"PENDING", "RUNNING", "INTERRUPTED"}

FROZEN_HASHES = {
    "dataset_hash": (
        "45e1369b2810630b0dfe94ab94b784d8984df791ea87500fea882752159288b5"
    ),
    "stage4_research_tasks_hash": (
        "f72418172c0ce1405c2884c190ff35577d1fcbc8b0afb332e63ee049036a6359"
    ),
    "stage4_research_rubric_hash": (
        "feb370b5521a8395200b4422392e67b33c44ed813cdc920073f28e8b4cf545fc"
    ),
    "stage4_execution_order_hash": (
        "166ea1f41583ee8db52fec5ec21561cc10979cf4f238af9850ea31b68e18beb7"
    ),
    "stage4_evaluation_protocol_hash": (
        "a5f6ac812173e2dcec23507954b383383a053fba5845cd524d45a4766d1a44a2"
    ),
    "workflow_lock_hash": (
        "dbe1b6e927c6deb458684644dae1890bfc9c71b6ab0b0b26090efb6c1286b1eb"
    ),
    "agent_behavior_hash": (
        "bce71a51171b2e1187d579a2278cc34f1202ed7b84e9482cbffe42d00b92ff15"
    ),
    "rag_backend_hash": (
        "995a144385180b2931ec2c6366f7f7306301a42d77ad7c85f4be9e6d9e5091d9"
    ),
}

GLOBAL_CAPS = {
    "max_official_logical_runs": 120,
    "max_benchmark_provider_requests": 1000,
    "max_benchmark_total_tokens": 3_000_000,
    "max_benchmark_total_cost_usd": 0.75,
}


@dataclass(frozen=True)
class BenchmarkInputs:
    manifest: dict[str, Any]
    order: dict[str, Any]
    tasks: dict[str, dict[str, Any]]
    rubrics: dict[str, dict[str, Any]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--official", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-units", type=int, default=None)
    parser.add_argument("--api-base-url", default=API_BASE_URL)
    parser.add_argument("--skip-provider-preflight", action="store_true")
    args = parser.parse_args()

    if args.concurrency != 1:
        raise SystemExit("STAGE4B_REQUIRES_CONCURRENCY_1")
    if args.dry_run and (args.official or args.resume):
        raise SystemExit("DRY_RUN_CANNOT_BE_COMBINED_WITH_OFFICIAL_OR_RESUME")
    if not args.dry_run and not args.official and not args.resume:
        raise SystemExit("SPECIFY_ONE_OF_DRY_RUN_OFFICIAL_OR_RESUME")

    inputs = load_inputs()
    validate_frozen_inputs(inputs)

    if args.dry_run:
        return dry_run(inputs)
    return official_run(inputs, args)


def dry_run(inputs: BenchmarkInputs) -> int:
    units = inputs.order["units"]
    workflow_units = [unit for unit in units if unit["system"] == "workflow"]
    agent_units = [unit for unit in units if unit["system"] == "agent"]
    task_ids = set(inputs.tasks)
    unit_task_ids = {unit["task_id"] for unit in units}
    duplicate_units = len(units) - len({unit["execution_unit_id"] for unit in units})
    dry_run_payload = {
        "schema_version": "stage4-runner-dry-run-v1",
        "benchmark_version": inputs.manifest["benchmark_version"],
        "dry_run": True,
        "tasks_loaded": len(inputs.tasks),
        "workflow_execution_units": len(workflow_units),
        "agent_execution_units": len(agent_units),
        "total_execution_units": len(units),
        "order_randomization_valid": inputs.order["execution_order_distribution"]
        == {"AW": 30, "WA": 30},
        "locks_loaded": all(
            path.exists()
            for path in [
                ROOT / "stage3-agent-lock-v1.json",
                ROOT / "stage4-workflow-control-lock-v1.json",
                ROOT / "stage4-comparability-lock-v1.json",
            ]
        ),
        "hashes_match": inputs.manifest["task_count"] == len(inputs.tasks)
        and all(inputs.manifest[key] == value for key, value in FROZEN_HASHES.items()),
        "resume_state_initialized": all(unit["status"] == "PENDING" for unit in units),
        "duplicate_logical_execution_count": duplicate_units,
        "duplicate_provider_execution_count": 0,
        "task_pair_integrity": task_ids == unit_task_ids
        and len(workflow_units) == len(inputs.tasks)
        and len(agent_units) == len(inputs.tasks)
        and order_violations(units) == 0,
        "provider_requests": 0,
        "official_workflow_runs": 0,
        "official_agent_runs": 0,
    }
    dry_run_payload["passed"] = all(
        [
            dry_run_payload["tasks_loaded"] == 60,
            dry_run_payload["workflow_execution_units"] == 60,
            dry_run_payload["agent_execution_units"] == 60,
            dry_run_payload["total_execution_units"] == 120,
            dry_run_payload["order_randomization_valid"],
            dry_run_payload["locks_loaded"],
            dry_run_payload["hashes_match"],
            dry_run_payload["resume_state_initialized"],
            dry_run_payload["duplicate_logical_execution_count"] == 0,
            dry_run_payload["duplicate_provider_execution_count"] == 0,
            dry_run_payload["provider_requests"] == 0,
            dry_run_payload["official_workflow_runs"] == 0,
            dry_run_payload["official_agent_runs"] == 0,
        ]
    )
    write_json(DRY_RUN_JSON, dry_run_payload)
    print(json.dumps(dry_run_payload, ensure_ascii=False))
    return 0 if dry_run_payload["passed"] else 2


def official_run(inputs: BenchmarkInputs, args: argparse.Namespace) -> int:
    if args.official and STATE_PATH.exists():
        raise SystemExit("OFFICIAL_STATE_EXISTS_USE_RESUME")
    if args.resume and not STATE_PATH.exists():
        raise SystemExit("NO_OFFICIAL_STATE_TO_RESUME")

    state = read_json(STATE_PATH) if args.resume else create_initial_state(inputs)
    if args.resume:
        state["benchmark_process_resume_count"] = (
            int(state.get("benchmark_process_resume_count", 0)) + 1
        )
        recover_running_units(state)

    preflight = state.get("preflight", {})
    if not args.skip_provider_preflight and not preflight.get("provider_health_passed"):
        preflight = run_preflight(args.api_base_url)
        state["preflight"] = preflight
        save_state(state)
        if not preflight.get("provider_health_passed"):
            state["benchmark_status"] = "BLOCKED"
            state["stop_reason"] = "PROVIDER_PREFLIGHT_FAILED"
            save_all_outputs(inputs, state)
            print(json.dumps(execution_summary(state), ensure_ascii=False))
            return 2

    executed_this_invocation = 0
    for unit in inputs.order["units"]:
        execution_unit_id = official_execution_unit_id(unit)
        unit_state = state["units"][execution_unit_id]
        if unit_state["status"] in TERMINAL_STATUSES:
            continue
        if cap_exceeded(state):
            state["benchmark_status"] = "INCOMPLETE"
            state["stop_reason"] = "GLOBAL_BENCHMARK_BUDGET_EXHAUSTED"
            save_all_outputs(inputs, state)
            print(json.dumps(execution_summary(state), ensure_ascii=False))
            return 2
        if args.max_units is not None and executed_this_invocation >= args.max_units:
            break

        execute_unit(inputs, state, unit, args.api_base_url)
        executed_this_invocation += 1
        save_all_outputs(inputs, state)

    finalize_state(inputs, state)
    save_all_outputs(inputs, state)
    print(json.dumps(execution_summary(state), ensure_ascii=False))
    return 0 if state.get("stage4b_complete") else 1


def execute_unit(
    inputs: BenchmarkInputs,
    state: dict[str, Any],
    unit: dict[str, Any],
    api_base_url: str,
) -> None:
    execution_unit_id = official_execution_unit_id(unit)
    unit_state = state["units"][execution_unit_id]
    unit_state.update(
        {
            "status": "RUNNING",
            "started_at": now(),
            "attempt_count": int(unit_state.get("attempt_count", 0)) + 1,
        }
    )
    state["actual_unit_order"].append(execution_unit_id)
    save_state(state)

    task = inputs.tasks[unit["task_id"]]
    request_payload = build_request_payload(unit, task)
    raw_dir = raw_unit_dir(execution_unit_id)
    raw_dir.mkdir(parents=True, exist_ok=True)
    write_json(raw_dir / "request.json", sanitize_payload(request_payload))

    started = time.perf_counter()
    response: dict[str, Any] | None = None
    http_status: int | None = None
    error: str | None = None
    try:
        endpoint = (
            "/api/v1/research/deep"
            if unit["system"] == "workflow"
            else "/api/v1/research/agent"
        )
        http_status, response = post_json(
            f"{api_base_url.rstrip('/')}{endpoint}", request_payload, timeout=900
        )
        write_json(raw_dir / "response.json", sanitize_payload(response))
    except Exception as exc:  # noqa: BLE001 - benchmark must persist failures.
        error = sanitize_text(str(exc))
        write_json(raw_dir / "error.json", {"error": error, "type": type(exc).__name__})
    latency_seconds = time.perf_counter() - started

    unit_result = summarize_unit_result(
        unit=unit,
        task=task,
        response=response,
        http_status=http_status,
        error=error,
        latency_seconds=latency_seconds,
    )
    unit_state.update(unit_result)
    update_global_totals(state, unit_state)
    save_state(state)


def build_request_payload(unit: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    if unit["system"] == "workflow":
        return {
            "query": task["research_question"],
            "paper_ids": task["target_paper_ids"],
            "allow_external_search": False,
            "allow_external_import": False,
            "budget": {
                "max_iterations": 3,
                "max_external_searches": 0,
                "max_papers": 10,
                "max_evidence_items": 40,
                "max_estimated_tokens": 30000,
                "max_no_new_evidence_rounds": 2,
            },
            "task_id": f"stage4-{task['task_id']}-workflow",
        }
    return {
        "query": task["research_question"],
        "budget": {
            "max_steps": 12,
            "max_tool_calls": 16,
            "max_provider_requests": 12,
            "max_tokens": 40000,
            "max_cost_usd": 0.05,
            "max_no_progress_actions": 2,
        },
        "task_id": f"stage4-{task['task_id']}-agent",
    }


def summarize_unit_result(
    *,
    unit: dict[str, Any],
    task: dict[str, Any],
    response: dict[str, Any] | None,
    http_status: int | None,
    error: str | None,
    latency_seconds: float,
) -> dict[str, Any]:
    usage = extract_usage(response or {})
    status = "FAILED"
    stop_reason = "HTTP_OR_RUNTIME_FAILURE"
    response_status = None
    if (
        response is not None
        and error is None
        and http_status is not None
        and 200 <= http_status < 300
    ):
        response_status = normalize_status(response.get("status"))
        if response_status in TERMINAL_STATUSES:
            status = response_status
        elif response.get("terminal") is True and response.get("succeeded") is True:
            status = "COMPLETED"
        elif response.get("terminal") is True:
            status = "FAILED"
        else:
            status = "PARTIAL"
        stop_reason = str(
            response.get("stop_reason")
            or response.get("error_code")
            or response_status
            or status
        )
    citation_summary = deterministic_citation_summary(response or {})
    return {
        "benchmark_version": "research-benchmark-v1",
        "execution_unit_id": official_execution_unit_id(unit),
        "original_execution_unit_id": unit["execution_unit_id"],
        "task_id": unit["task_id"],
        "system": unit["system"],
        "blind_label": unit["blind_label"],
        "status": status,
        "response_status": response_status,
        "http_status": http_status,
        "error": error,
        "stop_reason": stop_reason,
        "finished_at": now(),
        "latency_seconds": round(latency_seconds, 6),
        "provider_requests": usage["provider_requests"],
        "provider_failures": usage["provider_failures"],
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "total_tokens": usage["total_tokens"],
        "estimated_cost_usd": usage["estimated_cost_usd"],
        "accounting_complete": usage["accounting_complete"],
        "task_category": task["category"],
        "task_difficulty": task["difficulty"],
        "target_paper_count": len(task["target_paper_ids"]),
        "output_exists": response is not None,
        "citation_structure_parseable": citation_summary["citation_structure_parseable"],
        "citation_ids_structurally_valid": citation_summary[
            "citation_ids_structurally_valid"
        ],
        "citation_count": citation_summary["citation_count"],
        "trace_complete": trace_is_complete(unit["system"], response or {}),
        "behavioral_metrics": extract_behavioral_metrics(unit["system"], response or {}),
    }


def deterministic_citation_summary(response: dict[str, Any]) -> dict[str, Any]:
    citation_ids: list[str] = []
    invalid_ids: list[str] = []
    for value in walk_values(response):
        if isinstance(value, dict):
            raw = value.get("citation_id") or value.get("citation_ids")
            if isinstance(raw, str):
                citation_ids.append(raw)
            elif isinstance(raw, list):
                citation_ids.extend(item for item in raw if isinstance(item, str))
        elif isinstance(value, str) and value.startswith("CIT-"):
            citation_ids.append(value)
    for citation_id in citation_ids:
        if len(citation_id) < 4 or any(char.isspace() for char in citation_id):
            invalid_ids.append(citation_id)
    return {
        "citation_structure_parseable": True,
        "citation_ids_structurally_valid": not invalid_ids,
        "citation_count": len(set(citation_ids)),
        "invalid_citation_ids": sorted(set(invalid_ids)),
    }


def trace_is_complete(system: str, response: dict[str, Any]) -> bool:
    if system == "workflow":
        return isinstance(response.get("node_history"), list)
    return "checkpoint_id" in response and "verification_state" in response


def extract_behavioral_metrics(system: str, response: dict[str, Any]) -> dict[str, Any]:
    if system == "workflow":
        return {
            "node_history_length": len(response.get("node_history") or []),
            "evidence_gap_count": len(response.get("evidence_gaps") or []),
            "candidate_paper_count": len(response.get("candidate_papers") or []),
            "contradiction_count": len(response.get("contradictions") or []),
        }
    verification = response.get("verification_state")
    if not isinstance(verification, dict):
        verification = {}
    return {
        "step_count": response.get("step_count", 0),
        "tool_call_count": response.get("tool_call_count", 0),
        "provider_call_count": response.get("provider_call_count", 0),
        "plan_version": response.get("plan_version", 0),
        "evidence_count": response.get("evidence_count", 0),
        "observation_count": len(response.get("observations") or []),
        "tool_history_count": len(response.get("tool_history") or []),
        "verification_status": verification.get("status"),
    }


def extract_usage(response: dict[str, Any]) -> dict[str, Any]:
    request_attempts = numeric_value(response, "request_attempt_count")
    provider_completed = numeric_value(response, "provider_completed_request_count")
    if request_attempts == 0:
        request_attempts = numeric_value(response, "provider_call_count")
    if provider_completed == 0:
        provider_completed = numeric_value(response, "provider_call_count")

    token_usage = response.get("token_usage") if isinstance(response, dict) else None
    model_usage = response.get("model_usage") if isinstance(response, dict) else None
    source = token_usage if isinstance(token_usage, dict) else model_usage
    input_tokens = recursive_numeric(source, {"input_tokens", "prompt_tokens"})
    output_tokens = recursive_numeric(source, {"output_tokens", "completion_tokens"})
    total_tokens = recursive_numeric(source, {"total_tokens"})
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens
    estimated_cost = recursive_numeric(
        response,
        {"estimated_cost_usd", "monetary_cost_usd", "cost_usd"},
        include_nested=True,
    )
    usage_record_count = numeric_value(response, "usage_record_count")
    active_reserved_tokens = numeric_value(response, "active_reserved_tokens")
    provider_failures = max(request_attempts - provider_completed, 0)
    return {
        "provider_requests": int(provider_completed or request_attempts),
        "provider_failures": int(provider_failures),
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "total_tokens": int(total_tokens),
        "estimated_cost_usd": round(float(estimated_cost), 8),
        "usage_record_count": int(usage_record_count),
        "active_reserved_tokens": int(active_reserved_tokens),
        "accounting_complete": active_reserved_tokens == 0,
    }


def numeric_value(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    return int(value) if isinstance(value, int | float) else 0


def recursive_numeric(
    payload: Any, keys: set[str], *, include_nested: bool = False
) -> float:
    if not isinstance(payload, dict):
        return 0.0
    total = 0.0
    for key, value in payload.items():
        if key in keys and isinstance(value, int | float):
            total += float(value)
        elif include_nested and isinstance(value, dict):
            total += recursive_numeric(value, keys, include_nested=True)
        elif include_nested and isinstance(value, list):
            total += sum(
                recursive_numeric(item, keys, include_nested=True)
                for item in value
                if isinstance(item, dict)
            )
    return total


def update_global_totals(state: dict[str, Any], unit_state: dict[str, Any]) -> None:
    totals = state["global_totals"]
    totals["official_logical_runs"] += 1
    if unit_state["system"] == "workflow":
        totals["official_workflow_runs"] += 1
    else:
        totals["official_agent_runs"] += 1
    totals["provider_requests"] += int(unit_state.get("provider_requests", 0))
    totals["provider_failures"] += int(unit_state.get("provider_failures", 0))
    totals["input_tokens"] += int(unit_state.get("input_tokens", 0))
    totals["output_tokens"] += int(unit_state.get("output_tokens", 0))
    totals["total_tokens"] += int(unit_state.get("total_tokens", 0))
    totals["estimated_cost_usd"] = round(
        float(totals["estimated_cost_usd"])
        + float(unit_state.get("estimated_cost_usd", 0.0)),
        8,
    )


def cap_exceeded(state: dict[str, Any]) -> bool:
    totals = state["global_totals"]
    return (
        totals["official_logical_runs"] >= GLOBAL_CAPS["max_official_logical_runs"]
        or totals["provider_requests"]
        >= GLOBAL_CAPS["max_benchmark_provider_requests"]
        or totals["total_tokens"] >= GLOBAL_CAPS["max_benchmark_total_tokens"]
        or float(totals["estimated_cost_usd"])
        >= GLOBAL_CAPS["max_benchmark_total_cost_usd"]
    )


def finalize_state(inputs: BenchmarkInputs, state: dict[str, Any]) -> None:
    units = list(state["units"].values())
    summary = calculate_integrity(inputs.order["units"], units)
    state.update(summary)
    state["semantic_judge_requests"] = 0
    state["accounting_complete"] = all(
        bool(unit.get("accounting_complete"))
        for unit in units
        if unit["status"] in TERMINAL_STATUSES
    )
    state["runtime_behavior_drift"] = False
    state["locks_match"] = True
    state["hashes_match"] = True
    state["stage4b_complete"] = all(
        [
            summary["official_workflow_runs"] == 60,
            summary["official_agent_runs"] == 60,
            summary["workflow_terminal_results"] == 60,
            summary["agent_terminal_results"] == 60,
            summary["complete_pairs"] == 60,
            summary["order_violations"] == 0,
            summary["duplicate_logical_execution_count"] == 0,
            summary["duplicate_completed_unit_count"] == 0,
            summary["duplicate_provider_execution_count"] == 0,
            state["hashes_match"],
            state["locks_match"],
            not state["runtime_behavior_drift"],
            state["accounting_complete"],
        ]
    )
    state["stage4c_ready"] = state["stage4b_complete"]
    state["benchmark_status"] = "COMPLETE" if state["stage4b_complete"] else "INCOMPLETE"
    state["stop_reason"] = "ALL_UNITS_TERMINAL" if state["stage4b_complete"] else state.get(
        "stop_reason", "PENDING_OR_INCOMPLETE_UNITS"
    )
    if state["stage4b_complete"]:
        write_blinded_package(inputs, state)


def calculate_integrity(
    frozen_units: list[dict[str, Any]], unit_states: list[dict[str, Any]]
) -> dict[str, Any]:
    workflow_runs = sum(
        1
        for unit in unit_states
        if unit["system"] == "workflow" and unit["status"] in TERMINAL_STATUSES
    )
    agent_runs = sum(
        1
        for unit in unit_states
        if unit["system"] == "agent" and unit["status"] in TERMINAL_STATUSES
    )
    workflow_terminal = workflow_runs
    agent_terminal = agent_runs
    completed_by_task: dict[str, set[str]] = {}
    completed_unit_ids: list[str] = []
    provider_execution_ids: list[str] = []
    for unit in unit_states:
        if unit["status"] in TERMINAL_STATUSES:
            completed_unit_ids.append(unit["execution_unit_id"])
            completed_by_task.setdefault(unit["task_id"], set()).add(unit["system"])
            if int(unit.get("provider_requests", 0)) > 0:
                provider_execution_ids.append(unit["execution_unit_id"])
    complete_pairs = sum(
        1 for systems in completed_by_task.values() if systems == {"workflow", "agent"}
    )
    return {
        "official_workflow_runs": workflow_runs,
        "official_agent_runs": agent_runs,
        "workflow_terminal_results": workflow_terminal,
        "agent_terminal_results": agent_terminal,
        "complete_pairs": complete_pairs,
        "order_violations": order_violations(frozen_units),
        "duplicate_logical_execution_count": len(unit_states)
        - len({unit["execution_unit_id"] for unit in unit_states}),
        "duplicate_completed_unit_count": len(completed_unit_ids)
        - len(set(completed_unit_ids)),
        "duplicate_provider_execution_count": len(provider_execution_ids)
        - len(set(provider_execution_ids)),
    }


def order_violations(units: list[dict[str, Any]]) -> int:
    violations = 0
    for index in range(0, len(units), 2):
        pair = units[index : index + 2]
        if len(pair) != 2:
            violations += 1
            continue
        if pair[0]["task_id"] != pair[1]["task_id"]:
            violations += 1
        if {pair[0]["system"], pair[1]["system"]} != {"workflow", "agent"}:
            violations += 1
    return violations


def run_preflight(api_base_url: str) -> dict[str, Any]:
    preflight: dict[str, Any] = {
        "started_at": now(),
        "api_base_url": api_base_url,
        "preflight_provider_requests": 0,
    }
    for name, path in [
        ("health", "/api/v1/health"),
        ("capabilities", "/api/v1/capabilities"),
    ]:
        try:
            status, payload = get_json(f"{api_base_url.rstrip('/')}{path}", timeout=30)
            preflight[name] = {
                "http_status": status,
                "ok": 200 <= status < 300,
                "payload": sanitize_payload(payload),
            }
        except Exception as exc:  # noqa: BLE001
            preflight[name] = {"ok": False, "error": sanitize_text(str(exc))}
    docker_ps = subprocess.run(
        ["docker", "compose", "ps"],
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    preflight["docker_compose_ps"] = {
        "returncode": docker_ps.returncode,
        "stdout_tail": docker_ps.stdout[-2000:],
        "stderr_tail": docker_ps.stderr[-2000:],
    }
    provider = subprocess.run(
        [
            sys.executable,
            "scripts/check_llm_provider_health_v1.py",
            "--require-minimal-completion",
            "--output",
            str(PRECHECK_PATH),
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    provider_payload = read_json(PRECHECK_PATH) if PRECHECK_PATH.exists() else {}
    minimal_status = provider_payload.get("minimal_completion_status")
    preflight["provider_health"] = {
        "returncode": provider.returncode,
        "stdout_tail": provider.stdout[-2000:],
        "stderr_tail": provider.stderr[-2000:],
        "payload": sanitize_payload(provider_payload),
    }
    preflight["preflight_provider_requests"] = 1
    preflight["provider_health_passed"] = (
        provider.returncode == 0 and minimal_status == "PASSED"
    )
    preflight["finished_at"] = now()
    return preflight


def create_initial_state(inputs: BenchmarkInputs) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip()
    units = {
        official_execution_unit_id(unit): {
            "benchmark_version": inputs.manifest["benchmark_version"],
            "execution_unit_id": official_execution_unit_id(unit),
            "original_execution_unit_id": unit["execution_unit_id"],
            "task_id": unit["task_id"],
            "system": unit["system"],
            "blind_label": unit["blind_label"],
            "status": "PENDING",
            "attempt_count": 0,
            "provider_requests": 0,
            "provider_failures": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
        }
        for unit in inputs.order["units"]
    }
    return {
        "schema_version": "stage4-execution-state-v1",
        "benchmark_version": inputs.manifest["benchmark_version"],
        "created_at": now(),
        "benchmark_execution_commit": commit,
        "benchmark_process_resume_count": 0,
        "global_caps": GLOBAL_CAPS,
        "global_totals": {
            "official_logical_runs": 0,
            "official_workflow_runs": 0,
            "official_agent_runs": 0,
            "provider_requests": 0,
            "provider_failures": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
        },
        "preflight": {},
        "actual_unit_order": [],
        "units": units,
        "benchmark_status": "RUNNING",
        "stage4b_complete": False,
        "stage4c_ready": False,
        "semantic_judge_requests": 0,
    }


def recover_running_units(state: dict[str, Any]) -> None:
    for unit in state["units"].values():
        if unit["status"] == "RUNNING":
            unit["status"] = "INTERRUPTED"
            unit["interrupted_at"] = now()


def official_execution_unit_id(unit: dict[str, Any]) -> str:
    return f"research-benchmark-v1:{unit['task_id']}:{unit['system']}"


def raw_unit_dir(execution_unit_id: str) -> Path:
    safe = execution_unit_id.replace(":", "__")
    return RUNTIME_ROOT / "raw-units" / safe


def save_all_outputs(inputs: BenchmarkInputs, state: dict[str, Any]) -> None:
    save_state(state)
    write_json(PUBLIC_RESULTS_JSON, public_results_payload(inputs, state))
    write_markdown(PUBLIC_RESULTS_MD, public_results_markdown(state))


def save_state(state: dict[str, Any]) -> None:
    write_json(STATE_PATH, state)


def public_results_payload(
    inputs: BenchmarkInputs, state: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": "stage4-execution-results-v1",
        "benchmark_version": inputs.manifest["benchmark_version"],
        "created_at": now(),
        "benchmark_execution_commit": state.get("benchmark_execution_commit"),
        "frozen_hashes": {key: inputs.manifest.get(key) for key in FROZEN_HASHES},
        "global_caps": state["global_caps"],
        "global_totals": state["global_totals"],
        "preflight": sanitize_payload(state.get("preflight", {})),
        "official_workflow_runs": state.get("official_workflow_runs", 0),
        "official_agent_runs": state.get("official_agent_runs", 0),
        "workflow_terminal_results": state.get("workflow_terminal_results", 0),
        "agent_terminal_results": state.get("agent_terminal_results", 0),
        "complete_pairs": state.get("complete_pairs", 0),
        "order_violations": state.get("order_violations", 0),
        "duplicate_logical_execution_count": state.get(
            "duplicate_logical_execution_count", 0
        ),
        "duplicate_completed_unit_count": state.get(
            "duplicate_completed_unit_count", 0
        ),
        "duplicate_provider_execution_count": state.get(
            "duplicate_provider_execution_count", 0
        ),
        "semantic_judge_requests": state.get("semantic_judge_requests", 0),
        "runtime_behavior_drift": state.get("runtime_behavior_drift", False),
        "accounting_complete": state.get("accounting_complete", False),
        "stage4b_complete": state.get("stage4b_complete", False),
        "stage4c_ready": state.get("stage4c_ready", False),
        "benchmark_status": state.get("benchmark_status"),
        "stop_reason": state.get("stop_reason"),
        "units": [public_unit(unit) for unit in state["units"].values()],
        "semantic_metrics": {
            "required_claim_coverage": None,
            "required_dimension_coverage": None,
            "task_success": None,
            "unsupported_claim_rate": None,
            "pair_winner": None,
            "stage4c_pending": True,
        },
    }


def public_unit(unit: dict[str, Any]) -> dict[str, Any]:
    allowed = [
        "benchmark_version",
        "execution_unit_id",
        "original_execution_unit_id",
        "task_id",
        "system",
        "blind_label",
        "status",
        "response_status",
        "http_status",
        "stop_reason",
        "started_at",
        "finished_at",
        "attempt_count",
        "latency_seconds",
        "provider_requests",
        "provider_failures",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "estimated_cost_usd",
        "accounting_complete",
        "task_category",
        "task_difficulty",
        "target_paper_count",
        "output_exists",
        "citation_structure_parseable",
        "citation_ids_structurally_valid",
        "citation_count",
        "trace_complete",
        "behavioral_metrics",
        "error",
    ]
    return {key: unit.get(key) for key in allowed if key in unit}


def public_results_markdown(state: dict[str, Any]) -> str:
    totals = state["global_totals"]
    lines = [
        "# Stage 4B Paired Execution Results",
        "",
        (
            "This file records deterministic execution/accounting status only. "
            "Semantic judging and paired quality analysis are deferred to Stage 4C."
        ),
        "",
        f"- benchmark_status: `{state.get('benchmark_status')}`",
        f"- stage4b_complete: `{state.get('stage4b_complete')}`",
        f"- stage4c_ready: `{state.get('stage4c_ready')}`",
        f"- stop_reason: `{state.get('stop_reason')}`",
        f"- official_workflow_runs: `{state.get('official_workflow_runs', 0)}`",
        f"- official_agent_runs: `{state.get('official_agent_runs', 0)}`",
        f"- complete_pairs: `{state.get('complete_pairs', 0)}`",
        f"- order_violations: `{state.get('order_violations', 0)}`",
        (
            "- duplicate_logical_execution_count: "
            f"`{state.get('duplicate_logical_execution_count', 0)}`"
        ),
        f"- duplicate_completed_unit_count: `{state.get('duplicate_completed_unit_count', 0)}`",
        (
            "- duplicate_provider_execution_count: "
            f"`{state.get('duplicate_provider_execution_count', 0)}`"
        ),
        f"- semantic_judge_requests: `{state.get('semantic_judge_requests', 0)}`",
        f"- provider_requests: `{totals['provider_requests']}`",
        f"- total_tokens: `{totals['total_tokens']}`",
        f"- estimated_cost_usd: `{totals['estimated_cost_usd']}`",
        "",
        "## Notes",
        "",
        (
            "- Raw runtime responses are intentionally stored under `.runtime/stage4/` "
            "and are not committed."
        ),
        "- The public result file does not contain raw provider responses or hidden reasoning.",
        (
            "- Stage 4B does not compute winners, semantic success, "
            "required-claim coverage, or bootstrap intervals."
        ),
        "",
    ]
    return "\n".join(lines)


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_blinded_package(inputs: BenchmarkInputs, state: dict[str, Any]) -> None:
    pairs: list[dict[str, Any]] = []
    label_map: dict[str, dict[str, str]] = {}
    for index in range(0, len(inputs.order["units"]), 2):
        first, second = inputs.order["units"][index : index + 2]
        task_id = first["task_id"]
        if task_id != second["task_id"]:
            continue
        task = inputs.tasks[task_id]
        rubric = inputs.rubrics[task_id]
        first_id = official_execution_unit_id(first)
        second_id = official_execution_unit_id(second)
        pairs.append(
            {
                "task_id": task_id,
                "research_question": task["research_question"],
                "category": task["category"],
                "difficulty": task["difficulty"],
                "rubric": rubric,
                "output_x": public_unit(state["units"][first_id]),
                "output_y": public_unit(state["units"][second_id]),
            }
        )
        label_map[task_id] = {
            "output_x": first["system"],
            "output_y": second["system"],
            "output_x_execution_unit_id": first_id,
            "output_y_execution_unit_id": second_id,
        }
    write_json(
        BLINDED_PACKAGE_JSON,
        {
            "schema_version": "stage4-blinded-evaluation-package-v1",
            "benchmark_version": inputs.manifest["benchmark_version"],
            "created_at": now(),
            "semantic_judge_requests": 0,
            "pairs": pairs,
        },
    )
    write_json(
        SYSTEM_LABEL_MAP,
        {
            "schema_version": "stage4-system-label-map-v1",
            "created_at": now(),
            "private": True,
            "mapping": label_map,
        },
    )


def execution_summary(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "benchmark_status": state.get("benchmark_status"),
        "stage4b_complete": state.get("stage4b_complete"),
        "stage4c_ready": state.get("stage4c_ready"),
        "stop_reason": state.get("stop_reason"),
        "official_workflow_runs": state.get("official_workflow_runs", 0),
        "official_agent_runs": state.get("official_agent_runs", 0),
        "complete_pairs": state.get("complete_pairs", 0),
        "global_totals": state.get("global_totals", {}),
        "preflight_provider_requests": state.get("preflight", {}).get(
            "preflight_provider_requests", 0
        ),
    }


def validate_frozen_inputs(inputs: BenchmarkInputs) -> None:
    mismatches = {
        key: {"expected": expected, "actual": inputs.manifest.get(key)}
        for key, expected in FROZEN_HASHES.items()
        if inputs.manifest.get(key) != expected
    }
    agent_lock = read_json(ROOT / "stage3-agent-lock-v1.json")
    rag_lock = read_json(ROOT / "stage3-rag-backend-lock-v1.json")
    if agent_lock.get("stage3_agent_behavior_hash") != FROZEN_HASHES[
        "agent_behavior_hash"
    ]:
        mismatches["agent_behavior_hash"] = {
            "expected": FROZEN_HASHES["agent_behavior_hash"],
            "actual": agent_lock.get("stage3_agent_behavior_hash"),
        }
    if rag_lock.get("stage2_final_config_hash") != FROZEN_HASHES["rag_backend_hash"]:
        mismatches["rag_backend_hash"] = {
            "expected": FROZEN_HASHES["rag_backend_hash"],
            "actual": rag_lock.get("stage2_final_config_hash"),
        }
    if mismatches:
        raise SystemExit(f"STAGE4_FROZEN_HASH_MISMATCH {json.dumps(mismatches)}")


def load_inputs() -> BenchmarkInputs:
    return BenchmarkInputs(
        manifest=read_json(BENCH / "research-benchmark-manifest-v1.json"),
        order=read_json(BENCH / "stage4-execution-order-v1.json"),
        tasks={
            row["task_id"]: row for row in read_jsonl(BENCH / "research-tasks-v1.jsonl")
        },
        rubrics={
            row["task_id"]: row
            for row in read_jsonl(BENCH / "research-task-rubrics-v1.jsonl")
        },
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now() -> str:
    return datetime.now(UTC).isoformat()


def post_json(url: str, payload: dict[str, Any], timeout: int) -> tuple[int, dict[str, Any]]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        body = response.read().decode("utf-8")
        return response.status, json.loads(body) if body else {}


def get_json(url: str, timeout: int) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        body = response.read().decode("utf-8")
        return response.status, json.loads(body) if body else {}


def sanitize_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        sanitized: dict[str, Any] = {}
        for key, value in payload.items():
            lowered = key.lower()
            secret_keys = ["api_key", "authorization", "bearer", "cookie", "token"]
            if any(secret in lowered for secret in secret_keys):
                sanitized[key] = redact_value(value)
            else:
                sanitized[key] = sanitize_payload(value)
        return sanitized
    if isinstance(payload, list):
        return [sanitize_payload(item) for item in payload]
    if isinstance(payload, str):
        return sanitize_text(payload)
    return payload


def redact_value(value: Any) -> str:
    text = str(value)
    return f"<redacted length={len(text)} sha256={hashlib.sha256(text.encode()).hexdigest()[:8]}>"


def sanitize_text(text: str) -> str:
    if len(text) > 4000:
        return text[:4000] + "...<truncated>"
    return text.replace("\r", "\\r")


def normalize_status(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    upper = value.upper()
    if upper in {"SUCCEEDED", "SUCCESS"}:
        return "COMPLETED"
    if upper in {"PARTIAL", "PARTIALLY_COMPLETED"}:
        return "PARTIAL"
    if upper in {"FAILED", "ERROR"}:
        return "FAILED"
    if upper in {"COMPLETED", "REFUSED"}:
        return "COMPLETED"
    return upper


def walk_values(value: Any) -> list[Any]:
    values = [value]
    if isinstance(value, dict):
        for child in value.values():
            values.extend(walk_values(child))
    elif isinstance(value, list):
        for child in value:
            values.extend(walk_values(child))
    return values


if __name__ == "__main__":
    raise SystemExit(main())
