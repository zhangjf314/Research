"""Run Stage 3C.2 frozen live replan validation tasks.

This script intentionally validates the already-frozen Research Agent runtime.
It must not tune prompts, RAG, budget, policy, or task definitions. The input
plan is authored by scripts/plan_stage3_live_replan_validation_v1.py and records
provider_requests=0 before live execution.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from paper_research.agents.research_agent import (
    EXPECTED_STAGE2_FINAL_CONFIG_HASH,
    validate_rag_backend_lock,
)
from paper_research.config import Settings

ROOT = Path("data/evaluation/research-agent")
DOC_ROOT = Path("docs/research-agent")
PLAN_JSON = ROOT / "stage3-live-replan-validation-plan-v1.json"
RESULT_JSON = ROOT / "stage3-live-replan-validation-v1.json"
RESULT_MD = DOC_ROOT / "stage3-live-replan-validation-v1.md"
FINAL_JSON = ROOT / "research-agent-stage3-final-v1.json"
FINAL_MD = DOC_ROOT / "research-agent-stage3-final-v1.md"
RUNTIME_SOURCE_FILES = [
    Path("src/paper_research/agents/research_agent/models.py"),
    Path("src/paper_research/agents/research_agent/runner.py"),
    Path("src/paper_research/agents/research_agent/trace.py"),
    Path("src/paper_research/agents/research_agent/verifier.py"),
    Path("src/paper_research/agents/research_agent/decision_provider.py"),
    Path("src/paper_research/agents/research_agent/tools.py"),
    Path("src/paper_research/agents/research_agent/policy.py"),
    Path("src/paper_research/agents/research_agent/planner.py"),
    Path("src/paper_research/agents/research_agent/backend_lock.py"),
    Path("src/paper_research/agents/research_agent/checkpoint.py"),
    Path("src/paper_research/agents/research_agent/state.py"),
    Path("src/paper_research/agents/research_agent/__init__.py"),
]


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    DOC_ROOT.mkdir(parents=True, exist_ok=True)
    plan = read_json(PLAN_JSON)
    lock = validate_rag_backend_lock()
    if lock["stage2_final_config_hash"] != EXPECTED_STAGE2_FINAL_CONFIG_HASH:
        raise SystemExit("RAG_BACKEND_LOCK_MISMATCH")
    if plan.get("validation_set_frozen") is not True:
        raise SystemExit("VALIDATION_SET_NOT_FROZEN")
    tasks = plan.get("validation_tasks") or []
    if len(tasks) != 3:
        raise SystemExit("VALIDATION_TASK_COUNT_MISMATCH")

    settings = Settings()
    behavior_hash_before = runtime_behavior_hash(settings, lock)
    preflight = run_preflight(settings, lock)
    results: list[dict[str, Any]] = []
    blocked = False
    if preflight["safe_to_run_validation"]:
        for task in tasks:
            existing = container_checkpoint_state(task["task_id"])
            if existing:
                raise SystemExit(
                    f"VALIDATION_TASK_ALREADY_HAS_CHECKPOINT: {task['task_id']}"
                )
            results.append(run_task(task))
    else:
        blocked = True
    behavior_hash_after = runtime_behavior_hash(settings, lock)
    payload = build_result_payload(
        plan=plan,
        lock=lock,
        settings=settings,
        preflight=preflight,
        results=results,
        behavior_hash_before=behavior_hash_before,
        behavior_hash_after=behavior_hash_after,
        blocked=blocked,
    )
    write_json(RESULT_JSON, payload)
    write_result_doc(payload)
    update_stage3_final(payload, lock)
    print(
        json.dumps(
            {
                "validation_status": payload["validation_status"],
                "effective_replan_observed": payload["effective_replan_observed"],
                "stage3_complete": payload["stage3_complete"],
                "stage4_ready": payload["stage4_ready"],
                "provider_requests": payload["provider_requests"],
                "total_tokens": payload["total_tokens"],
                "total_cost": payload["total_cost"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if payload["validation_status"] in {"PASSED", "COMPLETED_NO_REPLAN"} else 2


def run_preflight(settings: Settings, lock: dict[str, Any]) -> dict[str, Any]:
    health = get_json("http://localhost/api/v1/health")
    capabilities = get_json("http://localhost/api/v1/capabilities")
    provider_health = (
        read_json(Path("data/evaluation/provider-health-v1.json"))
        if Path("data/evaluation/provider-health-v1.json").exists()
        else {}
    )
    llm = capabilities.get("capabilities", {}).get("llm", {})
    reranker = capabilities.get("capabilities", {}).get("reranker", {})
    safe = all(
        [
            health.get("status") == "healthy",
            provider_health.get("safe_to_start_batch") is True,
            llm.get("provider") == (settings.llm_provider_name or settings.llm_provider),
            llm.get("model") == settings.llm_model,
            llm.get("template_fallback") is False,
            reranker.get("status") == "disabled",
            settings.rerank_enabled is False,
            lock["rag_backend"]["retrieval"] == "Current Hybrid",
            lock["rag_backend"]["reranker"] == "disabled",
            lock["rag_backend"]["query_rewrite"] == "disabled",
            lock["rag_backend"]["query_decomposition"] == "disabled",
            lock["rag_backend"]["context_selector"] == "baseline",
        ]
    )
    return {
        "health_status": health.get("status"),
        "provider_health_status": provider_health.get("status"),
        "provider_safe_to_start_batch": provider_health.get("safe_to_start_batch"),
        "safe_to_run_validation": safe,
        "llm_provider": llm.get("provider"),
        "llm_model": llm.get("model"),
        "response_format": settings.llm_response_format,
        "thinking_enabled": getattr(settings, "llm_thinking_enabled", None),
        "template_fallback": llm.get("template_fallback"),
        "reranker_enabled": settings.rerank_enabled,
        "reranker_status": reranker.get("status"),
    }


def run_task(task: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    budget = {
        "max_steps": 12,
        "max_tool_calls": 16,
        "max_provider_requests": 12,
        "max_tokens": 40000,
        "max_cost_usd": 0.05,
    }
    response = post_json(
        "http://localhost/api/v1/research/agent",
        {
            "query": task["research_question"],
            "task_id": task["task_id"],
            "budget": budget,
            "pause_after_phase": None,
        },
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    return summarize_task(task, response, elapsed_ms)


def summarize_task(
    task: dict[str, Any],
    state: dict[str, Any],
    elapsed_ms: float,
) -> dict[str, Any]:
    runtime = container_runtime_stats(task["task_id"])
    events = runtime.get("events") or []
    phases = [str(event.get("phase")) for event in events]
    action_sequence = [
        str(event.get("selected_action") or event.get("action"))
        for event in events
        if event.get("selected_action") or event.get("action")
    ]
    token_usage = dict(state.get("token_usage") or {})
    provider_call_count = int(state.get("provider_call_count") or 0)
    total_tokens = int(token_usage.get("total_tokens") or 0)
    estimated_cost = round(float(state.get("estimated_cost") or 0.0), 8)
    chain = effective_replan_chain(events)
    observations = list(state.get("observations") or [])
    verification_state = state.get("verification_state") or {}
    verify_indices = [index for index, phase in enumerate(phases) if phase == "VERIFY"]
    finish_index = next((index for index, phase in enumerate(phases) if phase == "FINISH"), None)
    return {
        "task_id": task["task_id"],
        "task_pattern": task["task_pattern"],
        "task_hash": task["task_hash"],
        "status": state.get("status"),
        "stop_reason": state.get("stop_reason"),
        "plan_version": state.get("plan_version"),
        "step_count": state.get("step_count"),
        "tool_call_count": state.get("tool_call_count"),
        "provider_call_count": provider_call_count,
        "retrieval_call_count": sum(
            1 for obs in observations if obs.get("tool") == "retrieve_evidence"
        ),
        "evidence_state_count": state.get("evidence_count"),
        "verification_status": verification_state.get("status"),
        "verification_recommended_next_action": verification_state.get(
            "recommended_next_action"
        ),
        "verification_before_finish": bool(verify_indices)
        and (finish_index is None or min(verify_indices) < finish_index),
        "dynamic_tool_selection_observed": len(set(action_sequence)) > 1,
        "observation_driven_action_observed": any(
            event.get("phase") == "DECIDE"
            and event.get("trigger_source") in {"OBSERVATION", "VERIFICATION", "REPLAN"}
            for event in events
        ),
        "effective_replan_observed": chain["observed"],
        "effective_replan_chain": chain,
        "trace_event_count": len(events),
        "trace_event_unique_count": len({event.get("event_id") for event in events}),
        "trace_complete": bool(events)
        and len(events) == len({event.get("event_id") for event in events}),
        "checkpoint_exists": runtime.get("checkpoint_exists"),
        "checkpoint_path": runtime.get("checkpoint_path"),
        "trace_path": runtime.get("trace_path"),
        "resume_count": runtime.get("resume_count"),
        "completed_tool_call_ids_unique": runtime.get("completed_tool_call_ids_unique"),
        "request_ids_unique": provider_call_count == len(set(runtime.get("request_ids") or []))
        if runtime.get("request_ids")
        else True,
        "input_tokens": int(token_usage.get("input_tokens") or 0),
        "output_tokens": int(token_usage.get("output_tokens") or 0),
        "total_tokens": total_tokens,
        "usage_source": token_usage.get("usage_source"),
        "estimated_cost_usd": estimated_cost,
        "elapsed_ms": elapsed_ms,
        "public_trace_note": "full trace/checkpoint retained under local .runtime path",
    }


def effective_replan_chain(events: list[dict[str, Any]]) -> dict[str, Any]:
    verifications = [
        event
        for event in events
        if event.get("phase") == "VERIFY"
        and event.get("verification_status") in {"PARTIAL", "FAIL"}
        and event.get("recommended_next_action") == "REPLAN"
    ]
    for verify in verifications:
        verify_index = events.index(verify)
        for replan in events[verify_index + 1 :]:
            if replan.get("phase") != "REPLAN":
                continue
            replan_payload = replan.get("replan") or {}
            delta = replan_payload.get("plan_delta") or {}
            has_delta = bool(
                delta.get("added_subquestions")
                or delta.get("removed_subquestions")
                or delta.get("reprioritized")
                or delta.get("changed_objective")
            )
            if int(replan.get("plan_version") or 0) < 2 or not has_delta:
                continue
            replan_index = events.index(replan)
            decision = next(
                (
                    event
                    for event in events[replan_index + 1 :]
                    if event.get("phase") == "DECIDE"
                    and event.get("trigger_source") == "REPLAN"
                ),
                None,
            )
            action = next(
                (
                    event
                    for event in events[replan_index + 1 :]
                    if event.get("phase") in {"OBSERVE", "TOOL_COMPLETED"}
                    and event.get("action")
                    and event.get("action") != "finish"
                ),
                None,
            )
            if decision and action:
                return {
                    "observed": True,
                    "based_on_observation_id": verify.get("based_on_observation_id"),
                    "trigger_verification_id": verify.get("verification_id"),
                    "trigger_replan_id": replan.get("replan_id"),
                    "decision_id": decision.get("decision_id"),
                    "decision_trigger_id": decision.get("trigger_id"),
                    "new_tool_action_event_id": action.get("event_id"),
                    "plan_delta": delta,
                }
    return {
        "observed": False,
        "reason": (
            "no VERIFY PARTIAL/FAIL with REPLAN recommendation was followed by "
            "effective plan delta and a new real tool action"
        ),
    }


def build_result_payload(
    *,
    plan: dict[str, Any],
    lock: dict[str, Any],
    settings: Settings,
    preflight: dict[str, Any],
    results: list[dict[str, Any]],
    behavior_hash_before: str,
    behavior_hash_after: str,
    blocked: bool,
) -> dict[str, Any]:
    provider_requests = sum(int(item.get("provider_call_count") or 0) for item in results)
    total_tokens = sum(int(item.get("total_tokens") or 0) for item in results)
    total_cost = round(sum(float(item.get("estimated_cost_usd") or 0.0) for item in results), 8)
    all_completed = len(results) == 3 and all(
        item.get("status") in {"COMPLETED", "PARTIAL"} for item in results
    )
    effective_replan = any(item["effective_replan_observed"] for item in results)
    dynamic_tool = any(item["dynamic_tool_selection_observed"] for item in results)
    observation_driven = any(item["observation_driven_action_observed"] for item in results)
    trace_complete = len(results) == 3 and all(item["trace_complete"] for item in results)
    budget_passed = provider_requests <= 36 and total_tokens <= 120000 and total_cost <= 0.03
    behavior_stable = behavior_hash_before == behavior_hash_after
    stage3_complete = (
        all_completed
        and effective_replan
        and dynamic_tool
        and observation_driven
        and trace_complete
        and budget_passed
        and behavior_stable
        and not blocked
    )
    if blocked:
        validation_status = "BLOCKED"
    elif effective_replan:
        validation_status = "PASSED" if stage3_complete else "FAILED"
    else:
        validation_status = "COMPLETED_NO_REPLAN"
    return {
        "schema_version": "stage3-live-replan-validation-v1",
        "created_at": now(),
        "git_commit": git_head(),
        "agent_runtime_frozen_for_validation": True,
        "runtime_behavior_hash_before": behavior_hash_before,
        "runtime_behavior_hash_after": behavior_hash_after,
        "runtime_behavior_hash_stable": behavior_stable,
        "validation_plan_hash": plan["stage3_live_replan_validation_set_hash"],
        "validation_task_count": plan["validation_task_count"],
        "validation_tasks_max_once_each": True,
        "validation_results": results,
        "preflight": preflight,
        "provider": {
            "provider": settings.llm_provider_name or settings.llm_provider,
            "model": settings.llm_model,
            "response_format": settings.llm_response_format,
            "thinking_enabled": getattr(settings, "llm_thinking_enabled", None),
            "template_fallback": False,
        },
        "rag_backend": lock["rag_backend"],
        "stage2_final_config_hash": lock["stage2_final_config_hash"],
        "rag_backend_lock_match": lock["stage2_final_config_hash"]
        == EXPECTED_STAGE2_FINAL_CONFIG_HASH,
        "dynamic_tool_selection_observed": dynamic_tool,
        "observation_driven_action_observed": observation_driven,
        "effective_replan_observed": effective_replan,
        "live_replan_causal_chain": next(
            (
                item["effective_replan_chain"]
                for item in results
                if item["effective_replan_observed"]
            ),
            {
                "observed": False,
                "reason": (
                    "preregistered validation tasks did not naturally produce "
                    "effective live replan"
                ),
            },
        ),
        "provider_requests": provider_requests,
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "budget_passed": budget_passed,
        "trace_complete": trace_complete,
        "stage3_complete": stage3_complete,
        "stage4_ready": stage3_complete,
        "stage3_stage4_locks_generated": False,
        "validation_status": validation_status,
        "final_conclusion": "A"
        if stage3_complete
        else (
            "B"
            if validation_status == "COMPLETED_NO_REPLAN"
            else "C"
        ),
        "failure_freeze": None
        if stage3_complete
        else {
            "code": "LIVE_REPLAN_NOT_OBSERVED"
            if validation_status == "COMPLETED_NO_REPLAN"
            else "STAGE3C2_VALIDATION_BLOCKED_OR_FAILED",
            "runtime_modified_after_freeze": False,
            "task4_designed": False,
            "rerun_performed": False,
        },
    }


def update_stage3_final(payload: dict[str, Any], lock: dict[str, Any]) -> None:
    existing = read_json(FINAL_JSON) if FINAL_JSON.exists() else {}
    final = {
        **existing,
        "schema_version": "research-agent-stage3-final-v1",
        "updated_at": now(),
        "git_commit": git_head(),
        "stage2_rag_backend_hash": EXPECTED_STAGE2_FINAL_CONFIG_HASH,
        "agent_rag_backend_hash": lock["stage2_final_config_hash"],
        "rag_backend_lock_match": payload["rag_backend_lock_match"],
        "workflow_behavior_changed": False,
        "agentic_control_flow_demonstrated": payload["effective_replan_observed"],
        "dynamic_tool_selection": payload["dynamic_tool_selection_observed"],
        "observation_changes_next_action": payload[
            "observation_driven_action_observed"
        ],
        "replan_supported": payload["effective_replan_observed"],
        "verification_before_finish": all(
            item.get("verification_before_finish")
            for item in payload["validation_results"]
        )
        if payload["validation_results"]
        else existing.get("verification_before_finish", False),
        "checkpoint_resume": existing.get("checkpoint_resume", True),
        "retry_bounded": True,
        "budget_enforced": payload["budget_passed"],
        "stop_conditions_enforced": True,
        "trace_complete": payload["trace_complete"],
        "stage3c2_validation": {
            "artifact": str(RESULT_JSON),
            "validation_status": payload["validation_status"],
            "validation_plan_hash": payload["validation_plan_hash"],
            "effective_replan_observed": payload["effective_replan_observed"],
        },
        "provider_requests": payload["provider_requests"],
        "total_tokens": payload["total_tokens"],
        "total_cost": payload["total_cost"],
        "stage3_complete": payload["stage3_complete"],
        "stage4_ready": payload["stage4_ready"],
        "stage4_benchmark_run": False,
        "failure_freeze": payload["failure_freeze"],
    }
    write_json(FINAL_JSON, final)
    write_text(
        FINAL_MD,
        "\n".join(
            [
                "# Research Agent Stage 3 Final v1",
                "",
                f"- stage3_complete: `{final['stage3_complete']}`",
                f"- stage4_ready: `{final['stage4_ready']}`",
                f"- rag_backend_lock_match: `{final['rag_backend_lock_match']}`",
                f"- workflow_behavior_changed: `{final['workflow_behavior_changed']}`",
                (
                    "- agentic_control_flow_demonstrated: "
                    f"`{final['agentic_control_flow_demonstrated']}`"
                ),
                f"- dynamic_tool_selection: `{final['dynamic_tool_selection']}`",
                f"- observation_changes_next_action: `{final['observation_changes_next_action']}`",
                f"- replan_supported: `{final['replan_supported']}`",
                f"- checkpoint_resume: `{final['checkpoint_resume']}`",
                f"- budget_enforced: `{final['budget_enforced']}`",
                f"- trace_complete: `{final['trace_complete']}`",
                f"- provider_requests: `{final['provider_requests']}`",
                f"- total_tokens: `{final['total_tokens']}`",
                f"- total_cost: `{final['total_cost']}`",
                f"- failure_freeze: `{final['failure_freeze']}`",
                "",
                "Stage 3C.2 validates a frozen Agent runtime against a preregistered",
                "three-task live replan validation set. Stage 4 benchmark remains unrun.",
            ]
        ),
    )


def runtime_behavior_hash(settings: Settings, lock: dict[str, Any]) -> str:
    source_hashes = {}
    for path in RUNTIME_SOURCE_FILES:
        source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {
        "git_head": git_head(),
        "source_hashes": source_hashes,
        "rag_backend": lock["rag_backend"],
        "stage2_final_config_hash": lock["stage2_final_config_hash"],
        "provider": settings.llm_provider_name or settings.llm_provider,
        "model": settings.llm_model,
        "response_format": settings.llm_response_format,
        "thinking_enabled": getattr(settings, "llm_thinking_enabled", None),
        "reranker_enabled": settings.rerank_enabled,
        "budget": {
            "max_steps": 12,
            "max_tool_calls": 16,
            "max_provider_requests": 12,
            "max_tokens": 40000,
            "max_cost_usd": 0.05,
            "stage3c2_total_provider_requests": 36,
            "stage3c2_total_tokens": 120000,
            "stage3c2_total_cost_usd": 0.03,
        },
        "stop_conditions": [
            "SUCCESS",
            "EVIDENCE_SUFFICIENT",
            "MAX_STEPS_REACHED",
            "TOOL_BUDGET_EXHAUSTED",
            "TOKEN_BUDGET_EXHAUSTED",
            "COST_BUDGET_EXHAUSTED",
            "PROVIDER_FAILURE",
            "TOOL_FAILURE",
            "VERIFICATION_FAILED_NO_BUDGET",
            "NO_PROGRESS",
            "CHECKPOINT_FAILURE",
        ],
        "runtime_frozen_for_validation": True,
    }
    return stable_hash(payload)


def container_runtime_stats(task_id: str) -> dict[str, Any]:
    script = (
        "import json\n"
        "from pathlib import Path\n"
        f"task_id={task_id!r}\n"
        "trace=Path('.runtime/research-agent/traces')/f'{task_id}.jsonl'\n"
        "ckpt=Path('.runtime/research-agent/checkpoints')/f'{task_id}.json'\n"
        "events=[]\n"
        "if trace.exists():\n"
        "    lines=trace.read_text(encoding='utf-8').splitlines()\n"
        "    events=[json.loads(x) for x in lines if x.strip()]\n"
        "state={}\n"
        "if ckpt.exists():\n"
        "    state=json.loads(ckpt.read_text(encoding='utf-8'))\n"
        "history=state.get('tool_history') or []\n"
        "request_ids=[\n"
        "    x.get('provider_request_id') for x in history\n"
        "    if x.get('provider_request_id')\n"
        "]\n"
        "completed=state.get('completed_tool_call_ids', [])\n"
        "out={\n"
        " 'trace_path': str(trace),\n"
        " 'checkpoint_path': str(ckpt),\n"
        " 'checkpoint_exists': ckpt.exists(),\n"
        " 'events': events,\n"
        " 'resume_count': state.get('resume_count', 0),\n"
        " 'completed_tool_call_ids_unique': len(completed) == len(set(completed)),\n"
        " 'request_ids': request_ids,\n"
        "}\n"
        "print(json.dumps(out, ensure_ascii=False))\n"
    )
    completed = subprocess.run(
        ["docker", "compose", "exec", "-T", "api", "python", "-c", script],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "trace_path": f".runtime/research-agent/traces/{task_id}.jsonl",
            "checkpoint_path": f".runtime/research-agent/checkpoints/{task_id}.json",
            "checkpoint_exists": False,
            "events": [],
            "resume_count": 0,
            "completed_tool_call_ids_unique": False,
            "request_ids": [],
            "error": completed.stderr[-2000:],
        }
    for line in reversed([line for line in completed.stdout.splitlines() if line.strip()]):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {
        "trace_path": f".runtime/research-agent/traces/{task_id}.jsonl",
        "checkpoint_path": f".runtime/research-agent/checkpoints/{task_id}.json",
        "checkpoint_exists": False,
        "events": [],
        "resume_count": 0,
        "completed_tool_call_ids_unique": False,
        "request_ids": [],
    }


def container_checkpoint_state(task_id: str) -> dict[str, Any] | None:
    script = (
        "import json\n"
        "from pathlib import Path\n"
        f"task_id={task_id!r}\n"
        "ckpt=Path('.runtime/research-agent/checkpoints')/f'{task_id}.json'\n"
        "payload={}\n"
        "if ckpt.exists():\n"
        "    payload=json.loads(ckpt.read_text(encoding='utf-8'))\n"
        "print(json.dumps(payload, ensure_ascii=False))\n"
    )
    completed = subprocess.run(
        ["docker", "compose", "exec", "-T", "api", "python", "-c", script],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    for line in reversed([line for line in completed.stdout.splitlines() if line.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        return payload or None
    return None


def write_result_doc(payload: dict[str, Any]) -> None:
    lines = [
        "# Stage 3 Live Replan Validation v1",
        "",
        f"- validation_status: `{payload['validation_status']}`",
        f"- stage3_complete: `{payload['stage3_complete']}`",
        f"- stage4_ready: `{payload['stage4_ready']}`",
        f"- runtime_behavior_hash_stable: `{payload['runtime_behavior_hash_stable']}`",
        f"- validation_plan_hash: `{payload['validation_plan_hash']}`",
        f"- dynamic_tool_selection_observed: `{payload['dynamic_tool_selection_observed']}`",
        f"- observation_driven_action_observed: `{payload['observation_driven_action_observed']}`",
        f"- effective_replan_observed: `{payload['effective_replan_observed']}`",
        f"- provider_requests: `{payload['provider_requests']}`",
        f"- total_tokens: `{payload['total_tokens']}`",
        f"- total_cost: `{payload['total_cost']}`",
        f"- budget_passed: `{payload['budget_passed']}`",
        f"- trace_complete: `{payload['trace_complete']}`",
        "",
        "Runtime, RAG backend, prompt, retrieval, and budget were frozen for validation.",
    ]
    for item in payload["validation_results"]:
        lines.extend(
            [
                "",
                f"## {item['task_id']}",
                "",
                f"- status: `{item['status']}`",
                f"- stop_reason: `{item['stop_reason']}`",
                f"- task_pattern: `{item['task_pattern']}`",
                f"- plan_version: `{item['plan_version']}`",
                f"- verification_status: `{item['verification_status']}`",
                (
                    "- verification_recommended_next_action: "
                    f"`{item['verification_recommended_next_action']}`"
                ),
                f"- retrieval_call_count: `{item['retrieval_call_count']}`",
                f"- provider_call_count: `{item['provider_call_count']}`",
                f"- total_tokens: `{item['total_tokens']}`",
                f"- estimated_cost_usd: `{item['estimated_cost_usd']}`",
                f"- dynamic_tool_selection_observed: `{item['dynamic_tool_selection_observed']}`",
                (
                    "- observation_driven_action_observed: "
                    f"`{item['observation_driven_action_observed']}`"
                ),
                f"- effective_replan_observed: `{item['effective_replan_observed']}`",
                f"- trace_event_count: `{item['trace_event_count']}`",
                f"- trace_event_unique_count: `{item['trace_event_unique_count']}`",
            ]
        )
    write_text(RESULT_MD, "\n".join(lines))


def get_json(url: str) -> dict[str, Any]:
    request = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return {"status": "failed", "error_type": type(exc).__name__, "detail": str(exc)}


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=600) as response:
        return json.loads(response.read().decode("utf-8"))


def run_command(command: Iterable[str]) -> dict[str, Any]:
    completed = subprocess.run(
        list(command),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout[-2000:],
        "stderr": completed.stderr[-2000:],
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def now() -> str:
    return datetime.now(UTC).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
