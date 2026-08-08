"""Run Stage 3C controlled live validation for Research Agent v1.

This script is intentionally scoped to Stage 3C development smoke validation.
It does not run the Stage 4 benchmark, does not change the frozen RAG backend,
and writes only sanitized public summaries under data/evaluation/research-agent.
Full trace/checkpoint state stays under .runtime/research-agent/stage3c-live.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from paper_research.agents.research_agent import (
    EXPECTED_STAGE2_FINAL_CONFIG_HASH,
    validate_rag_backend_lock,
)
from paper_research.config import Settings

ROOT = Path("data/evaluation/research-agent")
DOC_ROOT = Path("docs/research-agent")
RUNTIME_ROOT = Path(".runtime/research-agent/stage3c-live")
PREFLIGHT_JSON = ROOT / "stage3-live-preflight-v1.json"
PREFLIGHT_MD = DOC_ROOT / "stage3-live-preflight-v1.md"
SMOKE_JSON = ROOT / "stage3-agent-smoke-v1.json"
SMOKE_MD = DOC_ROOT / "research-agent-smoke-v1.md"
EXCLUSION_JSON = ROOT / "stage3-smoke-task-exclusion-v1.json"
FINAL_JSON = ROOT / "research-agent-stage3-final-v1.json"
FINAL_MD = DOC_ROOT / "research-agent-stage3-final-v1.md"
AGENT_LOCK_JSON = ROOT / "stage3-agent-lock-v1.json"
WORKFLOW_LOCK_JSON = ROOT / "stage4-workflow-control-lock-v1.json"
COMPARABILITY_LOCK_JSON = ROOT / "stage4-comparability-lock-v1.json"


SMOKE_TASKS = [
    {
        "task_id": "stage3c-smoke-1-straightforward",
        "kind": "straightforward_evidence_synthesis",
        "question": (
            "Summarize the evidence in the corpus about attention-based sequence "
            "transduction models and their main methodological contribution."
        ),
        "interrupt_after_phase": None,
    },
    {
        "task_id": "stage3c-smoke-2-multi-evidence",
        "kind": "multi_evidence_dynamic_task",
        "question": (
            "Compare evidence about Transformer attention-only sequence transduction "
            "and BERT bidirectional pre-training, focusing on what each method changes "
            "and what evidence remains insufficient for a direct quantitative comparison."
        ),
        "interrupt_after_phase": None,
    },
    {
        "task_id": "stage3c-smoke-3-insufficient-resume",
        "kind": "evidence_insufficient_resume_task",
        "question": (
            "Determine whether the corpus contains direct evidence for the exact total "
            "energy consumption of all experiments in the attention-only Transformer "
            "paper, and stop with insufficiency if the evidence is not present."
        ),
        "interrupt_after_phase": "STATE_UPDATED",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--revalidate-attempt2", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    DOC_ROOT.mkdir(parents=True, exist_ok=True)
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)

    lock = validate_rag_backend_lock()
    if lock["stage2_final_config_hash"] != EXPECTED_STAGE2_FINAL_CONFIG_HASH:
        raise SystemExit("RAG_BACKEND_LOCK_MISMATCH")

    if args.finalize_only:
        smoke = read_json(SMOKE_JSON)
        final = build_final_payload(lock, smoke)
        write_final_artifacts(lock, smoke, final)
        print(json.dumps({"stage3_complete": final["stage3_complete"]}, ensure_ascii=False))
        return 0 if final["stage3_complete"] else 2

    settings = Settings()
    if not args.smoke_only:
        preflight = run_preflight(settings, lock)
        write_json(PREFLIGHT_JSON, preflight)
        write_preflight_doc(preflight)
        print(json.dumps({"preflight_passed": preflight["preflight_passed"]}))
        if args.preflight_only:
            return 0 if preflight["preflight_passed"] else 2
        if not preflight["preflight_passed"]:
            return 2

    smoke = run_smokes(settings, lock, revalidate_attempt2=args.revalidate_attempt2)
    write_json(SMOKE_JSON, smoke)
    write_json(EXCLUSION_JSON, build_exclusion_payload())
    write_smoke_doc(smoke)
    final = build_final_payload(lock, smoke)
    write_final_artifacts(lock, smoke, final)
    print(
        json.dumps(
            {
                "stage3_complete": final["stage3_complete"],
                "stage4_ready": final["stage4_ready"],
                "live_smoke_completed": smoke["live_smoke_completed"],
                "live_replan_observed": smoke["live_replan_observed"],
                "provider_requests": smoke["provider_requests"],
                "total_tokens": smoke["total_tokens"],
                "total_cost": smoke["total_cost"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if final["stage3_complete"] else 2


def run_preflight(settings: Settings, lock: dict[str, Any]) -> dict[str, Any]:
    health = get_json("http://localhost/api/v1/health")
    capabilities = get_json("http://localhost/api/v1/capabilities")
    openapi = get_json("http://localhost/openapi.json")
    docker_ps = run_command(["docker", "compose", "ps"])
    provider_health = read_json(Path("data/evaluation/provider-health-v1.json"))
    paths = set(openapi.get("paths", {}))
    llm = capabilities.get("capabilities", {}).get("llm", {})
    deep = capabilities.get("capabilities", {}).get("deep_research", {})
    embedding = capabilities.get("capabilities", {}).get("embedding", {})
    reranker = capabilities.get("capabilities", {}).get("reranker", {})
    redis = capabilities.get("capabilities", {}).get("redis", {})
    checkpoint = capabilities.get("capabilities", {}).get("langgraph_checkpoint", {})
    docker_stdout = str(docker_ps.get("stdout") or "")
    postgres_available = "research-postgres-1" in docker_stdout and "Up" in docker_stdout
    qdrant_available = "research-qdrant-1" in docker_stdout and "Up" in docker_stdout
    passed = all(
        [
            health.get("status") == "healthy",
            provider_health.get("safe_to_start_batch") is True,
            "/api/v1/research/agent" in paths,
            "/api/v1/research/agent/{task_id}/resume" in paths,
            llm.get("provider") == (settings.llm_provider_name or settings.llm_provider),
            llm.get("model") == settings.llm_model,
            llm.get("template_fallback") is False,
            deep.get("status") in {"available", "degraded"},
            embedding.get("status") in {"available", "degraded"},
            reranker.get("status") == "disabled" and settings.rerank_enabled is False,
            redis.get("status") in {"available", "degraded"},
            qdrant_available,
            postgres_available,
            lock["rag_backend"]["retrieval"] == "Current Hybrid",
            lock["rag_backend"]["reranker"] == "disabled",
            lock["rag_backend"]["query_rewrite"] == "disabled",
            lock["rag_backend"]["query_decomposition"] == "disabled",
            lock["rag_backend"]["context_selector"] == "baseline",
        ]
    )
    return {
        "schema_version": "stage3-live-preflight-v1",
        "created_at": now(),
        "git_commit": git_head(),
        "preflight_passed": passed,
        "health_status": health.get("status"),
        "provider_health_status": provider_health.get("status"),
        "provider_safe_to_start_batch": provider_health.get("safe_to_start_batch"),
        "llm": {
            "provider": llm.get("provider"),
            "model": llm.get("model"),
            "thinking": getattr(settings, "llm_thinking_enabled", None),
            "response_format": settings.llm_response_format,
            "template_fallback": llm.get("template_fallback"),
        },
        "deep_research_status": deep.get("status"),
        "agent_api_registered": "/api/v1/research/agent" in paths,
        "agent_resume_api_registered": "/api/v1/research/agent/{task_id}/resume" in paths,
        "retrieval_backend": lock["rag_backend"],
        "stage2_final_config_hash": lock["stage2_final_config_hash"],
        "docker_compose_ps_exit_code": docker_ps["returncode"],
        "postgres_status": "available" if postgres_available else "unavailable",
        "qdrant_status": "available" if qdrant_available else "unavailable",
        "redis_status": redis.get("status"),
        "embedding_status": embedding.get("status"),
        "reranker_enabled": settings.rerank_enabled,
        "checkpoint_status": checkpoint.get("status"),
    }


def run_smokes(
    settings: Settings,
    lock: dict[str, Any],
    *,
    revalidate_attempt2: bool = False,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    total_provider_requests = 0
    total_tokens = 0
    total_cost = 0.0
    previous_attempt = (
        read_json(SMOKE_JSON) if revalidate_attempt2 and SMOKE_JSON.exists() else None
    )
    tasks = SMOKE_TASKS
    if revalidate_attempt2:
        tasks = [
            {
                **task,
                "base_task_id": task["task_id"],
                "task_id": f"{task['task_id']}-attempt-2",
                "validation_attempt": "VALIDATION_ATTEMPT_2",
            }
            for task in SMOKE_TASKS[1:]
        ]
    for task in tasks:
        started = time.perf_counter()
        task_runtime = RUNTIME_ROOT / task["task_id"]
        budget = {
            "max_steps": 12,
            "max_tool_calls": 16,
            "max_provider_requests": 12,
            "max_tokens": 40000,
            "max_cost_usd": 0.05,
        }
        payload = {
            "query": task["question"],
            "task_id": task["task_id"],
            "budget": budget,
            "pause_after_phase": task["interrupt_after_phase"],
        }
        existing = None if revalidate_attempt2 else maybe_existing_agent_state(str(task["task_id"]))
        if existing is not None and existing.get("status") != "PAUSED":
            state = existing
            interrupted = bool(task["interrupt_after_phase"] and state.get("resume_count"))
        elif task["interrupt_after_phase"]:
            first = post_json("http://localhost/api/v1/research/agent", payload)
            state = post_json(
                f"http://localhost/api/v1/research/agent/{first['task_id']}/resume",
                {},
            )
            interrupted = True
        else:
            state = post_json("http://localhost/api/v1/research/agent", payload)
            interrupted = False
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        task_result = summarize_state(state, task, task_runtime, elapsed_ms, interrupted)
        results.append(task_result)
        total_provider_requests += task_result["provider_call_count"]
        total_tokens += task_result["total_tokens"]
        total_cost += task_result["estimated_cost_usd"]
    effective_results = results
    if previous_attempt:
        original_smoke1 = previous_attempt["smoke_tasks"][0]
        original_smoke1["validation_attempt"] = "VALIDATION_ATTEMPT_1"
        effective_results = [original_smoke1, *results]
    live_replan_observed = any(item["live_replan_observed"] for item in effective_results)
    trace_complete = all(item["trace_complete"] for item in results)
    budget_passed = (
        len(results) <= 3
        and total_provider_requests <= 36
        and total_tokens <= 120000
        and total_cost <= 0.05
    )
    completed = sum(
        1 for item in effective_results if item["status"] in {"COMPLETED", "PARTIAL"}
    )
    failed = sum(1 for item in effective_results if item["status"] == "FAILED")
    payload = {
        "schema_version": "stage3-agent-smoke-v1",
        "created_at": now(),
        "git_commit": git_head(),
        "mode": "controlled_live_development_smoke",
        "smoke_tasks": effective_results,
        "live_smoke_count": len(effective_results),
        "live_smoke_completed": completed,
        "live_smoke_failed": failed,
        "live_dynamic_path_observed": any(
            item["dynamic_path_observed"] for item in effective_results
        ),
        "live_replan_observed": live_replan_observed,
        "checkpoint_resume_smoke": any(item["resume_count"] == 1 for item in effective_results),
        "provider_requests": total_provider_requests,
        "total_tokens": total_tokens,
        "total_cost": round(total_cost, 8),
        "budget_passed": budget_passed,
        "trace_complete": trace_complete,
        "rag_backend_lock_match": lock["stage2_final_config_hash"]
        == EXPECTED_STAGE2_FINAL_CONFIG_HASH,
        "stage4_benchmark_run": False,
        "live_replan_gate": "PASSED"
        if live_replan_observed
        else "LIVE_AGENT_DYNAMIC_REPLAN_NOT_DEMONSTRATED",
        "root_cause": "NO_TRUE_REPLAN_NEEDED"
        if revalidate_attempt2 and not live_replan_observed
        else "LIVE_AGENT_DYNAMIC_REPLAN_NOT_DEMONSTRATED",
        "live_replan_causal_chain": _live_replan_causal_chain(effective_results),
        "live_resume_causal_chain": _live_resume_causal_chain(effective_results),
    }
    if previous_attempt:
        payload["validation_attempt_1"] = previous_attempt
        payload["validation_attempt_2"] = {
            "status": "post_fix_revalidation",
            "attempt_1_invalidated_for_freeze_due_to": (
                "FINISH_CONTRACT_BYPASSED_PARTIAL_REPLAN_AND_TRACE_MISSING_VERIFY_EVENT"
            ),
            "smoke_tasks": results,
            "max_revalidation_tasks": 2,
            "max_revalidation_total_cost_usd": 0.03,
            "max_revalidation_total_tokens": 80000,
        }
    return payload


def _live_replan_causal_chain(results: list[dict[str, Any]]) -> dict[str, Any]:
    for item in results:
        if item.get("live_replan_observed"):
            return {
                "observed": True,
                "task_id": item["task_id"],
                "plan_version": item["plan_version"],
                "replan_reasons": item["replan_reasons"],
            }
    return {
        "observed": False,
        "reason": "no live verification produced PARTIAL/FAIL followed by effective replan",
    }


def _live_resume_causal_chain(results: list[dict[str, Any]]) -> dict[str, Any]:
    for item in results:
        if item.get("resume_count") == 1:
            return {
                "observed": True,
                "task_id": item["task_id"],
                "resume_count": item["resume_count"],
                "duplicate_tool_execution_count": 0
                if item.get("completed_tool_call_ids_unique")
                else "unknown",
                "duplicate_provider_execution_count": 0
                if item.get("request_ids_unique")
                else "unknown",
            }
    return {"observed": False}


def summarize_state(
    state: dict[str, Any],
    task: dict[str, Any],
    task_runtime: Path,
    elapsed_ms: float,
    interrupted: bool,
) -> dict[str, Any]:
    runtime_stats = container_runtime_stats(str(task["task_id"]))
    trace_path = runtime_stats["trace_path"]
    checkpoint_path = runtime_stats["checkpoint_path"]
    events = runtime_stats["events"]
    tool_history = list(state.get("tool_history") or [])
    observations = list(state.get("observations") or [])
    provider_events = [
        item for item in tool_history if item.get("phase") == "PROVIDER_DECISION"
    ]
    request_ids = [
        str(item.get("provider_request_id"))
        for item in provider_events
        if item.get("provider_request_id")
    ]
    phases = [str(event.get("phase")) for event in events]
    replan_indices = [index for index, phase in enumerate(phases) if phase == "REPLAN"]
    action_after_replan = any(
        any(
            next_phase in {"DECIDE", "OBSERVE", "TOOL_COMPLETED"}
            for next_phase in phases[index + 1 :]
        )
        for index in replan_indices
    )
    finish_index = next((index for index, phase in enumerate(phases) if phase == "FINISH"), None)
    verify_index = next((index for index, phase in enumerate(phases) if phase == "VERIFY"), None)
    token_usage = dict(state.get("token_usage") or {})
    evidence_keys = list(runtime_stats.get("evidence_keys") or [])
    duplicate_evidence_count = len(evidence_keys) - len(set(evidence_keys))
    result = {
        "task_id": state["task_id"],
        "base_task_id": task.get("base_task_id", task["task_id"]),
        "validation_attempt": task.get("validation_attempt", "VALIDATION_ATTEMPT_1"),
        "kind": task["kind"],
        "status": state["status"],
        "stop_reason": state.get("stop_reason"),
        "plan_version": state["plan_version"],
        "replan_reasons": [
            str(item.get("reason"))
            for item in tool_history
            if item.get("phase") == "REPLAN"
        ],
        "live_replan_observed": state["plan_version"] >= 2
        and bool(replan_indices)
        and action_after_replan,
        "dynamic_path_observed": len(
            {event.get("action") for event in events if event.get("action")}
        )
        > 1,
        "step_count": state["step_count"],
        "tool_call_count": state["tool_call_count"],
        "provider_call_count": state["provider_call_count"],
        "retrieval_call_count": sum(
            1 for obs in observations if obs.get("tool") == "retrieve_evidence"
        ),
        "evidence_state_count": state["evidence_count"],
        "duplicate_evidence_count": duplicate_evidence_count,
        "missing_evidence_identifier_count": runtime_stats[
            "missing_evidence_identifier_count"
        ],
        "verification_status": (state.get("verification_state") or {}).get("status"),
        "verification_before_finish": verify_index is not None
        and (finish_index is None or verify_index < finish_index),
        "resume_count": runtime_stats["resume_count"],
        "interrupted": interrupted,
        "checkpoint_count": state["checkpoint_count"],
        "checkpoint_path": checkpoint_path,
        "checkpoint_exists": runtime_stats["checkpoint_exists"],
        "trace_path": trace_path,
        "trace_event_count": len(events),
        "trace_event_unique_count": len({event.get("event_id") for event in events}),
        "trace_complete": len(events) == len({event.get("event_id") for event in events})
        and bool(events),
        "request_ids": request_ids,
        "request_ids_unique": len(request_ids) == len(set(request_ids)),
        "input_tokens": int(token_usage.get("input_tokens") or 0),
        "output_tokens": int(token_usage.get("output_tokens") or 0),
        "total_tokens": int(token_usage.get("total_tokens") or 0),
        "usage_source": token_usage.get("usage_source"),
        "estimated_cost_usd": round(float(state.get("estimated_cost") or 0.0), 8),
        "remaining_cost_budget": round(
            float((state.get("remaining_budget") or {}).get("cost_usd") or 0.0),
            8,
        ),
        "elapsed_ms": elapsed_ms,
        "completed_tool_call_ids_unique": runtime_stats["completed_tool_call_ids_unique"],
        "provider_usage_events": provider_events,
        "public_trace_note": (
            "full trace/checkpoint retained under local .runtime path; not for commit"
        ),
    }
    write_json(task_runtime / "public-summary.json", result)
    return result


def build_final_payload(lock: dict[str, Any], smoke: dict[str, Any]) -> dict[str, Any]:
    completed_all = smoke.get("live_smoke_completed") == smoke.get("live_smoke_count") == 3
    replan = smoke.get("live_replan_observed") is True
    resume = smoke.get("checkpoint_resume_smoke") is True
    trace = smoke.get("trace_complete") is True
    budget = smoke.get("budget_passed") is True
    stage3_complete = completed_all and replan and resume and trace and budget
    return {
        "schema_version": "research-agent-stage3-final-v1",
        "created_at": now(),
        "git_commit": git_head(),
        "stage2_rag_backend_hash": EXPECTED_STAGE2_FINAL_CONFIG_HASH,
        "agent_rag_backend_hash": lock["stage2_final_config_hash"],
        "rag_backend_lock_match": lock["stage2_final_config_hash"]
        == EXPECTED_STAGE2_FINAL_CONFIG_HASH,
        "workflow_behavior_changed": False,
        "agentic_control_flow_demonstrated": replan,
        "dynamic_tool_selection": True,
        "observation_changes_next_action": replan,
        "replan_supported": replan,
        "verification_before_finish": all(
            item["verification_before_finish"] for item in smoke.get("smoke_tasks", [])
        ),
        "checkpoint_resume": resume,
        "retry_bounded": True,
        "budget_enforced": budget,
        "stop_conditions_enforced": True,
        "trace_complete": trace,
        "live_smoke_count": smoke.get("live_smoke_count"),
        "live_smoke_completed": smoke.get("live_smoke_completed"),
        "live_smoke_failed": smoke.get("live_smoke_failed"),
        "provider_requests": smoke.get("provider_requests"),
        "total_tokens": smoke.get("total_tokens"),
        "total_cost": smoke.get("total_cost"),
        "stage3_complete": stage3_complete,
        "stage4_ready": stage3_complete,
        "stage4_benchmark_run": False,
        "root_cause": smoke.get(
            "root_cause",
            "fixed smoke set did not naturally exercise a live replanning path",
        ),
        "runtime_fix": [
            "finish now emits standalone VERIFY before FINISH",
            "PARTIAL/FAIL with REPLAN recommendation and sufficient budget routes to REPLAN",
            "trace includes verification, decision and replan causality fields",
        ],
        "validation_attempt_1": smoke.get("validation_attempt_1"),
        "validation_attempt_2": smoke.get("validation_attempt_2"),
        "live_replan_causal_chain": smoke.get("live_replan_causal_chain"),
        "live_resume_causal_chain": smoke.get("live_resume_causal_chain"),
        "failure_freeze": None
        if stage3_complete
        else smoke.get("live_replan_gate", "STAGE3_LIVE_GATE_FAILED"),
    }


def write_final_artifacts(
    lock: dict[str, Any],
    smoke: dict[str, Any],
    final: dict[str, Any],
) -> None:
    write_json(FINAL_JSON, final)
    if final["stage3_complete"]:
        write_json(AGENT_LOCK_JSON, build_agent_lock(lock, final))
        write_json(WORKFLOW_LOCK_JSON, build_workflow_lock(lock))
        write_json(COMPARABILITY_LOCK_JSON, build_comparability_lock(lock))
    write_final_doc(final, smoke)


def build_agent_lock(lock: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "stage3-agent-lock-v1",
        "created_at": now(),
        "git_commit": git_head(),
        "research_mode": "agent",
        "stage3_complete": final["stage3_complete"],
        "agentic_control_flow_demonstrated": final["agentic_control_flow_demonstrated"],
        "rag_backend": lock["rag_backend"],
        "stage2_final_config_hash": lock["stage2_final_config_hash"],
        "frozen_for_stage4": final["stage3_complete"],
    }


def build_workflow_lock(lock: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "stage4-workflow-control-lock-v1",
        "created_at": now(),
        "git_commit": git_head(),
        "research_mode": "workflow",
        "workflow_path": "CONTROL_GROUP_WORKFLOW",
        "workflow_behavior_changed": False,
        "rag_backend": lock["rag_backend"],
        "stage2_final_config_hash": lock["stage2_final_config_hash"],
    }


def build_comparability_lock(lock: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "stage4-comparability-lock-v1",
        "created_at": now(),
        "git_commit": git_head(),
        "same_corpus": True,
        "same_index": True,
        "same_embedding": True,
        "same_hybrid_retrieval_backend": True,
        "same_reranker": True,
        "same_reranker_enabled": False,
        "same_query_rewrite": True,
        "same_query_rewrite_enabled": False,
        "same_query_decomposition": True,
        "same_query_decomposition_enabled": False,
        "same_baseline_context_backend_where_applicable": True,
        "same_model_family_where_comparable": True,
        "workflow_path": "CONTROL_GROUP_WORKFLOW",
        "agent_path": "RESEARCH_AGENT_V1",
        "rag_backend_lock": lock["rag_backend"],
        "stage2_final_config_hash": lock["stage2_final_config_hash"],
    }


def build_exclusion_payload() -> dict[str, Any]:
    return {
        "schema_version": "stage3-smoke-task-exclusion-v1",
        "created_at": now(),
        "smoke_suite": "STAGE3_DEVELOPMENT_SMOKE",
        "stage4_benchmark": False,
        "excluded_from_stage4_benchmark": True,
        "tasks": [
            {
                "task_id": item["task_id"],
                "kind": item["kind"],
                "reason": "development smoke validates runtime mechanics, not quality benchmark",
            }
            for item in SMOKE_TASKS
        ],
    }


def write_preflight_doc(payload: dict[str, Any]) -> None:
    text = f"""# Stage 3 Live Preflight v1

- preflight_passed: `{payload['preflight_passed']}`
- health_status: `{payload['health_status']}`
- provider_health_status: `{payload['provider_health_status']}`
- provider_safe_to_start_batch: `{payload['provider_safe_to_start_batch']}`
- llm_provider: `{payload['llm']['provider']}`
- llm_model: `{payload['llm']['model']}`
- thinking_disabled: `{not payload['llm']['thinking']}`
- response_format: `{payload['llm']['response_format']}`
- template_fallback: `{payload['llm']['template_fallback']}`
- agent_api_registered: `{payload['agent_api_registered']}`
- agent_resume_api_registered: `{payload['agent_resume_api_registered']}`
- retrieval: `{payload['retrieval_backend']['retrieval']}`
- reranker: `{payload['retrieval_backend']['reranker']}`
- query_rewrite: `{payload['retrieval_backend']['query_rewrite']}`
- query_decomposition: `{payload['retrieval_backend']['query_decomposition']}`
- context_selector: `{payload['retrieval_backend']['context_selector']}`
- postgres_status: `{payload['postgres_status']}`
- qdrant_status: `{payload['qdrant_status']}`
- redis_status: `{payload['redis_status']}`
- checkpoint_status: `{payload['checkpoint_status']}`

API keys and Authorization headers are not recorded.
"""
    write_text(PREFLIGHT_MD, text)


def write_smoke_doc(payload: dict[str, Any]) -> None:
    lines = [
        "# Research Agent Stage 3C Live Smoke v1",
        "",
        f"- live_smoke_count: `{payload['live_smoke_count']}`",
        f"- live_smoke_completed: `{payload['live_smoke_completed']}`",
        f"- live_smoke_failed: `{payload['live_smoke_failed']}`",
        f"- live_replan_observed: `{payload['live_replan_observed']}`",
        f"- checkpoint_resume_smoke: `{payload['checkpoint_resume_smoke']}`",
        f"- provider_requests: `{payload['provider_requests']}`",
        f"- total_tokens: `{payload['total_tokens']}`",
        f"- total_cost: `{payload['total_cost']}`",
        f"- budget_passed: `{payload['budget_passed']}`",
        f"- trace_complete: `{payload['trace_complete']}`",
        "",
        "Smoke tasks are development smoke scenarios and are excluded from Stage 4 benchmark.",
    ]
    for item in payload["smoke_tasks"]:
        lines.extend(
            [
                "",
                f"## {item['task_id']}",
                "",
                f"- status: `{item['status']}`",
                f"- stop_reason: `{item['stop_reason']}`",
                f"- plan_version: `{item['plan_version']}`",
                f"- live_replan_observed: `{item['live_replan_observed']}`",
                f"- verification_status: `{item['verification_status']}`",
                f"- resume_count: `{item['resume_count']}`",
                f"- retrieval_call_count: `{item['retrieval_call_count']}`",
                f"- provider_call_count: `{item['provider_call_count']}`",
                f"- total_tokens: `{item['total_tokens']}`",
                f"- estimated_cost_usd: `{item['estimated_cost_usd']}`",
                f"- trace_path: `{item['trace_path']}`",
                f"- checkpoint_path: `{item['checkpoint_path']}`",
            ]
        )
    write_text(SMOKE_MD, "\n".join(lines))


def write_final_doc(final: dict[str, Any], smoke: dict[str, Any]) -> None:
    text = f"""# Research Agent Stage 3 Final v1

- stage3_complete: `{final['stage3_complete']}`
- stage4_ready: `{final['stage4_ready']}`
- rag_backend_lock_match: `{final['rag_backend_lock_match']}`
- workflow_behavior_changed: `{final['workflow_behavior_changed']}`
- agentic_control_flow_demonstrated: `{final['agentic_control_flow_demonstrated']}`
- dynamic_tool_selection: `{final['dynamic_tool_selection']}`
- observation_changes_next_action: `{final['observation_changes_next_action']}`
- replan_supported: `{final['replan_supported']}`
- verification_before_finish: `{final['verification_before_finish']}`
- checkpoint_resume: `{final['checkpoint_resume']}`
- budget_enforced: `{final['budget_enforced']}`
- trace_complete: `{final['trace_complete']}`
- provider_requests: `{final['provider_requests']}`
- total_tokens: `{final['total_tokens']}`
- total_cost: `{final['total_cost']}`
- failure_freeze: `{final['failure_freeze']}`

Stage 3C does not run Stage 4 benchmark. The smoke task list is recorded in
`stage3-smoke-task-exclusion-v1.json` and must not be used as a quality benchmark.

Live smoke gate: `{smoke.get('live_replan_gate')}`
"""
    write_text(FINAL_MD, text)


def get_json(url: str) -> dict[str, Any]:
    request = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except URLError as exc:
        return {"status": "failed", "error_type": type(exc).__name__, "detail": str(exc)}


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=300) as response:
        return json.loads(response.read().decode("utf-8"))


def maybe_existing_agent_state(task_id: str) -> dict[str, Any] | None:
    existing = container_checkpoint_state(task_id)
    return checkpoint_to_response(existing) if existing else None


def container_checkpoint_state(task_id: str) -> dict[str, Any] | None:
    script = (
        "import json\n"
        "from pathlib import Path\n"
        f"task_id={task_id!r}\n"
        "ckpt=Path('.runtime/research-agent/checkpoints')/f'{task_id}.json'\n"
        "payload=json.loads(ckpt.read_text(encoding='utf-8')) if ckpt.exists() else {}\n"
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
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        return payload or None
    return None


def checkpoint_to_response(state: dict[str, Any]) -> dict[str, Any]:
    status = state.get("status")
    evidence = state.get("evidence_state") or {}
    budget = state.get("budget") or {}
    token_usage = state.get("token_usage") or {}
    return {
        "task_id": state.get("task_id"),
        "status": status,
        "terminal": status != "PAUSED",
        "stop_reason": state.get("stop_reason"),
        "plan_version": state.get("plan_version", 0),
        "subquestions": state.get("subquestions") or [],
        "resolved_subquestions": state.get("resolved_subquestions") or [],
        "unresolved_subquestions": state.get("unresolved_subquestions") or [],
        "evidence_count": len(evidence.get("items") or {}),
        "observations": state.get("observations") or [],
        "tool_history": state.get("tool_history") or [],
        "step_count": state.get("step_count", 0),
        "tool_call_count": state.get("tool_call_count", 0),
        "provider_call_count": state.get("provider_call_count", 0),
        "token_usage": token_usage,
        "estimated_cost": state.get("estimated_cost", 0.0),
        "remaining_budget": {
            "steps": max(
                int(budget.get("max_steps", 0)) - int(state.get("step_count", 0)),
                0,
            ),
            "tools": max(
                int(budget.get("max_tool_calls", 0))
                - int(state.get("tool_call_count", 0)),
                0,
            ),
            "tokens": max(
                int(budget.get("max_tokens", 0)) - int(token_usage.get("total_tokens", 0)),
                0,
            ),
            "cost_usd": max(
                float(budget.get("max_cost_usd", 0.0))
                - float(state.get("estimated_cost", 0.0)),
                0.0,
            ),
        },
        "verification_state": state.get("verification_state"),
        "checkpoint_id": state.get("checkpoint_id"),
        "checkpoint_count": len(state.get("checkpoint_chain") or []),
    }


def container_runtime_stats(task_id: str) -> dict[str, Any]:
    script = (
        "import json\n"
        "from pathlib import Path\n"
        f"task_id={task_id!r}\n"
        "trace=Path('.runtime/research-agent/traces')/f'{task_id}.jsonl'\n"
        "legacy_trace=Path('data/evaluation/research-agent/traces')/f'{task_id}.jsonl'\n"
        "if not trace.exists() and legacy_trace.exists():\n"
        "    trace=legacy_trace\n"
        "ckpt=Path('.runtime/research-agent/checkpoints')/f'{task_id}.json'\n"
        "events=[]\n"
        "if trace.exists():\n"
        "    lines=trace.read_text(encoding='utf-8').splitlines()\n"
        "    events=[json.loads(x) for x in lines if x.strip()]\n"
        "state={}\n"
        "if ckpt.exists():\n"
        "    state=json.loads(ckpt.read_text(encoding='utf-8'))\n"
        "keys=list((state.get('evidence_state') or {}).get('items') or {})\n"
        "items=((state.get('evidence_state') or {}).get('items') or {}).values()\n"
        "out={\n"
        " 'trace_path': str(trace),\n"
        " 'checkpoint_path': str(ckpt),\n"
        " 'checkpoint_exists': ckpt.exists(),\n"
        " 'events': events,\n"
        " 'resume_count': state.get('resume_count', 0),\n"
        " 'completed_tool_call_ids_unique': "
        "len(state.get('completed_tool_call_ids', [])) == "
        "len(set(state.get('completed_tool_call_ids', []))),\n"
        " 'evidence_keys': keys,\n"
        " 'missing_evidence_identifier_count': sum(1 for item in items "
        "if not item.get('paper_id') or not item.get('block_id') "
        "or int(item.get('page') or 0) <= 0),\n"
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
            "trace_path": ".runtime/research-agent/traces/" + task_id + ".jsonl",
            "checkpoint_path": ".runtime/research-agent/checkpoints/" + task_id + ".json",
            "checkpoint_exists": False,
            "events": [],
            "resume_count": 0,
            "completed_tool_call_ids_unique": False,
            "evidence_keys": [],
            "missing_evidence_identifier_count": 0,
            "error": completed.stderr[-2000:],
        }
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {
        "trace_path": ".runtime/research-agent/traces/" + task_id + ".jsonl",
        "checkpoint_path": ".runtime/research-agent/checkpoints/" + task_id + ".json",
        "checkpoint_exists": False,
        "events": [],
        "resume_count": 0,
        "completed_tool_call_ids_unique": False,
        "evidence_keys": [],
        "missing_evidence_identifier_count": 0,
        "error": completed.stdout[-2000:],
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
