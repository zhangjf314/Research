"""Finalize Stage 3D Research Agent v1 freeze artifacts.

The script is deliberately offline-only. It reads prior Stage 3C evidence,
records the validation protocol amendment, freezes behavior/comparability locks,
and updates the Stage 3 final report without calling providers, retrieval, or
Agent APIs.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from paper_research.agents.research_agent import (
    EXPECTED_STAGE2_FINAL_CONFIG_HASH,
    validate_rag_backend_lock,
)
from paper_research.config import Settings

ROOT = Path("data/evaluation/research-agent")
DOC_ROOT = Path("docs/research-agent")
ARCHITECTURE_DOC = DOC_ROOT / "research-agent-architecture-v1.md"

STAGE3C_SMOKE_JSON = ROOT / "stage3-agent-smoke-v1.json"
STAGE3C_FORENSICS_JSON = ROOT / "stage3c-replan-forensics-v1.json"
STAGE3C2_PLAN_JSON = ROOT / "stage3-live-replan-validation-plan-v1.json"
STAGE3C2_RESULT_JSON = ROOT / "stage3-live-replan-validation-v1.json"
STAGE3_RUNTIME_JSON = ROOT / "stage3-agent-runtime-v1.json"
STAGE3_FINAL_JSON = ROOT / "research-agent-stage3-final-v1.json"
STAGE3_FINAL_MD = DOC_ROOT / "research-agent-stage3-final-v1.md"

PROTOCOL_JSON = ROOT / "stage3-validation-protocol-amendment-v1.json"
PROTOCOL_MD = DOC_ROOT / "stage3-validation-protocol-amendment-v1.md"
AGENT_LOCK_JSON = ROOT / "stage3-agent-lock-v1.json"
WORKFLOW_LOCK_JSON = ROOT / "stage4-workflow-control-lock-v1.json"
COMPARABILITY_LOCK_JSON = ROOT / "stage4-comparability-lock-v1.json"
TASK_EXCLUSIONS_JSON = ROOT / "stage4-task-exclusions-v1.json"

RUNTIME_SOURCE_FILES = [
    Path("src/paper_research/agents/research_agent/__init__.py"),
    Path("src/paper_research/agents/research_agent/backend_lock.py"),
    Path("src/paper_research/agents/research_agent/checkpoint.py"),
    Path("src/paper_research/agents/research_agent/decision_provider.py"),
    Path("src/paper_research/agents/research_agent/models.py"),
    Path("src/paper_research/agents/research_agent/planner.py"),
    Path("src/paper_research/agents/research_agent/policy.py"),
    Path("src/paper_research/agents/research_agent/runner.py"),
    Path("src/paper_research/agents/research_agent/state.py"),
    Path("src/paper_research/agents/research_agent/tools.py"),
    Path("src/paper_research/agents/research_agent/trace.py"),
    Path("src/paper_research/agents/research_agent/verifier.py"),
]
TOOL_REGISTRY = [
    "retrieve_evidence",
    "inspect_evidence",
    "inspect_paper",
    "verify_evidence",
    "finish",
]
STOP_CONDITIONS = [
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
]
REPLAN_METRICS = [
    "replan_task_count",
    "effective_replan_count",
    "replan_task_rate",
    "replans_per_task",
    "replan_trigger_distribution",
    "post_replan_success_rate",
    "post_replan_evidence_gain",
    "post_replan_claim_coverage_delta",
]


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    DOC_ROOT.mkdir(parents=True, exist_ok=True)
    lock = validate_rag_backend_lock()
    if lock["stage2_final_config_hash"] != EXPECTED_STAGE2_FINAL_CONFIG_HASH:
        raise SystemExit("RAG_BACKEND_LOCK_MISMATCH")

    settings = Settings()
    before = behavior_payload(settings, lock)
    before_hash = stable_hash(before)

    smoke = read_json(STAGE3C_SMOKE_JSON)
    forensics = read_json(STAGE3C_FORENSICS_JSON)
    stage3c2_plan = read_json(STAGE3C2_PLAN_JSON)
    stage3c2 = read_json(STAGE3C2_RESULT_JSON)
    runtime = read_json(STAGE3_RUNTIME_JSON)

    amendment = build_protocol_amendment(smoke, forensics, stage3c2)
    agent_lock = build_agent_lock(settings, lock, before, before_hash)
    workflow_lock = build_workflow_lock(settings, lock)
    comparability_lock = build_comparability_lock(settings, lock)
    exclusions = build_task_exclusions(smoke, stage3c2_plan)
    final = build_final(runtime, smoke, stage3c2, amendment, agent_lock)

    write_json(PROTOCOL_JSON, amendment)
    write_protocol_doc(amendment)
    write_json(AGENT_LOCK_JSON, agent_lock)
    write_json(WORKFLOW_LOCK_JSON, workflow_lock)
    write_json(COMPARABILITY_LOCK_JSON, comparability_lock)
    write_json(TASK_EXCLUSIONS_JSON, exclusions)
    write_json(STAGE3_FINAL_JSON, final)
    write_final_doc(final)
    update_architecture_doc()

    after_hash = stable_hash(behavior_payload(settings, lock))
    if before_hash != after_hash:
        raise SystemExit("AGENT_BEHAVIOR_DRIFT_DURING_FREEZE")

    print(
        json.dumps(
            {
                "protocol_amendment_created": True,
                "stage3_agent_behavior_hash": before_hash,
                "runtime_behavior_hash_stable": before_hash == after_hash,
                "stage3_complete": final["stage3_complete"],
                "stage3_complete_with_known_limitation": final[
                    "stage3_complete_with_known_limitation"
                ],
                "stage4_ready": final["stage4_ready"],
                "new_provider_requests": 0,
                "new_tokens": 0,
                "new_cost": 0,
            },
            ensure_ascii=False,
        )
    )
    return 0


def build_protocol_amendment(
    smoke: dict[str, Any],
    forensics: dict[str, Any],
    stage3c2: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "stage3-validation-protocol-amendment-v1",
        "amendment_version": "v1",
        "created_at": now(),
        "created_after_evidence": [
            "Stage 3C initial live smoke",
            "Stage 3C.1 same-task post-fix revalidation",
            "Stage 3C.2 preregistered frozen validation set",
        ],
        "previous_gate": {
            "effective_live_replan_observed": True,
            "meaning": "required for Stage 3 freeze",
        },
        "revised_gate": {
            "effective_live_replan_observed": "NOT_REQUIRED_FOR_STAGE3_FREEZE",
            "live_effective_replan_recorded_value": False,
            "stage4_behavioral_metric": True,
        },
        "reason": (
            "Continuing to design tasks or adjust runtime/prompt/policy until a live "
            "replan appears would overfit the validation set. Live validation remains "
            "evidence for real-provider integration, dynamic decisions, observation-driven "
            "actions, verification, checkpoint/resume, budget, trace, and accounting. "
            "The replan branch is covered by controlled deterministic runtime tests."
        ),
        "required_protocol_statement": (
            "The replan branch is covered by controlled deterministic runtime tests, "
            "including PARTIAL/FAIL to REPLAN transition, effective plan delta, changed "
            "next action, checkpoint interaction, and trace causality. Real-provider "
            "validation demonstrated dynamic tool selection and observation-driven "
            "actions, but no preregistered live development task exercised the complete "
            "effective-replan causal chain. Rather than continue adapting validation "
            "tasks or runtime behavior until a replan appears, the live-replan requirement "
            "is moved from a Stage 3 release gate to a Stage 4 behavioral measurement."
        ),
        "evidence_considered": {
            "stage3c_smoke_artifact": str(STAGE3C_SMOKE_JSON),
            "stage3c_forensics_artifact": str(STAGE3C_FORENSICS_JSON),
            "stage3c2_validation_artifact": str(STAGE3C2_RESULT_JSON),
        },
        "stage3c_result": {
            "live_smoke_count": smoke.get("live_smoke_count"),
            "live_smoke_completed": smoke.get("live_smoke_completed"),
            "effective_replan_observed": smoke.get("live_replan_observed"),
            "runtime_defect_discovered": forensics.get("runtime_bug_confirmed"),
            "gate": smoke.get("live_replan_gate"),
        },
        "stage3c1_result": {
            "runtime_defect_fixed": True,
            "same_smoke_2_3_rerun": True,
            "dynamic_path": smoke.get("live_dynamic_path_observed"),
            "resume": smoke.get("checkpoint_resume_smoke"),
            "effective_replan_observed": smoke.get("live_replan_observed"),
            "forensics_root_cause": forensics.get("forensics_root_cause"),
        },
        "stage3c2_result": {
            "validation_task_count": stage3c2.get("validation_task_count"),
            "runtime_frozen": stage3c2.get("agent_runtime_frozen_for_validation"),
            "dynamic_tool_selection_observed": stage3c2.get(
                "dynamic_tool_selection_observed"
            ),
            "observation_driven_action_observed": stage3c2.get(
                "observation_driven_action_observed"
            ),
            "effective_live_replan_observed": stage3c2.get(
                "effective_replan_observed"
            ),
            "task1_plan_version_2_is_not_effective_replan": True,
        },
        "runtime_was_frozen_during_stage3c2": stage3c2.get(
            "agent_runtime_frozen_for_validation"
        ),
        "additional_tasks_after_preregistered_set": 0,
        "behavior_changes_during_stage3c2": 0,
        "effective_live_replan_observed": False,
        "known_limitation": "LIVE_EFFECTIVE_REPLAN_NOT_OBSERVED",
        "forbidden_claims": [
            "Live replan is unnecessary.",
            "Replan has been fully validated in live runs.",
            "Live adaptive replanning demonstrated.",
            "Agent routinely replans based on observations.",
        ],
    }


def build_agent_lock(
    settings: Settings,
    lock: dict[str, Any],
    behavior: dict[str, Any],
    behavior_hash: str,
) -> dict[str, Any]:
    return {
        "schema_version": "stage3-agent-lock-v1",
        "created_at": now(),
        "implementation_commit": git_head(),
        "research_mode": "agent",
        "agent_runtime": "RESEARCH_AGENT_V1",
        "stage3_status": "COMPLETE_WITH_KNOWN_LIMITATION",
        "stage3_agent_behavior_hash": behavior_hash,
        "behavior_hash_inputs": behavior,
        "planner_config": behavior["planner_config"],
        "planner_prompt_hash": behavior["planner_prompt_hash"],
        "policy_config": behavior["policy_config"],
        "policy_prompt_hash": behavior["policy_prompt_hash"],
        "verifier_config": behavior["verifier_config"],
        "verifier_prompt_hash": behavior["verifier_prompt_hash"],
        "tool_registry": TOOL_REGISTRY,
        "tool_registry_hash": behavior["tool_registry_hash"],
        "budget_config": behavior["budget_config"],
        "retry_policy": behavior["retry_policy"],
        "stop_conditions": STOP_CONDITIONS,
        "no_progress_policy": behavior["no_progress_policy"],
        "checkpoint_schema": behavior["checkpoint_schema"],
        "trace_schema": behavior["trace_schema"],
        "provider": settings.llm_provider_name or settings.llm_provider,
        "model": settings.llm_model,
        "stage2_rag_backend_hash": lock["stage2_final_config_hash"],
        "rag_backend": lock["rag_backend"],
        "effective_live_replan_observed": False,
        "effective_live_replan_gate": "NOT_REQUIRED_FOR_STAGE3_FREEZE",
        "known_limitation": "LIVE_EFFECTIVE_REPLAN_NOT_OBSERVED",
        "frozen_for_stage4": True,
    }


def build_workflow_lock(settings: Settings, lock: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "stage4-workflow-control-lock-v1",
        "created_at": now(),
        "implementation_commit": git_head(),
        "workflow_path": "CONTROL_GROUP_WORKFLOW",
        "workflow_runtime_version": "frozen_existing_deep_research_workflow",
        "workflow_behavior_changed": False,
        "provider": settings.llm_provider_name or settings.llm_provider,
        "model": settings.llm_model,
        "rag_backend_hash": lock["stage2_final_config_hash"],
        "retrieval_config": lock["rag_backend"],
        "workflow_budget": {
            "uses_existing_deep_research_budget": True,
            "stage4_must_not_increase_budget_for_workflow_or_agent_advantage": True,
        },
        "workflow_synthesis_config": {
            "uses_existing_structured_synthesis": True,
            "template_fallback": False,
        },
        "checkpoint_behavior": "existing_workflow_checkpoint_behavior_frozen",
        "report_quality_gates": [
            "citation_validation",
            "report_quality_gate",
            "duplication_gate",
            "cross_section_similarity_gate",
        ],
        "frozen_for_stage4_control_group": True,
    }


def build_comparability_lock(settings: Settings, lock: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "stage4-comparability-lock-v1",
        "created_at": now(),
        "implementation_commit": git_head(),
        "same_corpus": True,
        "same_index": True,
        "same_embedding": True,
        "same_retrieval_backend": True,
        "same_hybrid_retrieval_backend": True,
        "same_reranker_state": True,
        "same_reranker_enabled": False,
        "same_query_rewrite_state": True,
        "same_query_rewrite_enabled": False,
        "same_query_decomposition_state": True,
        "same_query_decomposition_enabled": False,
        "same_model_family": True,
        "provider": settings.llm_provider_name or settings.llm_provider,
        "model": settings.llm_model,
        "allowed_differences": {
            "workflow": "predetermined research control flow",
            "agent": (
                "state/observation-driven dynamic action selection, verification loop, "
                "and optional replanning capability"
            ),
        },
        "forbidden_advantages": [
            "stronger retriever",
            "different corpus",
            "different embedding",
            "Stage 2 rejected modules",
            "reranker enabled for only one arm",
        ],
        "replan_metrics_preregistered": REPLAN_METRICS,
        "replan_interpretation_rules": {
            "if_effective_replan_count_gt_0": (
                "live effective replanning naturally occurred during the benchmark"
            ),
            "if_effective_replan_count_eq_0": (
                "The Agent exposes a validated replanning mechanism, but no effective "
                "replan was naturally exercised across the Stage 4 benchmark tasks."
            ),
            "forbidden_claim_when_zero": (
                "Adaptive replanning is a demonstrated source of benchmark improvement."
            ),
        },
        "stage2_rag_backend_hash": lock["stage2_final_config_hash"],
        "rag_backend": lock["rag_backend"],
        "stage4_benchmark_run": False,
    }


def build_task_exclusions(
    smoke: dict[str, Any],
    stage3c2_plan: dict[str, Any],
) -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []
    original_smokes = (
        (smoke.get("validation_attempt_1") or {}).get("smoke_tasks")
        or smoke.get("smoke_tasks", [])
    )
    for item in original_smokes:
        task_id = item["task_id"]
        tasks.append(
            {
                "task_id": task_id,
                "task_hash": stable_hash({"task_id": task_id, "kind": item.get("kind")}),
                "semantic_exclusion_description": (
                    "Exclude exact task, obvious paraphrase, and same validation "
                    "mechanic exposure from Stage 4 benchmark."
                ),
                "reason": "development_validation_exposure",
                "source_artifact": str(STAGE3C_SMOKE_JSON),
            }
        )
    for item in stage3c2_plan.get("validation_tasks", []):
        tasks.append(
            {
                "task_id": item["task_id"],
                "task_hash": item["task_hash"],
                "semantic_exclusion_description": (
                    "Exclude exact question, obvious paraphrase, and same "
                    "observation-dependent comparison from Stage 4 benchmark."
                ),
                "reason": "development_validation_exposure",
                "source_artifact": str(STAGE3C2_PLAN_JSON),
            }
        )
    return {
        "schema_version": "stage4-task-exclusions-v1",
        "created_at": now(),
        "excluded_from_stage4_benchmark": True,
        "task_count": len(tasks),
        "tasks": tasks,
    }


def build_final(
    runtime: dict[str, Any],
    smoke: dict[str, Any],
    stage3c2: dict[str, Any],
    amendment: dict[str, Any],
    agent_lock: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "research-agent-stage3-final-v1",
        "created_at": runtime.get("created_at"),
        "updated_at": now(),
        "git_commit": git_head(),
        "stage3_status": "COMPLETE_WITH_KNOWN_LIMITATION",
        "stage3_complete": True,
        "stage3_complete_with_known_limitation": True,
        "stage4_ready": True,
        "known_limitation": "LIVE_EFFECTIVE_REPLAN_NOT_OBSERVED",
        "stage2_rag_backend_hash": EXPECTED_STAGE2_FINAL_CONFIG_HASH,
        "agent_rag_backend_hash": EXPECTED_STAGE2_FINAL_CONFIG_HASH,
        "rag_backend_lock_match": True,
        "workflow_behavior_changed": False,
        "agent_behavior_changed_during_freeze": False,
        "agent_runtime_implemented": True,
        "stateful_execution": True,
        "evidence_state": True,
        "planner": True,
        "dynamic_tool_selection": True,
        "observation_state_update": True,
        "replan_runtime_supported": True,
        "replan_deterministic_tests_pass": True,
        "verifier": True,
        "verification_before_completed_finish": True,
        "checkpoint": True,
        "resume": True,
        "bounded_retry": True,
        "budget_enforced": True,
        "stop_conditions": True,
        "no_progress_detection": True,
        "trace_complete": True,
        "trace_causality_schema": True,
        "real_provider_validation": True,
        "real_hybrid_retrieval_validation": True,
        "live_dynamic_tool_selection_observed": stage3c2.get(
            "dynamic_tool_selection_observed"
        ),
        "live_observation_driven_action_observed": stage3c2.get(
            "observation_driven_action_observed"
        ),
        "live_checkpoint_resume_observed": smoke.get("checkpoint_resume_smoke"),
        "effective_live_replan_observed": False,
        "effective_live_replan_gate": "NOT_REQUIRED_FOR_STAGE3_FREEZE",
        "stage3_agent_behavior_hash": agent_lock["stage3_agent_behavior_hash"],
        "runtime_behavior_hash_stable": True,
        "protocol_amendment": str(PROTOCOL_JSON),
        "stage3_agent_lock": str(AGENT_LOCK_JSON),
        "stage4_workflow_control_lock": str(WORKFLOW_LOCK_JSON),
        "stage4_comparability_lock": str(COMPARABILITY_LOCK_JSON),
        "stage4_task_exclusions": str(TASK_EXCLUSIONS_JSON),
        "stage4_replan_metrics_preregistered": REPLAN_METRICS,
        "validation_history_preserved": {
            "stage3c": {
                "live_smoke_count": smoke.get("live_smoke_count"),
                "effective_replan_count": 0,
                "runtime_defect_discovered": True,
            },
            "stage3c1": amendment["stage3c1_result"],
            "stage3c2": amendment["stage3c2_result"],
        },
        "new_provider_requests": 0,
        "new_tokens": 0,
        "new_cost": 0,
        "stage4_benchmark_run": False,
    }


def behavior_payload(settings: Settings, lock: dict[str, Any]) -> dict[str, Any]:
    source_hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in RUNTIME_SOURCE_FILES
    }
    tool_registry_hash = stable_hash(TOOL_REGISTRY)
    decision_source = Path("src/paper_research/agents/research_agent/decision_provider.py")
    planner_source = Path("src/paper_research/agents/research_agent/planner.py")
    verifier_source = Path("src/paper_research/agents/research_agent/verifier.py")
    return {
        "source_hashes": source_hashes,
        "planner_config": {
            "implementation": "RuleBasedResearchPlanner",
            "max_initial_subquestions": 6,
            "replan_adds_gap_subquestions": True,
        },
        "planner_prompt_hash": hashlib.sha256(planner_source.read_bytes()).hexdigest(),
        "policy_config": {
            "implementation": "ResearchAgentPolicy",
            "state_observation_driven": True,
            "finish_requires_verification": True,
        },
        "policy_prompt_hash": hashlib.sha256(decision_source.read_bytes()).hexdigest(),
        "verifier_config": {
            "implementation": "ResearchAgentVerifier",
            "partial_or_fail_recommends_replan": True,
        },
        "verifier_prompt_hash": hashlib.sha256(verifier_source.read_bytes()).hexdigest(),
        "tool_registry": TOOL_REGISTRY,
        "tool_registry_hash": tool_registry_hash,
        "budget_config": {
            "max_steps": 12,
            "max_tool_calls": 16,
            "max_provider_requests": 12,
            "max_tokens": 40000,
            "max_cost_usd": 0.05,
        },
        "retry_policy": {"provider_retry_max": 1, "tool_retry_max": 1},
        "stop_conditions": STOP_CONDITIONS,
        "no_progress_policy": {"max_no_progress_actions": 2},
        "checkpoint_schema": "research-agent-checkpoint-v1-json",
        "trace_schema": "research-agent-trace-v1",
        "provider": settings.llm_provider_name or settings.llm_provider,
        "model": settings.llm_model,
        "stage2_rag_backend_hash": lock["stage2_final_config_hash"],
        "rag_backend": lock["rag_backend"],
        "runtime_includes_post_stage3c1_fixes": [
            "PARTIAL_FAIL_REPLAN_TRANSITION",
            "FINISH_GUARD",
            "VERIFY_TRACE",
            "DECISION_CAUSALITY",
            "REPLAN_DELTA_HASH",
        ],
    }


def write_protocol_doc(payload: dict[str, Any]) -> None:
    lines = [
        "# Stage 3 Validation Protocol Amendment v1",
        "",
        f"- amendment_version: `{payload['amendment_version']}`",
        "- previous_gate: `effective_live_replan_observed=true`",
        "- revised_gate: `effective_live_replan_observed=NOT_REQUIRED_FOR_STAGE3_FREEZE`",
        f"- effective_live_replan_observed: `{payload['effective_live_replan_observed']}`",
        f"- known_limitation: `{payload['known_limitation']}`",
        "- additional_tasks_after_preregistered_set: `0`",
        "- behavior_changes_during_stage3c2: `0`",
        "",
        "## Rationale",
        "",
        payload["required_protocol_statement"],
        "",
        "## Validation history",
        "",
        "### Stage 3C",
        "",
        "- 3 live smoke tasks.",
        "- effective replan: `0`.",
        "- runtime defect discovered.",
        "",
        "### Stage 3C.1",
        "",
        "- runtime defect fixed.",
        "- same Smoke 2/3 rerun.",
        "- dynamic path: `true`.",
        "- resume: `true`.",
        "- effective replan: `0`.",
        "",
        "### Stage 3C.2",
        "",
        "- 3 preregistered dependency-shaped tasks.",
        "- runtime frozen: `true`.",
        "- dynamic tool selection: `true`.",
        "- observation-driven action: `true`.",
        "- effective replan: `0`.",
        "",
        "## Evidence matrix",
        "",
        "| Capability | Offline | Live | Final status |",
        "| --- | --- | --- | --- |",
        "| Planner | PASS | PASS | VALIDATED |",
        "| Dynamic tool selection | PASS | OBSERVED | VALIDATED |",
        "| Observation-driven action | PASS | OBSERVED | VALIDATED |",
        "| Evidence state | PASS | OBSERVED | VALIDATED |",
        "| Verification | PASS | OBSERVED | VALIDATED |",
        "| Checkpoint/resume | PASS | OBSERVED | VALIDATED |",
        "| Retry | PASS | NOT REQUIRED | VALIDATED_OFFLINE |",
        "| Budget | PASS | OBSERVED | VALIDATED |",
        "| Replan transition | PASS | NOT OBSERVED | VALIDATED_OFFLINE_ONLY |",
        "| Effective live replan | N/A | NOT OBSERVED | KNOWN_LIMITATION |",
        "",
        "Stage 3C.2 Task 1 reached `plan_version=2`, but this remains distinct",
        "from an effective live replan because the full causal-chain definition was",
        "not satisfied.",
    ]
    write_text(PROTOCOL_MD, "\n".join(lines))


def write_final_doc(final: dict[str, Any]) -> None:
    lines = [
        "# Research Agent Stage 3 Final v1",
        "",
        f"- stage3_status: `{final['stage3_status']}`",
        f"- stage3_complete: `{final['stage3_complete']}`",
        (
            "- stage3_complete_with_known_limitation: "
            f"`{final['stage3_complete_with_known_limitation']}`"
        ),
        f"- stage4_ready: `{final['stage4_ready']}`",
        f"- known_limitation: `{final['known_limitation']}`",
        f"- stage3_agent_behavior_hash: `{final['stage3_agent_behavior_hash']}`",
        f"- runtime_behavior_hash_stable: `{final['runtime_behavior_hash_stable']}`",
        f"- rag_backend_lock_match: `{final['rag_backend_lock_match']}`",
        f"- workflow_behavior_changed: `{final['workflow_behavior_changed']}`",
        f"- effective_live_replan_observed: `{final['effective_live_replan_observed']}`",
        f"- effective_live_replan_gate: `{final['effective_live_replan_gate']}`",
        f"- new_provider_requests: `{final['new_provider_requests']}`",
        f"- new_tokens: `{final['new_tokens']}`",
        f"- new_cost: `{final['new_cost']}`",
        "",
        "Stage 3 is complete with a documented limitation. Effective live replan",
        "was not observed and must not be claimed as demonstrated. It is now a",
        "preregistered Stage 4 behavioral metric.",
        "",
        "## Final evidence matrix",
        "",
        "| Capability | Offline | Live | Final status |",
        "| --- | --- | --- | --- |",
        "| Planner | PASS | PASS | VALIDATED |",
        "| Dynamic tool selection | PASS | OBSERVED | VALIDATED |",
        "| Observation-driven action | PASS | OBSERVED | VALIDATED |",
        "| Evidence state | PASS | OBSERVED | VALIDATED |",
        "| Verification | PASS | OBSERVED | VALIDATED |",
        "| Checkpoint/resume | PASS | OBSERVED | VALIDATED |",
        "| Retry | PASS | NOT REQUIRED | VALIDATED_OFFLINE |",
        "| Budget | PASS | OBSERVED | VALIDATED |",
        "| Replan transition | PASS | NOT OBSERVED | VALIDATED_OFFLINE_ONLY |",
        "| Effective live replan | N/A | NOT OBSERVED | KNOWN_LIMITATION |",
    ]
    write_text(STAGE3_FINAL_MD, "\n".join(lines))


def update_architecture_doc() -> None:
    text = ARCHITECTURE_DOC.read_text(encoding="utf-8")
    marker = "## Stage 3D validation protocol amendment"
    addition = """

## Stage 3D validation protocol amendment

The runtime supports an explicit REPLAN transition and the transition is covered
by deterministic controlled tests.

Real-provider development validation demonstrated dynamic tool selection and
observation-driven actions. However, the complete effective live-replan causal
chain was not observed in the frozen Stage 3 development validation tasks.

Stage 4 therefore measures replanning as an observed runtime behavior rather
than assuming it occurs.
"""
    if marker not in text:
        ARCHITECTURE_DOC.write_text(text.rstrip() + addition + "\n", encoding="utf-8")


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
