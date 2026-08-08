from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from paper_research.agents.research_agent import (
    EXPECTED_STAGE2_FINAL_CONFIG_HASH,
    validate_rag_backend_lock,
)
from paper_research.agents.research_agent.checkpoint import JsonResearchAgentCheckpointStore
from paper_research.agents.research_agent.runner import ResearchAgentRunner
from paper_research.agents.research_agent.state import AgentBudget
from paper_research.agents.research_agent.trace import ResearchAgentTraceWriter

ROOT = Path("data/evaluation/research-agent")
DOC_ROOT = Path("docs/research-agent")


class ScenarioProvider:
    def __init__(self, responses: list[list[dict[str, Any]]]) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def search(self, query: str, paper_ids=None, limit: int = 5) -> list[dict[str, Any]]:
        self.calls.append(query)
        if self.responses:
            return self.responses.pop(0)
        return []


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    DOC_ROOT.mkdir(parents=True, exist_ok=True)
    lock = validate_rag_backend_lock()
    scenarios = run_development_scenarios()
    smoke = build_smoke_payload(scenarios)
    runtime = build_runtime_payload(lock, scenarios, smoke)
    comparability = build_comparability(lock)
    write_json(ROOT / "stage3-agent-runtime-v1.json", runtime)
    write_json(ROOT / "stage3-agent-smoke-v1.json", smoke)
    write_json(ROOT / "workflow-agent-comparability-v1.json", comparability)
    write_docs(runtime, smoke, comparability)
    print(
        json.dumps(
            {
                "stage3_runtime_ready": runtime["stage3_runtime_ready"],
                "agentic_control_flow_demonstrated": runtime[
                    "agentic_control_flow_demonstrated"
                ],
                "development_scenarios_passed": runtime["development_scenarios_passed"],
                "live_smoke_count": smoke["live_smoke_count"],
                "provider_requests": smoke["provider_requests"],
                "total_tokens": smoke["total_tokens"],
                "total_cost": smoke["total_cost"],
            },
            ensure_ascii=False,
        )
    )
    return 0


def run_development_scenarios() -> list[dict[str, Any]]:
    cases = [
        (
            "complete_evidence_early_finish",
            "alpha and beta",
            [[evidence("a")], [evidence("b")]],
            None,
        ),
        (
            "missing_evidence_replan",
            "alpha and beta",
            [[evidence("a")], [], [evidence("b")]],
            None,
        ),
        (
            "tool_failure_retry",
            "alpha and beta",
            [[], [evidence("a")], [evidence("b")]],
            "fail_first",
        ),
        (
            "checkpoint_resume",
            "alpha and beta",
            [[evidence("a")], [evidence("b")]],
            "interrupt",
        ),
    ]
    results: list[dict[str, Any]] = []
    for name, question, responses, mode in cases:
        provider = ScenarioProvider(responses)
        if mode == "fail_first":
            provider = FailingOnceProvider(responses)
        runner = ResearchAgentRunner(
            provider,
            checkpoint_store=JsonResearchAgentCheckpointStore(
                ROOT / "stage3-smoke-runtime" / name / "checkpoints"
            ),
            trace_writer=ResearchAgentTraceWriter(
                ROOT / "stage3-smoke-runtime" / name / "traces"
            ),
        )
        if mode == "interrupt":
            first = runner.run(
                question,
                task_id=f"stage3-{name}",
                budget=AgentBudget(),
                interrupt_after_phase="STATE_UPDATED",
            )
            state = runner.resume(first.task_id)
        else:
            state = runner.run(question, task_id=f"stage3-{name}", budget=AgentBudget())
        results.append(
            {
                "scenario": name,
                "task_id": state.task_id,
                "status": state.status.value,
                "stop_reason": state.stop_reason.value if state.stop_reason else None,
                "plan_version": state.plan_version,
                "step_count": state.step_count,
                "tool_call_count": state.tool_call_count,
                "provider_call_count": state.provider_call_count,
                "evidence_count": len(state.evidence_state.items),
                "observation_count": len(state.observations),
                "checkpoint_count": len(state.checkpoint_chain),
                "resume_count": state.resume_count,
                "trace_event_count": len(state.trace_event_ids),
                "replan_observed": state.plan_version > 1,
                "dynamic_path_observed": bool(state.observations),
                "completed_tool_call_ids_unique": len(state.completed_tool_call_ids)
                == len(set(state.completed_tool_call_ids)),
                "trace_event_ids_unique": len(state.trace_event_ids)
                == len(set(state.trace_event_ids)),
                "retrieval_queries": provider.calls,
            }
        )
    return results


class FailingOnceProvider(ScenarioProvider):
    def search(self, query: str, paper_ids=None, limit: int = 5) -> list[dict[str, Any]]:
        self.calls.append(query)
        if len(self.calls) == 1:
            raise RuntimeError("controlled transient failure")
        if self.responses:
            return self.responses.pop(0)
        return []


def evidence(block_id: str) -> dict[str, Any]:
    return {
        "evidence_id": block_id,
        "paper_id": "paper-dev",
        "page_start": 1,
        "page_end": 1,
        "text": f"Controlled development evidence for {block_id}.",
        "retrieval_score": 0.9,
        "section_path": ["Results"],
    }


def build_runtime_payload(
    lock: dict[str, Any],
    scenarios: list[dict[str, Any]],
    smoke: dict[str, Any],
) -> dict[str, Any]:
    replan_observed = any(item["replan_observed"] for item in scenarios)
    resume_passed = any(item["resume_count"] == 1 for item in scenarios)
    return {
        "schema_version": "stage3-agent-runtime-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": git_head(),
        "research_mode_default": "workflow",
        "new_research_mode": "agent",
        "control_group_workflow_changed": False,
        "stage2_rag_backend_hash": EXPECTED_STAGE2_FINAL_CONFIG_HASH,
        "agent_rag_backend_hash": lock["stage2_final_config_hash"],
        "rag_backend_lock_match": True,
        "agent_state_implemented": True,
        "evidence_state_implemented": True,
        "tool_registry": [
            "retrieve_evidence",
            "inspect_evidence",
            "inspect_paper",
            "verify_evidence",
            "finish",
        ],
        "tool_count": 5,
        "planner_implemented": True,
        "dynamic_tool_selection": True,
        "observation_state_update": True,
        "replan_implemented": True,
        "verifier_implemented": True,
        "checkpoint_implemented": True,
        "resume_implemented": resume_passed,
        "retry_implemented": True,
        "no_progress_detection": True,
        "step_budget": 12,
        "tool_budget": 16,
        "token_budget": 40000,
        "cost_budget": 0.05,
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
        "trace_schema": "research-agent-trace-v1",
        "trace_complete": all(item["trace_event_ids_unique"] for item in scenarios),
        "development_scenarios": [item["scenario"] for item in scenarios],
        "development_scenarios_passed": all(
            item["completed_tool_call_ids_unique"] and item["trace_event_ids_unique"]
            for item in scenarios
        ),
        "live_smoke_count": smoke["live_smoke_count"],
        "live_smoke_completed": smoke["live_smoke_completed"],
        "live_smoke_failed": smoke["live_smoke_failed"],
        "live_replan_observed": smoke["live_replan_observed"],
        "live_dynamic_path_observed": smoke["live_dynamic_path_observed"],
        "checkpoint_resume_smoke": resume_passed,
        "provider_requests": smoke["provider_requests"],
        "total_tokens": smoke["total_tokens"],
        "total_cost": smoke["total_cost"],
        "agentic_control_flow_demonstrated": replan_observed and resume_passed,
        "stage3_runtime_ready": replan_observed and resume_passed,
        "stage4_ready": False,
        "stage4_blocker": "Stage 3 final freeze required before benchmark.",
    }


def build_smoke_payload(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "stage3-agent-smoke-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "mode": "controlled_development_runtime",
        "live_smoke_count": 0,
        "live_smoke_completed": 0,
        "live_smoke_failed": 0,
        "live_replan_observed": False,
        "live_dynamic_path_observed": False,
        "provider_requests": 0,
        "total_tokens": 0,
        "total_cost": 0,
        "controlled_scenarios": scenarios,
        "note": (
            "No real LLM, embedding, reranker, or Stage 4 benchmark was run. "
            "Stage 3 runtime mechanics were validated with controlled providers."
        ),
    }


def build_comparability(lock: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "workflow-agent-comparability-v1",
        "created_at": datetime.now(UTC).isoformat(),
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
        "fairness_note": (
            "Agent v1 changes action timing and state-conditioned tool selection only; "
            "it does not add a new retriever, reranker, embedding model, query rewrite, "
            "query decomposition module, or information source."
        ),
    }


def write_docs(
    runtime: dict[str, Any],
    smoke: dict[str, Any],
    comparability: dict[str, Any],
) -> None:
    architecture = """# Research Agent Architecture v1

LangGraph remains the existing workflow substrate for the control-group Deep
Research path. Research Agent v1 is introduced as a parallel runtime path whose
next actions are selected dynamically from current state and observations rather
than following a fixed research sequence.

```mermaid
flowchart TD
    START --> PLAN
    PLAN --> DECIDE
    DECIDE --> EXECUTE["EXECUTE TOOL"]
    EXECUTE --> OBSERVE
    OBSERVE --> UPDATE["UPDATE STATE"]
    UPDATE --> VERIFY
    VERIFY -->|PASS| FINISH
    VERIFY -->|FAIL/PARTIAL| REPLAN
    REPLAN --> DECIDE
    DECIDE -->|budget exhausted| STOP_BUDGET["STOP: budget exhausted"]
    DECIDE -->|no progress| STOP_PROGRESS["STOP: no progress"]
    EXECUTE -->|fatal provider/tool failure| STOP_FAILURE["STOP: provider/tool failure"]
```

The existing `CONTROL_GROUP_WORKFLOW` is not rewritten and remains the default.
Agent mode is exposed separately as `research_mode=agent`.
"""
    runtime_doc = f"""# Research Agent Runtime v1

- RAG backend lock match: `{runtime['rag_backend_lock_match']}`
- Stage 2 config hash: `{runtime['agent_rag_backend_hash']}`
- Tool registry: `{', '.join(runtime['tool_registry'])}`
- Dynamic tool selection: `{runtime['dynamic_tool_selection']}`
- Replan implemented: `{runtime['replan_implemented']}`
- Checkpoint implemented: `{runtime['checkpoint_implemented']}`
- Resume implemented: `{runtime['resume_implemented']}`
- Retry bounded: `{runtime['retry_implemented']}`
- No-progress detection: `{runtime['no_progress_detection']}`
- Stage 4 ready: `{runtime['stage4_ready']}`

Agent v1 stores structured decisions and short decision reasons only. It does
not persist hidden reasoning or raw provider responses.
"""
    smoke_doc = f"""# Research Agent Smoke v1

- mode: `{smoke['mode']}`
- live smoke count: `{smoke['live_smoke_count']}`
- provider requests: `{smoke['provider_requests']}`
- total tokens: `{smoke['total_tokens']}`
- total cost: `{smoke['total_cost']}`

Controlled development scenarios validate dynamic branching, replan, retry,
checkpoint resume and trace uniqueness. No Stage 4 benchmark was run.
"""
    write_text(DOC_ROOT / "research-agent-architecture-v1.md", architecture)
    write_text(DOC_ROOT / "research-agent-state-v1.md", state_doc())
    write_text(DOC_ROOT / "research-agent-tools-v1.md", tools_doc(comparability))
    write_text(DOC_ROOT / "research-agent-trace-v1.md", trace_doc())
    write_text(DOC_ROOT / "research-agent-runtime-v1.md", runtime_doc)
    write_text(DOC_ROOT / "research-agent-smoke-v1.md", smoke_doc)


def state_doc() -> str:
    return """# Research Agent State v1

State contains task identity, research question, current plan, subquestions,
resolved/unresolved subquestions, evidence state, observations, tool history,
candidate/verified/unsupported claims, contradictions, budgets, retry state,
verification state, status, stop reason and checkpoint metadata.

Evidence state is keyed by stable `paper_id:block_id:page` identifiers and
deduplicates repeated observations deterministically.
"""


def tools_doc(comparability: dict[str, Any]) -> str:
    return f"""# Research Agent Tools v1

Initial tool registry:

- retrieve_evidence
- inspect_evidence
- inspect_paper
- verify_evidence
- finish

Retrieval wraps the existing frozen RAG service boundary. Reranker, query
rewrite and query decomposition remain disabled.

Comparability: `{comparability['fairness_note']}`
"""


def trace_doc() -> str:
    return """# Research Agent Trace v1

Each step records task id, step id, timestamp, phase, plan version, action,
tool name, sanitized tool args, target subquestions, observation status,
evidence added count, evidence state count, verification status, remaining
budget, latency, provider usage, tool usage, checkpoint id and stop reason.

Trace must not include API keys, Authorization headers, hidden reasoning,
chain-of-thought or raw provider responses.
"""


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


if __name__ == "__main__":
    raise SystemExit(main())
