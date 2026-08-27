from __future__ import annotations

import time
from hashlib import sha256
from pathlib import Path
from typing import Any

from paper_research.agents.research_agent.backend_lock import validate_rag_backend_lock
from paper_research.agents.research_agent.checkpoint import JsonResearchAgentCheckpointStore
from paper_research.agents.research_agent.decision_provider import (
    AgentDecisionProviderError,
    LLMResearchAgentDecisionProvider,
    provider_usage_delta,
)
from paper_research.agents.research_agent.models import (
    AgentStatus,
    ObservationStatus,
    StopReason,
    SubquestionStatus,
    VerificationStatus,
)
from paper_research.agents.research_agent.planner import RuleBasedResearchPlanner
from paper_research.agents.research_agent.policy import ResearchAgentPolicy
from paper_research.agents.research_agent.state import AgentBudget, AgentState
from paper_research.agents.research_agent.tools import (
    EvidenceSearchProvider,
    ResearchAgentToolRegistry,
)
from paper_research.agents.research_agent.trace import ResearchAgentTraceWriter
from paper_research.agents.research_agent.verifier import DeterministicResearchVerifier


class ResearchAgentRunner:
    def __init__(
        self,
        retrieval_provider: EvidenceSearchProvider | None = None,
        *,
        decision_provider: LLMResearchAgentDecisionProvider | None = None,
        checkpoint_store: JsonResearchAgentCheckpointStore | None = None,
        trace_writer: ResearchAgentTraceWriter | None = None,
        lock_path: Path = Path("data/evaluation/research-agent/stage3-rag-backend-lock-v1.json"),
    ) -> None:
        self.backend_lock = validate_rag_backend_lock(lock_path)
        self.planner = RuleBasedResearchPlanner()
        self.policy = ResearchAgentPolicy()
        self.verifier = DeterministicResearchVerifier()
        self.tools = ResearchAgentToolRegistry(retrieval_provider)
        self.decision_provider = decision_provider
        self.checkpoints = checkpoint_store or JsonResearchAgentCheckpointStore()
        self.trace = trace_writer or ResearchAgentTraceWriter()

    def run(
        self,
        research_question: str,
        *,
        task_id: str | None = None,
        paper_ids: list[str] | None = None,
        budget: AgentBudget | None = None,
        interrupt_after_phase: str | None = None,
    ) -> AgentState:
        state = AgentState(
            research_question=research_question,
            paper_ids=list(paper_ids) if paper_ids else None,
            budget=budget or AgentBudget(),
        )
        if task_id:
            state.task_id = task_id
        state.refresh_remaining_budget()
        return self._drive(state, interrupt_after_phase=interrupt_after_phase)

    def resume(self, task_id: str, *, interrupt_after_phase: str | None = None) -> AgentState:
        state = self.checkpoints.load(task_id)
        state.resume_count += 1
        if state.status not in {AgentStatus.PAUSED, AgentStatus.RUNNING}:
            return state
        state.status = AgentStatus.RUNNING
        state.refresh_remaining_budget()
        return self._drive(state, interrupt_after_phase=interrupt_after_phase)

    def _drive(
        self,
        state: AgentState,
        *,
        interrupt_after_phase: str | None = None,
    ) -> AgentState:
        try:
            self._plan_if_needed(state)
            if self._checkpoint_and_trace(state, "PLAN", interrupt_after_phase):
                return state
            while state.status == AgentStatus.RUNNING:
                stop = self._pre_budget_stop(state)
                if stop:
                    self._stop(state, stop)
                    break
                deterministic_action = self.policy.decide(state)
                action = self._provider_decide(state, deterministic_action)
                state.last_action = action
                if self._checkpoint_and_trace(state, "DECIDE", interrupt_after_phase):
                    return state
                if action.action == "finish":
                    if self._finish(state, interrupt_after_phase):
                        break
                    continue
                if action.action == "verify_evidence":
                    self._verify(state)
                    if self._checkpoint_and_trace(
                        state,
                        "VERIFY",
                        interrupt_after_phase,
                        extra=self._verification_trace_extra(state),
                    ):
                        return state
                    if state.verification_state and state.verification_state.status == (
                        VerificationStatus.PASS
                    ):
                        continue
                    if not self._can_replan(state):
                        self._stop(state, StopReason.VERIFICATION_FAILED_NO_BUDGET)
                        break
                    self._replan(state, "verification_partial_or_failed")
                    if self._checkpoint_and_trace(
                        state,
                        "REPLAN",
                        interrupt_after_phase,
                        extra=self._replan_trace_extra(state),
                    ):
                        return state
                    continue
                self._execute_tool(state)
                if self._checkpoint_and_trace(state, "TOOL_COMPLETED", interrupt_after_phase):
                    return state
                self._update_state(state)
                if self._checkpoint_and_trace(state, "STATE_UPDATED", interrupt_after_phase):
                    return state
                if state.observations and state.observations[-1].status == ObservationStatus.FAILED:
                    if state.observations[-1].retryable and state.retry_state.tool_retry_count <= 1:
                        continue
                    self._stop(state, StopReason.TOOL_FAILURE)
                    break
        except Exception:
            state.status = AgentStatus.FAILED
            state.stop_reason = StopReason.CHECKPOINT_FAILURE
            raise
        finally:
            state.refresh_remaining_budget()
            self.checkpoints.save(state, "FINAL")
            self.trace.append(state, phase="FINAL")
        return state

    def _plan_if_needed(self, state: AgentState) -> None:
        if state.current_plan is not None:
            return
        plan = self.planner.initial_plan(state.research_question)
        if self.decision_provider is not None:
            try:
                plan, usage = self.decision_provider.plan(
                    state.research_question,
                    plan,
                    state.task_id,
                )
                self._record_provider_usage(state, usage)
            except AgentDecisionProviderError as exc:
                if exc.usage_result is not None:
                    self._record_provider_usage(state, exc.usage_result)
                raise
        state.current_plan = plan
        state.plan_version = 1
        state.subquestions = plan.subquestions
        state.unresolved_subquestions = [item.id for item in plan.subquestions]
        for subquestion in state.subquestions:
            state.evidence_state.subquestion_evidence.setdefault(subquestion.id, [])

    def _provider_decide(
        self,
        state: AgentState,
        deterministic_action,
    ):
        if self.decision_provider is None:
            return deterministic_action
        try:
            action, usage = self.decision_provider.decide(state, deterministic_action)
            self._record_provider_usage(state, usage)
            return action
        except AgentDecisionProviderError as exc:
            if exc.usage_result is not None:
                self._record_provider_usage(state, exc.usage_result)
            raise

    def _record_provider_usage(self, state: AgentState, result) -> None:
        usage = provider_usage_delta(result)
        state.provider_call_count += max(usage["request_attempt_count"], 1)
        state.token_usage.input_tokens += int(usage["input_tokens"])
        state.token_usage.output_tokens += int(usage["output_tokens"])
        state.token_usage.total_tokens += int(usage["total_tokens"])
        state.token_usage.usage_source = usage["usage_source"]
        state.estimated_cost += float(usage["estimated_cost_usd"])
        state.tool_history.append({"phase": "PROVIDER_DECISION", **usage})
        state.refresh_remaining_budget()
        self.checkpoints.save(state, "POST_USAGE_ACCOUNTING")
        self.trace.append(state, phase="POST_USAGE_ACCOUNTING", extra={"provider_decision": usage})

    def _execute_tool(self, state: AgentState) -> None:
        state.step_count += 1
        state.tool_call_count += 1
        state.refresh_remaining_budget()
        tool_call_id = f"{state.task_id}-tool-{state.tool_call_count:04d}"
        if tool_call_id in state.completed_tool_call_ids:
            return
        started = time.perf_counter()
        observation = self.tools.execute(state, state.last_action, tool_call_id)
        latency_ms = (time.perf_counter() - started) * 1000
        if observation.status == ObservationStatus.FAILED:
            state.retry_state.last_error = observation.error
            state.retry_state.retryable = observation.retryable
            if observation.retryable:
                state.retry_state.tool_retry_count += 1
        else:
            state.retry_state.retryable = False
            state.retry_state.last_error = None
        state.observations.append(observation)
        state.completed_tool_call_ids.append(tool_call_id)
        self.trace.append(state, phase="OBSERVE", latency_ms=latency_ms)

    def _update_state(self, state: AgentState) -> None:
        self.policy.observe_progress(state)
        resolved: list[str] = []
        unresolved: list[str] = []
        for subquestion in state.subquestions:
            if state.evidence_state.evidence_for(subquestion.id):
                subquestion.status = SubquestionStatus.RESOLVED
                resolved.append(subquestion.id)
            else:
                subquestion.status = SubquestionStatus.OPEN
                unresolved.append(subquestion.id)
        state.resolved_subquestions = resolved
        state.unresolved_subquestions = unresolved
        state.evidence_gaps = [
            f"missing evidence for {subquestion_id}" for subquestion_id in unresolved
        ]
        state.contradictions = list(state.evidence_state.contradictions)
        state.refresh_remaining_budget()

    def _verify(self, state: AgentState) -> None:
        state.step_count += 1
        result = self.verifier.verify(state)
        state.verification_state = result
        state.verified_claims = result.verified_claims
        state.unsupported_claims = result.unsupported_claims
        state.contradictions = result.contradictions
        state.unresolved_subquestions = result.unresolved_subquestions
        state.evidence_gaps = result.evidence_gaps
        state.refresh_remaining_budget()

    def _replan(self, state: AgentState, reason: str) -> None:
        old_plan = state.current_plan
        old_plan_version = state.plan_version
        old_hash = _plan_hash(old_plan.model_dump() if old_plan else {})
        plan = self.planner.replan(state, reason)
        state.current_plan = plan
        state.plan_version += 1
        state.subquestions = plan.subquestions
        known = {item.id for item in state.subquestions}
        state.resolved_subquestions = [
            item for item in state.resolved_subquestions if item in known
        ]
        state.unresolved_subquestions = [
            item.id
            for item in state.subquestions
            if item.id not in state.resolved_subquestions
        ]
        delta = _plan_delta(old_plan, plan)
        state.tool_history.append(
            {
                "phase": "REPLAN",
                "old_plan_version": old_plan_version,
                "new_plan_version": state.plan_version,
                "old_plan_hash": old_hash,
                "new_plan_hash": _plan_hash(plan.model_dump()),
                "reason": reason,
                "trigger": "VERIFICATION_PARTIAL_OR_FAILED",
                "effective_replan": delta["effective_plan_delta"],
                "plan_delta": delta,
            }
        )

    def _finish(self, state: AgentState, interrupt_after_phase: str | None) -> bool:
        if state.verification_state is None:
            self._verify(state)
            if self._checkpoint_and_trace(
                state,
                "VERIFY",
                interrupt_after_phase,
                extra=self._verification_trace_extra(state),
            ):
                return True
        if state.verification_state and state.verification_state.status == VerificationStatus.PASS:
            state.status = AgentStatus.COMPLETED
            state.stop_reason = StopReason.SUCCESS
            state.refresh_remaining_budget()
            self._checkpoint_and_trace(state, "FINISH", None)
            return True
        if self._can_replan(state):
            self._replan(state, "finish_requested_with_partial_verification")
            self._checkpoint_and_trace(
                state,
                "REPLAN",
                interrupt_after_phase,
                extra=self._replan_trace_extra(state),
            )
            return False
        if state.verification_state and state.verification_state.status in {
            VerificationStatus.PARTIAL,
            VerificationStatus.FAIL,
        }:
            state.status = AgentStatus.PARTIAL
            state.stop_reason = StopReason.VERIFICATION_FAILED_NO_BUDGET
        else:
            state.status = AgentStatus.PARTIAL
            state.stop_reason = StopReason.VERIFICATION_FAILED_NO_BUDGET
        state.refresh_remaining_budget()
        self._checkpoint_and_trace(state, "FINISH", None)
        return True

    def _can_replan(self, state: AgentState) -> bool:
        if state.verification_state is None:
            return False
        if state.verification_state.status not in {
            VerificationStatus.PARTIAL,
            VerificationStatus.FAIL,
        }:
            return False
        if state.verification_state.recommended_next_action != "REPLAN":
            return False
        state.refresh_remaining_budget()
        return (
            state.remaining_step_budget > 1
            and state.remaining_tool_budget > 0
            and state.remaining_token_budget > 0
            and state.remaining_cost_budget > 0
        )

    def _verification_trace_extra(self, state: AgentState) -> dict[str, Any]:
        verification = state.verification_state
        return {
            "verification_id": f"{state.task_id}-verification-{state.step_count:04d}",
            "based_on_observation_id": state.observations[-1].tool_call_id
            if state.observations
            else None,
            "evidence_state_version": len(state.evidence_state.items),
            "verification": verification.model_dump() if verification else None,
            "unresolved_count": len(verification.unresolved_subquestions)
            if verification
            else 0,
            "gap_count": len(verification.evidence_gaps) if verification else 0,
            "recommended_next_action": verification.recommended_next_action
            if verification
            else None,
        }

    def _replan_trace_extra(self, state: AgentState) -> dict[str, Any]:
        replan = next(
            (item for item in reversed(state.tool_history) if item.get("phase") == "REPLAN"),
            {},
        )
        return {
            "replan_id": f"{state.task_id}-replan-{state.plan_version:04d}",
            "trigger_verification_id": f"{state.task_id}-verification-{state.step_count:04d}",
            "replan": replan,
        }

    def _stop(self, state: AgentState, reason: StopReason) -> None:
        state.status = AgentStatus.PARTIAL if reason in {
            StopReason.NO_PROGRESS,
            StopReason.MAX_STEPS_REACHED,
            StopReason.TOOL_BUDGET_EXHAUSTED,
            StopReason.VERIFICATION_FAILED_NO_BUDGET,
        } else AgentStatus.FAILED
        state.stop_reason = reason
        state.refresh_remaining_budget()

    def _pre_budget_stop(self, state: AgentState) -> StopReason | None:
        state.refresh_remaining_budget()
        if state.remaining_step_budget <= 0:
            return StopReason.MAX_STEPS_REACHED
        if state.remaining_tool_budget <= 0:
            return StopReason.TOOL_BUDGET_EXHAUSTED
        if state.remaining_token_budget <= 0:
            return StopReason.TOKEN_BUDGET_EXHAUSTED
        if state.remaining_cost_budget <= 0:
            return StopReason.COST_BUDGET_EXHAUSTED
        if state.no_progress_count >= state.budget.max_no_progress_actions:
            return StopReason.NO_PROGRESS
        return None

    def _checkpoint_and_trace(
        self,
        state: AgentState,
        phase: str,
        interrupt_after_phase: str | None,
        extra: dict[str, Any] | None = None,
    ) -> bool:
        state.refresh_remaining_budget()
        self.checkpoints.save(state, phase)
        trace_extra = extra or {}
        if phase == "DECIDE":
            trace_extra = {
                **trace_extra,
                "decision_id": (
                    f"{state.task_id}-decision-{len(state.trace_event_ids) + 1:04d}"
                ),
                "trigger_source": _decision_trigger_source(state),
                "trigger_id": state.checkpoint_id,
                "selected_action": state.last_action.action if state.last_action else None,
            }
        self.trace.append(state, phase=phase, extra=trace_extra)
        if interrupt_after_phase == phase:
            state.status = AgentStatus.PAUSED
            self.checkpoints.save(state, "PAUSED")
            self.trace.append(state, phase="PAUSED")
            return True
        return False


def _decision_trigger_source(state: AgentState) -> str:
    if state.tool_history and state.tool_history[-1].get("phase") == "REPLAN":
        return "REPLAN"
    if state.verification_state is not None:
        return "VERIFICATION"
    if state.observations:
        return "OBSERVATION"
    return "INITIAL_PLAN"


def _plan_hash(payload: dict[str, Any]) -> str:
    import json

    return sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _plan_delta(old_plan, new_plan) -> dict[str, Any]:
    old_items = {item.id: item for item in old_plan.subquestions} if old_plan else {}
    new_items = {item.id: item for item in new_plan.subquestions}
    added = [item.model_dump() for key, item in new_items.items() if key not in old_items]
    removed = [item.model_dump() for key, item in old_items.items() if key not in new_items]
    reprioritized = [
        {
            "id": key,
            "old_priority": old_items[key].priority,
            "new_priority": new_items[key].priority,
        }
        for key in old_items.keys() & new_items.keys()
        if old_items[key].priority != new_items[key].priority
    ]
    changed_objective = bool(old_plan and old_plan.objective != new_plan.objective)
    return {
        "added_subquestions": added,
        "removed_subquestions": removed,
        "reprioritized_subquestions": reprioritized,
        "changed_research_objective_decomposition": changed_objective,
        "changed_next_action": bool(added or removed or reprioritized or changed_objective),
        "effective_plan_delta": bool(added or removed or reprioritized or changed_objective),
    }
