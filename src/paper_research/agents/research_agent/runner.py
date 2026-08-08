from __future__ import annotations

import time
from pathlib import Path

from paper_research.agents.research_agent.backend_lock import validate_rag_backend_lock
from paper_research.agents.research_agent.checkpoint import JsonResearchAgentCheckpointStore
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
        checkpoint_store: JsonResearchAgentCheckpointStore | None = None,
        trace_writer: ResearchAgentTraceWriter | None = None,
        lock_path: Path = Path("data/evaluation/research-agent/stage3-rag-backend-lock-v1.json"),
    ) -> None:
        self.backend_lock = validate_rag_backend_lock(lock_path)
        self.planner = RuleBasedResearchPlanner()
        self.policy = ResearchAgentPolicy()
        self.verifier = DeterministicResearchVerifier()
        self.tools = ResearchAgentToolRegistry(retrieval_provider)
        self.checkpoints = checkpoint_store or JsonResearchAgentCheckpointStore()
        self.trace = trace_writer or ResearchAgentTraceWriter()

    def run(
        self,
        research_question: str,
        *,
        task_id: str | None = None,
        budget: AgentBudget | None = None,
        interrupt_after_phase: str | None = None,
    ) -> AgentState:
        state = AgentState(research_question=research_question, budget=budget or AgentBudget())
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
                action = self.policy.decide(state)
                state.last_action = action
                if self._checkpoint_and_trace(state, "DECIDE", interrupt_after_phase):
                    return state
                if action.action == "finish":
                    self._finish(state)
                    break
                if action.action == "verify_evidence":
                    self._verify(state)
                    if self._checkpoint_and_trace(state, "VERIFY", interrupt_after_phase):
                        return state
                    if state.verification_state and state.verification_state.status == (
                        VerificationStatus.PASS
                    ):
                        continue
                    if state.remaining_step_budget <= 0 or state.remaining_tool_budget <= 0:
                        self._stop(state, StopReason.VERIFICATION_FAILED_NO_BUDGET)
                        break
                    self._replan(state, "verification_partial_or_failed")
                    if self._checkpoint_and_trace(state, "REPLAN", interrupt_after_phase):
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
        state.current_plan = plan
        state.plan_version = 1
        state.subquestions = plan.subquestions
        state.unresolved_subquestions = [item.id for item in plan.subquestions]
        for subquestion in state.subquestions:
            state.evidence_state.subquestion_evidence.setdefault(subquestion.id, [])

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
        state.refresh_remaining_budget()

    def _replan(self, state: AgentState, reason: str) -> None:
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
        state.tool_history.append(
            {"phase": "REPLAN", "plan_version": state.plan_version, "reason": reason}
        )

    def _finish(self, state: AgentState) -> None:
        if state.verification_state is None:
            self._verify(state)
        if state.verification_state and state.verification_state.status == VerificationStatus.PASS:
            state.status = AgentStatus.COMPLETED
            state.stop_reason = StopReason.SUCCESS
        else:
            state.status = AgentStatus.PARTIAL
            state.stop_reason = StopReason.VERIFICATION_FAILED_NO_BUDGET
        state.refresh_remaining_budget()
        self._checkpoint_and_trace(state, "FINISH", None)

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
    ) -> bool:
        state.refresh_remaining_budget()
        self.checkpoints.save(state, phase)
        self.trace.append(state, phase=phase)
        if interrupt_after_phase == phase:
            state.status = AgentStatus.PAUSED
            self.checkpoints.save(state, "PAUSED")
            self.trace.append(state, phase="PAUSED")
            return True
        return False

