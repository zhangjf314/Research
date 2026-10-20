from __future__ import annotations

import sys
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

from paper_research.agents.research_agent.backend_lock import validate_rag_backend_lock
from paper_research.agents.research_agent.checkpoint import (
    JsonResearchAgentCheckpointStore, require_current_resume,
)
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
    ToolAction,
    ToolObservation,
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
from paper_research.agents.research_agent.verifier import (
    DeterministicResearchVerifier, REPORT_SAFETY_NAMESPACE, REPORT_SAFETY_RECEIPT,
    ResearchSafetyError, validate_research_task_id,
)
from paper_research.indexing.registry import IndexIdentityError


class ResearchAgentRunner:
    def __init__(
        self,
        retrieval_provider: EvidenceSearchProvider | None = None,
        *,
        decision_provider: LLMResearchAgentDecisionProvider | None = None,
        checkpoint_store: JsonResearchAgentCheckpointStore | None = None,
        trace_writer: ResearchAgentTraceWriter | None = None,
        lock_path: Path = Path("data/evaluation/research-agent/stage3-rag-backend-lock-v1.json"),
        frozen_backend=None,
        agentic_config=None,
        contract_planner=None,
        contract_verifier=None,
        structured_provider=None,
        readonly: bool = False,
    ) -> None:
        self.readonly = readonly
        if readonly:
            if frozen_backend is not None or (
                agentic_config is not None and agentic_config.agentic_retrieval_enabled
            ):
                raise ValueError("RESEARCH_BACKEND_OUT_OF_SCOPE")
            self.retrieval_provider = retrieval_provider
            self.checkpoints = checkpoint_store or JsonResearchAgentCheckpointStore()
            self._trace = trace_writer
            return
        from paper_research.agents.research_agent.evidence_contract import AgenticRetrievalConfig
        from paper_research.retrieval.frozen_backend import FrozenRetrievalBackend

        self.agentic_config = agentic_config or AgenticRetrievalConfig()
        self.frozen_backend = frozen_backend
        if frozen_backend is not None:
            if not isinstance(frozen_backend, FrozenRetrievalBackend):
                raise ValueError("Phase 2 requires the explicit FrozenRetrievalBackend loader")
            frozen_backend.guard()
        if self.agentic_config.agentic_retrieval_enabled and frozen_backend is None:
            raise ValueError("FROZEN_BACKEND_IDENTITY_MISMATCH: missing experimental backend")
        self.backend_lock = (
            frozen_backend.manifest if frozen_backend else validate_rag_backend_lock(lock_path)
        )
        self.planner = RuleBasedResearchPlanner()
        self.policy = ResearchAgentPolicy()
        self.verifier = DeterministicResearchVerifier()
        self.tools = ResearchAgentToolRegistry(frozen_backend or retrieval_provider)
        self.decision_provider = decision_provider
        self.checkpoints = checkpoint_store or JsonResearchAgentCheckpointStore()
        self._trace = trace_writer
        self.agentic_controller = None
        if self.agentic_config.agentic_retrieval_enabled:
            from paper_research.agents.research_agent.agentic_retrieval import (
                EvidenceDrivenRetrievalController,
            )
            from paper_research.agents.research_agent.contract_verifier import (
                StructuredContractAdapter,
            )

            adapter = (
                StructuredContractAdapter(structured_provider) if structured_provider else None
            )
            self.agentic_controller = EvidenceDrivenRetrievalController(
                self, contract_planner or self.planner, contract_verifier or adapter
            )

    @property
    def trace(self) -> ResearchAgentTraceWriter:
        if self._trace is None:
            self._trace = ResearchAgentTraceWriter()
        return self._trace

    def _prepare_new_task(self, task_id: str) -> None:
        from paper_research.agents.research_agent.final_report import AgentFinalReportStore
        validate_research_task_id(task_id)
        trace_root = self._trace.root if self._trace is not None else Path(".runtime/research-agent/traces")
        reports = AgentFinalReportStore().root
        if any(path.exists() for path in (
            self.checkpoints.path_for(task_id), trace_root / f"{task_id}.jsonl",
            reports / f"{task_id}-final-report.json", reports / f"{task_id}-final-report.md",
        )):
            raise ResearchSafetyError("RESEARCH_TASK_ALREADY_EXISTS", 409)
        self._guard_current_identity()

    def _guard_current_identity(self) -> None:
        provider = getattr(self, "retrieval_provider", None)
        if provider is None:
            provider = getattr(getattr(self, "tools", None), "retrieval_provider", None)
        if provider is not None and getattr(provider, "require_identity", False):
            provider._guard_identity()

    def run_readonly_retrieval(
        self, research_question: str, *, task_id: str | None = None,
        paper_ids: list[str] | None = None, budget: AgentBudget | None = None,
    ) -> AgentState:
        if not self.readonly:
            raise ValueError("readonly runner required")
        state = AgentState(
            research_question=research_question,
            paper_ids=list(paper_ids) if paper_ids else None,
            budget=budget or AgentBudget(),
        )
        if task_id:
            state.task_id = task_id
        self._prepare_new_task(state.task_id)
        observation = self.retrieval_provider.retrieve_readonly_observation(
            research_question, state.paper_ids, 5,
        )
        state.evidence_provenance["p1_readonly_retrieval_bridge_v1"] = {
            "mode": "READONLY_RETRIEVAL_OBSERVATION", "schema_version": 1,
            "retrieval_observation": observation.model_dump(mode="json"),
        }
        state.status = AgentStatus.PARTIAL
        state.step_count = state.tool_call_count = 1
        state.last_action = ToolAction(
            action="retrieve_evidence", arguments={"query": research_question, "top_k": 5},
            decision_reason="READONLY_RETRIEVAL_OBSERVATION",
        )
        tool_call_id = f"{state.task_id}-tool-0001"
        state.observations.append(ToolObservation(
            tool_call_id=tool_call_id, tool="retrieve_evidence", status=ObservationStatus.SUCCESS,
        ))
        state.completed_tool_call_ids.append(tool_call_id)
        state.tool_history.append({
            "phase": "READONLY_RETRIEVAL", "retrieval_trace_id": observation.trace.trace_id,
        })
        state.refresh_remaining_budget()
        self.checkpoints.save(state, "FINAL_READONLY")
        self.trace.append(state, phase="FINAL_READONLY")
        return state

    def run(
        self,
        research_question: str,
        *,
        task_id: str | None = None,
        paper_ids: list[str] | None = None,
        budget: AgentBudget | None = None,
        interrupt_after_phase: str | None = None,
        multimodal_enabled: bool = False,
    ) -> AgentState:
        state = AgentState(
            multimodal_enabled=multimodal_enabled,
            research_question=research_question,
            paper_ids=list(paper_ids) if paper_ids else None,
            budget=budget or AgentBudget(),
            agentic_config=self.agentic_config if self.agentic_controller else None,
            retrieval_backend_id=self.frozen_backend.backend_id if self.frozen_backend else None,
        )
        if task_id:
            state.task_id = task_id
        self._prepare_new_task(state.task_id)
        if self.agentic_controller or self.frozen_backend:
            raise ResearchSafetyError("RESEARCH_BACKEND_OUT_OF_SCOPE", 409)
        provider = getattr(self.tools, "retrieval_provider", None)
        if provider is not None and hasattr(provider, "active_binding"):
            binding, digest = provider.active_binding(state.paper_ids)
            # The active D6 branch is canonical retrieval, under the same C1
            # candidate-only policy. It does not grant semantic qualification.
            if REPORT_SAFETY_RECEIPT["mode"] != "CANDIDATE_ONLY_REPORT_SUPPRESSED":
                raise ResearchSafetyError("RESEARCH_TRUST_SCHEMA_INVALID")
            try:
                from paper_research.generation.evidence_contract_adapter import RetrievalIdentity
                identity = {key: binding["index_identity"][key]
                            for key in RetrievalIdentity.model_fields if key != "collection"}
                identity["collection"] = binding["collection_identity"]["collection_name"]
                scope = list(binding["source_scope"]["paper_ids"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ResearchSafetyError("CANONICAL_PROVENANCE_INVALID") from exc
            state.evidence_provenance["p1_active_canonical_provenance_v2"] = {
                "mode": "CANONICAL_" + REPORT_SAFETY_RECEIPT["mode"],
                "schema_version": 2,
                "source_identity_schema": "canonical-agent-provenance-v2",
                "policy_id": REPORT_SAFETY_RECEIPT["policy_id"],
                "semantic_qualification_id": REPORT_SAFETY_RECEIPT["semantic_qualification_id"],
                "report_eligibility": REPORT_SAFETY_RECEIPT["report_eligibility"],
                "retrieval_identity": identity,
                "paper_scope": scope,
                "generation_binding": binding,
                "generation_binding_digest": digest,
                "semantic_status": REPORT_SAFETY_RECEIPT["semantic_status"],
            }
        else:
            state.evidence_provenance[REPORT_SAFETY_NAMESPACE] = dict(REPORT_SAFETY_RECEIPT)
        state.refresh_remaining_budget()
        return self._drive(state, interrupt_after_phase=interrupt_after_phase)

    def resume(self, task_id: str, *, interrupt_after_phase: str | None = None) -> AgentState:
        state = self.checkpoints.load(task_id)
        require_current_resume(state)
        self._guard_current_identity()
        if "p1_active_canonical_provenance_v2" in state.evidence_provenance:
            provider = getattr(self.tools, "retrieval_provider", None)
            if provider is None or not hasattr(provider, "validate_active_snapshot"):
                raise ResearchSafetyError("CANONICAL_PROVENANCE_INVALID")
            provider.validate_active_snapshot(state)
        state.resume_count += 1
        state.status = AgentStatus.RUNNING
        state.refresh_remaining_budget()
        return self._drive(state, interrupt_after_phase=interrupt_after_phase)

    def _drive(
        self,
        state: AgentState,
        *,
        interrupt_after_phase: str | None = None,
    ) -> AgentState:
        if state.agentic_config and state.agentic_config.agentic_retrieval_enabled:
            if self.agentic_controller is None:
                raise ValueError("POLICY_FAILURE: cannot resume agentic checkpoint in static mode")
            return self.agentic_controller.drive(state, interrupt_after_phase)
        try:
            if state.verification_state is not None:
                self._restricted_finish(state)
                return state
            # A paused TOOL_COMPLETED checkpoint has already performed retrieval.
            # Apply its pending state update once, without repeating the tool or its costs.
            phases = [value.rsplit("-", 1)[-1] for value in state.checkpoint_chain]
            last_phase = next((value for value in reversed(phases)
                               if value not in {"FINAL", "PAUSED"}), None)
            if last_phase == "TOOL_COMPLETED":
                self._update_state(state)
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
                    self._restricted_finish(state)
                    break
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
        except IndexIdentityError:
            state.status = AgentStatus.FAILED
            state.stop_reason = StopReason.RETRIEVAL_FAILURE
            raise
        except Exception:
            state.status = AgentStatus.FAILED
            state.stop_reason = StopReason.CHECKPOINT_FAILURE
            raise
        finally:
            root_failure = sys.exc_info()[0] is not None
            state.refresh_remaining_budget()
            if root_failure:
                # A secondary checkpoint/trace failure must not replace a typed
                # P0, D6, or C2 rejection from the actual tool operation.
                # A rejected C2 operation must also preserve the last committed
                # state: retrying a failed replacement here would overwrite it.
                if "p1_active_canonical_provenance_v2" not in state.evidence_provenance:
                    try:
                        self.checkpoints.save(state, "FINAL")
                        self.trace.append(state, phase="FINAL")
                    except Exception:
                        pass
            else:
                self.checkpoints.save(state, "FINAL")
                self.trace.append(state, phase="FINAL")
        return state

    def _plan_if_needed(self, state: AgentState) -> None:
        if state.current_plan is not None:
            return
        if "p1_active_canonical_provenance_v2" in state.evidence_provenance:
            plan = self.planner.initial_plan(state.research_question, paper_ids=state.paper_ids)
        else:
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
        if "p1_active_canonical_provenance_v2" in state.evidence_provenance:
            identifiers = [item.id for item in plan.subquestions]
            if len(identifiers) != len(set(identifiers)):
                raise ResearchSafetyError("ACTIVE_SUBQUESTION_INVALID")
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
        if "p1_active_canonical_provenance_v2" in state.evidence_provenance:
            provider = getattr(self.tools, "retrieval_provider", None)
            if provider is None or not hasattr(provider, "validate_active_snapshot"):
                raise ResearchSafetyError("CANONICAL_PROVENANCE_INVALID")
            provider.validate_active_snapshot(state)
        state.step_count += 1
        state.tool_call_count += 1
        state.refresh_remaining_budget()
        tool_call_id = f"{state.task_id}-tool-{state.tool_call_count:04d}"
        if tool_call_id in state.completed_tool_call_ids:
            return
        started = time.perf_counter()
        observation = self.tools.execute(state, state.last_action, tool_call_id)
        if state.last_action.action == "retrieve_multimodal_evidence":
            state.tool_history.append({"phase": "NATIVE_TOOL", "tool_call_id": tool_call_id,
                                       "action": state.last_action.model_dump()})
        elif state.multimodal_enabled and state.last_action.action == "retrieve_evidence":
            state.tool_history.append({"phase": "MULTIMODAL_CONTEXT_TOOL",
                "tool_call_id": tool_call_id, "action": state.last_action.model_dump()})
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
        state.resolved_subquestions = []
        for subquestion in state.subquestions:
            subquestion.status = SubquestionStatus.OPEN
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
            item.id for item in state.subquestions if item.id not in state.resolved_subquestions
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
        self._restricted_finish(state)
        return True

    def _restricted_finish(self, state: AgentState) -> None:
        state.status = AgentStatus.PARTIAL
        state.stop_reason = StopReason.UNRESOLVABLE_GAP
        state.refresh_remaining_budget()
        self._checkpoint_and_trace(state, "FINISH", None)

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
            "unresolved_count": len(verification.unresolved_subquestions) if verification else 0,
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
        state.status = (
            AgentStatus.PARTIAL
            if reason
            in {
                StopReason.NO_PROGRESS,
                StopReason.MAX_STEPS_REACHED,
                StopReason.TOOL_BUDGET_EXHAUSTED,
                StopReason.VERIFICATION_FAILED_NO_BUDGET,
            }
            else AgentStatus.FAILED
        )
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
                "decision_id": (f"{state.task_id}-decision-{len(state.trace_event_ids) + 1:04d}"),
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
