"""Opt-in contract controller using the existing AgentState/checkpoint/trace."""

import re
from hashlib import sha256

from paper_research.agents.research_agent.contract_verifier import (
    AgenticFailure,
    contract_satisfied,
    detect_gaps,
    update_coverage,
    validate_assessments,
)
from paper_research.agents.research_agent.evidence_contract import (
    ActionType,
    ContractPlan,
    CoverageStatus,
    GapType,
    InformationGain,
    ReasonCode,
    ReplanRecord,
    RequirementCoverage,
    RetrievalAction,
    Verdict,
)
from paper_research.agents.research_agent.models import (
    AgentStatus,
    EvidenceItem,
    EvidenceVerificationStatus,
    StopReason,
    ToolAction,
    VerificationResult,
    VerificationStatus,
)
from paper_research.retrieval.frozen_backend import BACKEND_ID, FrozenBackendIdentityError
from paper_research.search.query_rewriter import QueryRewriter

ORDER = {
    CoverageStatus.UNSEARCHED: 0,
    CoverageStatus.SEARCHED_NO_EVIDENCE: 1,
    CoverageStatus.PARTIALLY_SUPPORTED: 2,
    CoverageStatus.SUPPORTED: 3,
}


def coverage_snapshot(state):
    return {key: value.model_dump(mode="json") for key, value in state.requirement_coverage.items()}


def improved(before, after):
    return [
        rid
        for rid, cov in after.items()
        if (
            before[rid]["status"] == "CONFLICTING"
            and (cov["status"] == "SUPPORTED" or cov["unresolvable"])
        )
        or ORDER.get(cov["status"], -1) > ORDER.get(before[rid]["status"], -1)
    ]


class EvidenceDrivenRetrievalController:
    def __init__(self, runner, planner, verifier):
        self.runner, self.planner, self.verifier = runner, planner, verifier

    def drive(self, state, interrupt_after_phase=None):
        r = self.runner
        try:
            if state.retrieval_backend_id != BACKEND_ID:
                raise FrozenBackendIdentityError("resume backend_id")
            r.frozen_backend.guard()
            if state.evidence_contract is None:
                self._plan(state)
                if self._checkpoint(state, "PLAN", interrupt_after_phase):
                    return state
            while state.status == AgentStatus.RUNNING:
                if state.pending_retrieval_action is None:
                    action = self._decide(state)
                    state.retrieval_actions.append(action)
                    state.pending_retrieval_action = action
                    state.last_action = ToolAction(
                        action="finish"
                        if action.action_type == ActionType.STOP
                        else "retrieve_evidence",
                        arguments={"query": action.query, "top_k": 5},
                        target_subquestion_ids=action.requirement_ids,
                        decision_reason=action.reason_code.value,
                    )
                    if self._checkpoint(
                        state,
                        "DECIDE",
                        interrupt_after_phase,
                        extra=self._action_trace(state, action),
                    ):
                        return state
                action = state.pending_retrieval_action
                if action.action_type == ActionType.STOP:
                    self._stop(state, action)
                    state.pending_retrieval_action = None
                    self._checkpoint(state, "FINISH", None, extra=self._action_trace(state, action))
                    break
                if state.pending_action_result is None:
                    self._retrieve(state, action)
                    if self._checkpoint(state, "RETRIEVED", interrupt_after_phase):
                        return state
                self._assess(state, action)
                if self._checkpoint(
                    state, "GAP", interrupt_after_phase, extra=state.tool_history[-1]
                ):
                    return state
        except FrozenBackendIdentityError:
            # Identity drift is not an evidence gap; never continue experimental calls.
            state.agentic_failure_reason = "FROZEN_BACKEND_IDENTITY_MISMATCH"
            state.status, state.stop_reason = AgentStatus.FAILED, StopReason.POLICY_FAILURE
            r.trace.append(
                state, phase="FAILURE", extra={"failure_reason": state.agentic_failure_reason}
            )
            raise
        except AgenticFailure as exc:
            state.agentic_failure_reason = str(exc)[:600]
            state.stop_reason = StopReason(exc.code)
            state.status = (
                AgentStatus.PARTIAL if exc.code == "BUDGET_EXHAUSTED" else AgentStatus.FAILED
            )
            r.trace.append(
                state, phase="FAILURE", extra={"failure_reason": state.agentic_failure_reason}
            )
        except Exception as exc:
            state.status, state.stop_reason = AgentStatus.FAILED, StopReason.POLICY_FAILURE
            state.agentic_failure_reason = "POLICY_FAILURE: " + type(exc).__name__
            r.trace.append(
                state, phase="FAILURE", extra={"failure_reason": state.agentic_failure_reason}
            )
            raise
        finally:
            state.refresh_remaining_budget()
            r.checkpoints.save(state, "FINAL")
            r.trace.append(
                state,
                phase="FINAL",
                extra={
                    "backend_id": state.retrieval_backend_id,
                    "contract_version": state.evidence_contract.contract_version
                    if state.evidence_contract
                    else None,
                    "failure_reason": state.agentic_failure_reason,
                },
            )
        return state

    def _plan(self, state):
        try:
            if hasattr(self.planner, "initial_contract_plan"):
                output = self.planner.initial_contract_plan(
                    state.research_question, state.paper_ids
                )
            else:
                output = self.planner.plan(state, self.runner)
            output = ContractPlan.model_validate(output)
        except AgenticFailure:
            raise
        except Exception as exc:
            raise AgenticFailure("CONTRACT_CREATION_FAILURE", type(exc).__name__) from exc
        contract = output.contract
        if any(req.min_sources > 1 for req in contract.requirements) and not re.search(
            r"compare|comparison|cross.paper|multi.source|versus|\bvs\b|跨论文|比较|多来源",
            state.research_question,
            re.I,
        ):
            raise AgenticFailure("CONTRACT_CREATION_FAILURE", "unsolicited source diversity")
        ids = {req.requirement_id for req in contract.requirements}
        if (
            len(ids) > state.agentic_config.max_requirements
            or contract.question != state.research_question
            or any(req.status != CoverageStatus.UNSEARCHED for req in contract.requirements)
            or any(
                state.paper_ids and not set(req.source_scope) <= set(state.paper_ids)
                for req in contract.requirements
            )
        ):
            raise AgenticFailure("CONTRACT_CREATION_FAILURE", "contract/plan/scope inconsistency")
        state.evidence_contract = contract
        from paper_research.agents.research_agent.models import Subquestion

        # SQ IDs and requirement IDs are separate namespaces. Canonical binding is
        # code-owned, using validated requirements without inventing fallback needs.
        output.plan.subquestions = [
            Subquestion(id=req.requirement_id, question=req.description)
            for req in contract.requirements
        ]
        state.contract_history = [contract.model_copy(deep=True)]
        state.current_plan, state.plan_version = output.plan, 1
        state.subquestions = output.plan.subquestions
        state.unresolved_subquestions = [sq.id for sq in state.subquestions]
        state.requirement_coverage = {rid: RequirementCoverage(requirement_id=rid) for rid in ids}
        state.contract_gaps = detect_gaps(state)

    def _decide(self, state):
        parent = state.retrieval_actions[-1] if state.retrieval_actions else None
        args = {
            "action_id": f"{state.task_id}-A{len(state.retrieval_actions) + 1:03d}",
            "parent_action_id": parent.action_id if parent else None,
        }
        state.refresh_remaining_budget()
        reason = None
        if contract_satisfied(state):
            reason = ReasonCode.CONTRACT_SATISFIED
        elif (
            state.remaining_step_budget < 2
            or state.remaining_tool_budget <= 0
            or state.remaining_token_budget <= 0
            or state.remaining_cost_budget <= 0
        ):
            reason = ReasonCode.BUDGET_EXHAUSTED
        elif state.targeted_no_gain_count >= state.agentic_config.no_information_gain_threshold:
            reason = ReasonCode.LOW_INFORMATION_GAIN
        elif parent and len(state.replan_history) >= state.agentic_config.max_replans:
            reason = ReasonCode.MAX_REPLANS
        if reason:
            return RetrievalAction(
                **args, action_type=ActionType.STOP, reason_code=reason, reason=reason.value
            )
        if parent is None:
            return RetrievalAction(
                **args,
                action_type=ActionType.INITIAL_SEARCH,
                query=state.research_question[:800],
                requirement_ids=[r.requirement_id for r in state.evidence_contract.requirements],
                reason_code=ReasonCode.INITIAL_REQUIREMENT,
                reason="initial contract retrieval",
            )
        reqs = {req.requirement_id: req for req in state.evidence_contract.requirements}
        gaps = [g for g in state.contract_gaps if reqs[g.requirement_id].required]
        if not gaps:
            return RetrievalAction(
                **args,
                action_type=ActionType.STOP,
                reason_code=ReasonCode.UNRESOLVABLE_GAP,
                reason="no actionable required gap",
            )
        gap = min(gaps, key=lambda g: (g.severity, g.requirement_id))
        req = reqs[gap.requirement_id]
        kind = ActionType.TARGETED_SEARCH
        code = ReasonCode.MISSING_EVIDENCE
        if gap.gap_type in {GapType.PARTIAL_SUPPORT, GapType.INSUFFICIENT_SOURCE_DIVERSITY}:
            code = ReasonCode(gap.gap_type.value)
        if gap.gap_type == GapType.CONTRADICTORY_EVIDENCE:
            code = ReasonCode.CONTRADICTION
        if gap.gap_type == GapType.RETRIEVAL_FAILURE:
            kind, code = ActionType.REFORMULATE_QUERY, ReasonCode.MISSING_EVIDENCE
        elif any(req.requirement_id in a.requirement_ids for a in state.retrieval_actions[1:]):
            kind = ActionType.SEARCH_SUBQUESTION
        seen = {a.query.strip().lower() for a in state.retrieval_actions}
        candidates = []
        if kind == ActionType.REFORMULATE_QUERY:
            candidates.extend(QueryRewriter().rewrite(parent.query)[1:])
        candidates.extend(
            [
                req.description,
                req.description + " " + state.research_question,
                state.research_question + " " + req.description,
            ]
        )
        query = next((q[:800] for q in candidates if q.strip().lower() not in seen), None)
        if query is None:
            return RetrievalAction(
                **args,
                action_type=ActionType.STOP,
                reason_code=ReasonCode.UNRESOLVABLE_GAP,
                requirement_ids=[req.requirement_id],
                trigger_gap_ids=[gap.gap_id],
                reason="all grounded query variants already searched",
            )
        return RetrievalAction(
            **args,
            action_type=kind,
            query=query,
            requirement_ids=[req.requirement_id],
            reason_code=code,
            reason=gap.description,
            trigger_gap_ids=[gap.gap_id],
        )

    def _retrieve(self, state, action):
        # Resume executes a durable pending action, not a new policy decision.
        if state.remaining_step_budget < 2 or state.remaining_tool_budget <= 0:
            raise AgenticFailure("BUDGET_EXHAUSTED", "pending retrieval reservation")
        before = coverage_snapshot(state)
        gaps = [g.model_dump(mode="json") for g in state.contract_gaps]
        state.step_count += 1
        state.tool_call_count += 1
        call_id = f"{state.task_id}-tool-{state.tool_call_count:04d}"
        try:
            raw_items = self.runner.frozen_backend.search(action.query, state.paper_ids, 5)
            error = None
        except FrozenBackendIdentityError:
            raise
        except Exception as exc:
            raw_items, error = [], "RETRIEVAL_FAILURE: " + type(exc).__name__
        retrieved, new, duplicates = [], [], []
        for raw in raw_items:
            chunk = self.runner.frozen_backend.chunks.get(raw["chunk_id"])
            if (
                chunk is None
                or raw["paper_id"] != chunk.paper_id
                or raw["block_ids"] != chunk.block_ids
            ):
                raise FrozenBackendIdentityError("authoritative evidence identity")
            eid = raw["evidence_id"]
            old_items = list(state.evidence_state.items.values())
            new_blocks = {
                (raw["paper_id"], block, raw["block_page_map"][block]) for block in raw["block_ids"]
            }
            duplicate = next(
                (
                    e
                    for e in old_items
                    if e.evidence_id == eid
                    or (e.paper_id, e.chunk_id) == (raw["paper_id"], raw["chunk_id"])
                    or (
                        new_blocks
                        and new_blocks
                        <= {
                            (e.paper_id, block, e.block_page_map.get(block, e.page))
                            for block in e.block_ids
                        }
                    )
                ),
                None,
            )
            if duplicate:
                retrieved.append(duplicate.evidence_id)
                duplicates.append(duplicate.evidence_id)
                continue
            page = raw["page_start"]
            item = EvidenceItem(
                evidence_id=eid,
                paper_id=raw["paper_id"],
                block_id=raw["chunk_id"],
                chunk_id=raw["chunk_id"],
                block_ids=raw["block_ids"],
                block_page_map=raw["block_page_map"],
                page=page,
                page_end=raw["page_end"],
                section=" > ".join(raw["section_path"]),
                text_or_reference=raw["text"],
                discovered_by_tool=call_id,
                discovered_at_step=state.step_count,
                relevance=raw["retrieval_score"],
            )
            state.evidence_state.add(item, action.requirement_ids)
            state.evidence_provenance[eid] = {
                k: raw[k]
                for k in [
                    "paper_id",
                    "chunk_id",
                    "block_ids",
                    "block_page_map",
                    "page_start",
                    "page_end",
                ]
            }
            retrieved.append(eid)
            new.append(eid)
        state.completed_tool_call_ids.append(call_id)
        state.pending_action_result = {
            "retrieved": list(dict.fromkeys(retrieved)),
            "new": new,
            "duplicates": duplicates,
            "error": error,
            "coverage_before": before,
            "gap_before": gaps,
        }
        state.refresh_remaining_budget()

    def _assess(self, state, action):
        result = state.pending_action_result
        # Include previously supporting/conflicting evidence when assessing a resolution;
        # never erase contradictions just because the new retrieval omits them.
        previous_ids = [
            a.evidence_id
            for a in state.evidence_assessments
            if a.requirement_id in action.requirement_ids and a.verdict != Verdict.IRRELEVANT
        ]
        ids = list(dict.fromkeys(result["retrieved"] + previous_ids))[
            : state.agentic_config.max_assessment_evidence
        ]
        assessments = []
        if ids:
            if self.verifier is None:
                raise AgenticFailure("VERIFIER_FAILURE", "semantic assessment provider required")
            assessments = self.verifier.assess(state, action.requirement_ids, ids, self.runner)
            validate_assessments(state, assessments, action.requirement_ids, ids)
            known = {(a.requirement_id, a.evidence_id): a for a in state.evidence_assessments}
            for assessment in assessments:
                known[(assessment.requirement_id, assessment.evidence_id)] = assessment
            state.evidence_assessments = list(known.values())
            useful = any(a.verdict != Verdict.IRRELEVANT for a in assessments)
        else:
            useful = False
        for rid in action.requirement_ids:
            if result["error"] or not useful:
                state.retrieval_failures[rid] = (
                    result["error"] or "retrieval returned no supporting evidence"
                )
            else:
                state.retrieval_failures.pop(rid, None)
        state.step_count += 1
        update_coverage(state)
        state.contract_gaps = detect_gaps(state)
        before, after = result["coverage_before"], coverage_snapshot(state)
        better = improved(before, after)
        contradicted_before = {
            g["requirement_id"]
            for g in result["gap_before"]
            if g["gap_type"] == "CONTRADICTORY_EVIDENCE"
        }
        contradictions = sorted(
            {
                g.requirement_id
                for g in state.contract_gaps
                if g.gap_type == GapType.CONTRADICTORY_EVIDENCE
            }
            - contradicted_before
        )
        gain = InformationGain(
            new_evidence_count=len(result["new"]),
            improved_requirements=better,
            newly_supported_requirements=[
                rid for rid in better if after[rid]["status"] == "SUPPORTED"
            ],
            contradictions_added=contradictions,
        )
        state.information_gain_history.append(gain)
        replan = action.action_type != ActionType.INITIAL_SEARCH
        effective = False
        if replan:
            effective = (
                bool(action.trigger_gap_ids)
                and any(before[rid]["status"] != "SUPPORTED" for rid in action.requirement_ids)
                and bool(result["new"])
                and any(rid in better for rid in action.requirement_ids)
                and all(
                    action.query.strip().lower() != old.query.strip().lower()
                    for old in state.retrieval_actions[:-1]
                )
            )
            state.replan_history.append(
                ReplanRecord(
                    trigger_gap_ids=action.trigger_gap_ids,
                    target_requirement_ids=action.requirement_ids,
                    previous_action=state.retrieval_actions[-2],
                    new_action=action,
                    reason=action.reason,
                    new_evidence_ids=result["new"],
                    improved_requirements=better,
                    effective_replan=effective,
                )
            )
            state.targeted_no_gain_count = (
                state.targeted_no_gain_count + 1 if not result["new"] and not better else 0
            )
        state.resolved_subquestions = [
            rid
            for rid, cov in state.requirement_coverage.items()
            if cov.status == CoverageStatus.SUPPORTED
        ]
        state.unresolved_subquestions = [
            rid for rid in state.requirement_coverage if rid not in state.resolved_subquestions
        ]
        state.evidence_gaps = [g.description for g in state.contract_gaps]
        for item in state.evidence_state.items.values():
            links = [a for a in state.evidence_assessments if a.evidence_id == item.evidence_id]
            item.supports = [a.requirement_id for a in links if a.verdict == Verdict.SUPPORTS]
            item.contradicts = [a.requirement_id for a in links if a.verdict == Verdict.CONTRADICTS]
            item.verification_status = (
                EvidenceVerificationStatus.CONTRADICTED
                if item.contradicts
                else EvidenceVerificationStatus.VERIFIED
                if item.supports
                else EvidenceVerificationStatus.CANDIDATE
            )
        event = self._action_trace(state, action) | {
            "phase": "AGENTIC_RETRIEVAL_ROUND",
            "retrieved_evidence_ids": result["retrieved"],
            "new_evidence_ids": result["new"],
            "duplicate_evidence_ids": result["duplicates"],
            "coverage_before": before,
            "coverage_after": after,
            "gap_before": result["gap_before"],
            "gap_after": [g.model_dump(mode="json") for g in state.contract_gaps],
            "information_gain": gain.model_dump(),
            "evidence_assessments": [a.model_dump(mode="json") for a in assessments],
            "replan_triggered": replan,
            "effective_replan": effective,
            "failure_reason": result["error"],
            "stop_reason": None,
        }
        state.tool_history.append(event)
        state.pending_retrieval_action, state.pending_action_result = None, None
        state.refresh_remaining_budget()

    def _action_trace(self, state, action):
        return {
            "backend_id": state.retrieval_backend_id,
            "execution_mode": self.runner.frozen_backend.execution_mode,
            "contract_version": state.evidence_contract.contract_version,
            "active_requirement_ids": action.requirement_ids,
            "action_id": action.action_id,
            "action_type": action.action_type.value,
            "action_query": action.query[:800],
            "action_reason": action.reason_code.value,
            "query_sha256": sha256(action.query.encode()).hexdigest(),
            "trigger_gap_ids": action.trigger_gap_ids,
            "parent_action_id": action.parent_action_id,
        }

    def _stop(self, state, action):
        mapping = {ReasonCode.LOW_INFORMATION_GAIN: StopReason.NO_INFORMATION_GAIN}
        state.stop_reason = mapping.get(
            action.reason_code,
            StopReason(action.reason_code.value)
            if action.reason_code != ReasonCode.LOW_INFORMATION_GAIN
            else None,
        )
        satisfied = action.reason_code == ReasonCode.CONTRACT_SATISFIED
        limitations = [
            cov.limitation for cov in state.requirement_coverage.values() if cov.unresolvable
        ]
        state.status = (
            AgentStatus.COMPLETED if satisfied and not limitations else AgentStatus.PARTIAL
        )
        state.verification_state = VerificationResult(
            status=VerificationStatus.PASS
            if satisfied and not limitations
            else VerificationStatus.PARTIAL,
            verified_claims=[
                state.requirement_coverage[rid].reason for rid in state.resolved_subquestions
            ],
            unresolved_subquestions=state.unresolved_subquestions,
            evidence_gaps=[g.description for g in state.contract_gaps] + limitations,
            recommended_next_action="FINISH",
        )
        state.evidence_gaps = state.verification_state.evidence_gaps

    def _checkpoint(self, state, phase, interrupt, extra=None):
        return self.runner._checkpoint_and_trace(state, phase, interrupt, extra=extra)
