from __future__ import annotations

import math
from typing import Any, Protocol

from paper_research.agents.research_agent.models import (
    CanonicalToolObservation,
    EvidenceItem,
    ObservationStatus,
    ToolAction,
    ToolObservation,
)
from paper_research.agents.research_agent.state import AgentState
from paper_research.indexing.registry import IndexIdentityError
from paper_research.agents.research_agent.verifier import ResearchSafetyError


class EvidenceSearchProvider(Protocol):
    def search(
        self,
        query: str,
        paper_ids: list[str] | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Return existing retrieval-service evidence dictionaries."""


class ResearchAgentToolRegistry:
    def __init__(self, retrieval_provider: EvidenceSearchProvider | None = None) -> None:
        self.retrieval_provider = retrieval_provider
        self.tool_names = [
            "retrieve_evidence",
            "inspect_evidence",
            "inspect_paper",
            "verify_evidence",
            "finish",
        ]
        if retrieval_provider is not None and hasattr(retrieval_provider, "retrieve_native"):
            self.tool_names.append("retrieve_multimodal_evidence")

    def execute(self, state: AgentState, action: ToolAction, tool_call_id: str) -> ToolObservation:
        if action.action == "retrieve_multimodal_evidence":
            return self._retrieve_native(state, action, tool_call_id)
        if action.action == "retrieve_evidence":
            return self._retrieve_evidence(state, action, tool_call_id)
        if action.action in {"inspect_evidence", "inspect_paper", "verify_evidence", "finish"}:
            return ToolObservation(
                tool_call_id=tool_call_id,
                tool=action.action,
                status=ObservationStatus.SUCCESS,
                target_subquestions=action.target_subquestion_ids,
                new_information=False,
            )
        return ToolObservation(
            tool_call_id=tool_call_id,
            tool=action.action,
            status=ObservationStatus.FAILED,
            target_subquestions=action.target_subquestion_ids,
            possible_gaps=[f"unknown tool: {action.action}"],
            error=f"unknown tool: {action.action}",
            retryable=False,
        )

    def _retrieve_native(self, state, action, tool_call_id):
        if (not state.multimodal_enabled or self.retrieval_provider is None
                or not hasattr(self.retrieval_provider, "retrieve_native")):
            raise ResearchSafetyError("MULTIMODAL_OPT_IN_REQUIRED", 422)
        known = {sq.id for sq in state.subquestions}
        targets = action.target_subquestion_ids
        if not targets or len(set(targets)) != len(targets) or not set(targets) <= known:
            raise ResearchSafetyError("ACTIVE_SUBQUESTION_INVALID")
        items = self.retrieval_provider.retrieve_native(state, action, tool_call_id)
        # Commit only after the entire batch validates; no partial admission.
        staged = state.evidence_state.model_copy(deep=True)
        added, duplicates, linked = [], [], []
        for item in items:
            key = item.stable_key
            previous = staged.items.get(key)
            if previous is None:
                staged.items[key] = item
                added.append(key)
            else:
                invariant = lambda x: x.model_dump(exclude={
                    "occurrences", "discovered_by_tool", "discovered_at_step", "relevance"})
                if type(previous) is not type(item) or invariant(previous) != invariant(item):
                    raise ResearchSafetyError("NATIVE_PROVENANCE_CONFLICT")
                previous.occurrences.extend(item.occurrences)
                duplicates.append(key)
            for target in targets:
                relation = staged.subquestion_evidence.setdefault(target, [])
                if key not in relation:
                    relation.append(key)
                    linked.append(f"{target}:{key}")
        state.evidence_state = staged
        return CanonicalToolObservation(
            tool_call_id=tool_call_id, tool=action.action, status=ObservationStatus.SUCCESS,
            target_subquestions=targets, evidence_added=added, evidence_duplicates=duplicates,
            evidence_linked=linked, new_information=bool(added or linked),
            retrieval_trace_id=tool_call_id,
            possible_gaps=[] if items else ["no source-valid native evidence for this modality"],
        )

    def _retrieve_evidence(
        self,
        state: AgentState,
        action: ToolAction,
        tool_call_id: str,
    ) -> ToolObservation:
        if self.retrieval_provider is None:
            return ToolObservation(
                tool_call_id=tool_call_id,
                tool="retrieve_evidence",
                status=ObservationStatus.FAILED,
                target_subquestions=action.target_subquestion_ids,
                possible_gaps=["retrieval provider unavailable"],
                error="retrieval provider unavailable",
                retryable=True,
            )
        query = str(action.arguments.get("query") or state.research_question)
        top_k = int(action.arguments.get("top_k") or 5)
        top_k = max(1, min(top_k, 20))
        active = state.evidence_provenance.get("p1_active_canonical_provenance_v2")
        if active is not None:
            if not hasattr(self.retrieval_provider, "retrieve_active_observation"):
                raise ResearchSafetyError("CANONICAL_PROVENANCE_INVALID")
            known = {subquestion.id for subquestion in state.subquestions}
            targets = action.target_subquestion_ids
            if len(targets) != len(set(targets)) or any(target not in known for target in targets):
                raise ResearchSafetyError("ACTIVE_SUBQUESTION_INVALID")
            narrowed = action.arguments.get("paper_ids")
            options = {}
            if narrowed is not None:
                if (not isinstance(narrowed, list) or not narrowed
                        or any(not isinstance(p, str) for p in narrowed)
                        or len(set(narrowed)) != len(narrowed)
                        or not set(narrowed) <= set(state.paper_ids or [])):
                    raise ResearchSafetyError("CANONICAL_PROVENANCE_INVALID")
                options["retrieval_paper_ids"] = narrowed
            try:
                batch = self.retrieval_provider.retrieve_active_observation(
                    query, state.paper_ids, top_k, tool_call_id, state, **options,
                )
            except (IndexIdentityError, ResearchSafetyError):
                raise
            except Exception as exc:
                return ToolObservation(
                    tool_call_id=tool_call_id, tool="retrieve_evidence",
                    status=ObservationStatus.FAILED,
                    target_subquestions=action.target_subquestion_ids,
                    possible_gaps=[f"retrieval failed: {type(exc).__name__}"],
                    error=f"{type(exc).__name__}: {exc}", retryable=True,
                )
            if (batch.generation_binding_digest != active.get("generation_binding_digest")
                    or batch.generation_binding != active.get("generation_binding")):
                raise ResearchSafetyError("CANONICAL_PROVENANCE_INVALID")
            added, duplicates, linked = state.evidence_state.add_canonical_batch(
                batch.items, targets, known,
            )
            return CanonicalToolObservation(
                tool_call_id=tool_call_id, tool="retrieve_evidence",
                status=ObservationStatus.SUCCESS, target_subquestions=targets,
                evidence_added=added, evidence_duplicates=duplicates,
                evidence_linked=linked, new_information=bool(added or linked),
                structural_errors=batch.errors, structural_gaps=batch.gaps,
                structural_coverage=batch.coverage, retrieval_trace_id=batch.trace_id,
                possible_gaps=[error.get("code", "SOURCE_INVALID") for error in batch.errors]
                if batch.errors else ([] if added or linked else ["no new evidence returned"]),
            )
        try:
            raw_items = self.retrieval_provider.search(query, state.paper_ids, top_k)
        except IndexIdentityError:
            raise
        except Exception as exc:
            return ToolObservation(
                tool_call_id=tool_call_id,
                tool="retrieve_evidence",
                status=ObservationStatus.FAILED,
                target_subquestions=action.target_subquestion_ids,
                possible_gaps=[f"retrieval failed: {type(exc).__name__}"],
                error=f"{type(exc).__name__}: {exc}",
                retryable=True,
            )
        added: list[str] = []
        duplicates: list[str] = []
        for raw in raw_items:
            item = _evidence_from_raw(raw, state.step_count + 1, tool_call_id)
            if state.evidence_state.add(item, action.target_subquestion_ids):
                added.append(item.stable_key)
            else:
                duplicates.append(item.stable_key)
        return ToolObservation(
            tool_call_id=tool_call_id,
            tool="retrieve_evidence",
            status=ObservationStatus.SUCCESS,
            target_subquestions=action.target_subquestion_ids,
            evidence_added=added,
            evidence_duplicates=duplicates,
            new_information=bool(added),
            possible_gaps=[] if added else ["no new evidence returned"],
        )


def _evidence_from_raw(raw: dict[str, Any], step: int, tool_call_id: str) -> EvidenceItem:
    page = int(raw.get("page_start") or raw.get("page") or raw.get("source_page") or 0)
    block_id = str(raw.get("block_id") or raw.get("evidence_id") or raw.get("chunk_id"))
    score = raw.get("retrieval_score", raw.get("score"))
    relevance = float(score) if isinstance(score, int | float) and math.isfinite(score) else None
    return EvidenceItem(
        evidence_id=str(raw.get("evidence_id") or block_id),
        paper_id=str(raw.get("paper_id")),
        block_id=block_id,
        page=page,
        section=" > ".join(raw.get("section_path") or []),
        text_or_reference=str(raw.get("text") or raw.get("quote") or "")[:800],
        discovered_by_tool=tool_call_id,
        discovered_at_step=step,
        relevance=relevance,
        chunk_id=raw.get("chunk_id"),
        block_ids=raw.get("block_ids") or [],
        block_page_map=raw.get("block_page_map") or {},
        page_end=raw.get("page_end"),
    )
