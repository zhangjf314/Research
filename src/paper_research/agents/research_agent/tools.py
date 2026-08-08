from __future__ import annotations

import math
from typing import Any, Protocol

from paper_research.agents.research_agent.models import (
    EvidenceItem,
    ObservationStatus,
    ToolAction,
    ToolObservation,
)
from paper_research.agents.research_agent.state import AgentState


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

    def execute(self, state: AgentState, action: ToolAction, tool_call_id: str) -> ToolObservation:
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
        try:
            raw_items = self.retrieval_provider.search(query, None, top_k)
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
    )
