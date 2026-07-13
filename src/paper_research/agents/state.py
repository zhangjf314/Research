import uuid
from typing import TypedDict

from pydantic import BaseModel, Field


class ResearchBudget(BaseModel):
    max_iterations: int = Field(default=3, ge=1, le=10)
    max_external_searches: int = Field(default=2, ge=0, le=10)
    max_papers: int = Field(default=10, ge=1, le=50)
    max_evidence_items: int = Field(default=40, ge=1, le=200)
    max_estimated_tokens: int = Field(default=30000, ge=1000)
    max_no_new_evidence_rounds: int = Field(default=2, ge=1, le=5)


class ResearchState(TypedDict, total=False):
    task_id: str
    original_query: str
    normalized_query: str
    research_goal: str
    research_plan: list[str]
    sub_questions: list[str]
    search_queries: list[str]
    requested_paper_ids: list[str]
    candidate_papers: list[dict]
    selected_papers: list[dict]
    local_evidence: list[dict]
    external_evidence: list[dict]
    evidence_gaps: list[str]
    contradictions: list[dict]
    draft_report: str
    citation_results: list[dict]
    iteration_count: int
    external_search_count: int
    no_new_evidence_rounds: int
    previous_evidence_count: int
    estimated_tokens: int
    budget: dict
    stop_reason: str | None
    status: str
    node_history: list[str]


def initial_state(
    query: str, budget: ResearchBudget, paper_ids: list[str] | None = None
) -> ResearchState:
    return ResearchState(
        task_id=str(uuid.uuid4()),
        original_query=query,
        requested_paper_ids=paper_ids or [],
        budget=budget.model_dump(),
        iteration_count=0,
        external_search_count=0,
        no_new_evidence_rounds=0,
        previous_evidence_count=0,
        estimated_tokens=0,
        local_evidence=[],
        external_evidence=[],
        candidate_papers=[],
        selected_papers=[],
        evidence_gaps=[],
        contradictions=[],
        citation_results=[],
        node_history=[],
        status="RUNNING",
        stop_reason=None,
    )
