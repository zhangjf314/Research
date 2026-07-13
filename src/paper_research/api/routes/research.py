import json
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from paper_research.agents.checkpointing import checkpoint_saver
from paper_research.agents.deep_research_graph import DeepResearchGraph
from paper_research.agents.providers import (
    ArtifactLocalResearchProvider,
    SearchServiceExternalProvider,
)
from paper_research.agents.state import ResearchBudget
from paper_research.config import get_settings
from paper_research.db import get_db
from paper_research.search.clients import ArxivClient, SemanticScholarClient
from paper_research.search.http import CachedRetryClient
from paper_research.search.import_service import PaperImportService
from paper_research.search.models import PaperCandidate
from paper_research.search.service import PaperSearchService

router = APIRouter()
DbSession = Annotated[Session, Depends(get_db)]


class DeepResearchRequest(BaseModel):
    query: str = Field(min_length=3)
    paper_ids: list[str] | None = None
    allow_external_search: bool = True
    allow_external_import: bool = False
    budget: ResearchBudget = Field(default_factory=ResearchBudget)
    task_id: str | None = None
    pause_after_node: str | None = None


class DeepResearchResponse(BaseModel):
    task_id: str
    status: str
    stop_reason: str | None
    research_plan: list[str]
    sub_questions: list[str]
    evidence_gaps: list[str]
    candidate_papers: list[dict]
    contradictions: list[dict]
    citation_results: list[dict]
    node_history: list[str]
    report_path: str
    report: str


def _providers(payload: DeepResearchRequest, db: Session):
    settings = get_settings()
    external = None
    http = None
    if payload.allow_external_search:
        http = CachedRetryClient(
            settings.search_cache_dir,
            settings.search_cache_ttl_seconds,
            settings.external_request_retries,
        )
        external = SearchServiceExternalProvider(
            PaperSearchService(
                [ArxivClient(http), SemanticScholarClient(http, settings.semantic_scholar_api_key)]
            )
        )
    import_provider = None
    if payload.allow_external_import and http is not None:
        importer = PaperImportService(db, settings, http)

        def import_provider(candidate: dict) -> str | None:
            result = importer.import_candidate(PaperCandidate.model_validate(candidate))
            return str(result.paper.id)

    return settings, external, import_provider


def _response(state: dict) -> DeepResearchResponse:
    output_dir = Path("data/reports/research")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{state['task_id']}.md"
    state_path = output_dir / f"{state['task_id']}.json"
    report = state.get("draft_report", "")
    report_path.write_text(report, encoding="utf-8")
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return DeepResearchResponse(
        task_id=state["task_id"],
        status=state.get("status", "PAUSED"),
        stop_reason=state.get("stop_reason"),
        research_plan=state.get("research_plan", []),
        sub_questions=state.get("sub_questions", []),
        evidence_gaps=state.get("evidence_gaps", []),
        candidate_papers=state.get("candidate_papers", []),
        contradictions=state.get("contradictions", []),
        citation_results=state.get("citation_results", []),
        node_history=state.get("node_history", []),
        report_path=str(report_path),
        report=report,
    )


@router.post("/deep", response_model=DeepResearchResponse)
def run_deep_research(payload: DeepResearchRequest, db: DbSession) -> DeepResearchResponse:
    try:
        settings, external, import_provider = _providers(payload, db)
        with checkpoint_saver(settings) as saver:
            state = DeepResearchGraph(
                ArtifactLocalResearchProvider(settings.parsed_papers_dir),
                external,
                import_provider,
                checkpointer=saver,
                interrupt_after=[payload.pause_after_node] if payload.pause_after_node else None,
            ).run(
                payload.query,
                budget=payload.budget,
                paper_ids=payload.paper_ids,
                task_id=payload.task_id,
            )
    except Exception as exc:
        detail = f"deep research failed: {type(exc).__name__}"
        raise HTTPException(status_code=503, detail=detail) from exc
    return _response(state)


@router.post("/deep/{task_id}/resume", response_model=DeepResearchResponse)
def resume_deep_research(
    task_id: str,
    payload: DeepResearchRequest,
    db: DbSession,
) -> DeepResearchResponse:
    try:
        settings, external, import_provider = _providers(payload, db)
        with checkpoint_saver(settings) as saver:
            state = DeepResearchGraph(
                ArtifactLocalResearchProvider(settings.parsed_papers_dir),
                external,
                import_provider,
                checkpointer=saver,
            ).resume(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        detail = f"deep research resume failed: {type(exc).__name__}"
        raise HTTPException(status_code=503, detail=detail) from exc
    return _response(state)
