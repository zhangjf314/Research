import json
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from paper_research.agents.checkpointing import checkpoint_saver
from paper_research.agents.deep_research_graph import DeepResearchGraph
from paper_research.agents.providers import (
    ArtifactLocalResearchProvider,
    HybridLocalResearchProvider,
    SearchServiceExternalProvider,
)
from paper_research.agents.research_agent import AgentBudget, ResearchAgentRunner
from paper_research.agents.research_agent.decision_provider import (
    AgentDecisionProviderError,
    LLMResearchAgentDecisionProvider,
)
from paper_research.agents.research_agent.final_report import (
    FINAL_REPORT_INPUT_FIELDS,
    AgentFinalReportStore,
    AgentFinalReportSynthesizer,
    AgentReportResult,
    AgentReportStatus,
)
from paper_research.agents.research_agent.models import AgentStatus, StopReason
from paper_research.agents.state import ResearchBudget
from paper_research.config import get_settings
from paper_research.db import get_db
from paper_research.providers.factory import (
    ProviderConfigurationError,
    build_llm_provider,
    build_research_synthesis_provider,
)
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
    succeeded: bool = False
    terminal: bool = True
    error_code: str | None = None
    stop_reason: str | None
    research_plan: list[str]
    sub_questions: list[str]
    evidence_gaps: list[str]
    candidate_papers: list[dict]
    contradictions: list[dict]
    citation_results: list[dict]
    node_history: list[str]
    report_path: str
    report_available: bool = False
    report: str
    report_quality: dict | None = None
    model_usage: dict | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    request_attempt_count: int = 0
    provider_completed_request_count: int = 0
    usage_record_count: int = 0
    active_reserved_tokens: int = 0


class ResearchAgentRequest(BaseModel):
    query: str = Field(min_length=3)
    paper_ids: list[str] | None = None
    budget: AgentBudget = Field(default_factory=AgentBudget)
    task_id: str | None = None
    pause_after_phase: str | None = None


class ResearchAgentResponse(BaseModel):
    task_id: str
    research_mode: str = "agent"
    status: str
    terminal: bool
    stop_reason: str | None
    plan_version: int
    subquestions: list[dict]
    resolved_subquestions: list[str]
    unresolved_subquestions: list[str]
    evidence_count: int
    observations: list[dict]
    tool_history: list[dict]
    step_count: int
    tool_call_count: int
    provider_call_count: int
    token_usage: dict
    estimated_cost: float
    remaining_budget: dict
    verification_state: dict | None
    checkpoint_id: str | None
    checkpoint_count: int
    failure_code: str | None = None
    report_status: str = "NOT_STARTED"
    report_markdown: str = ""
    report_available: bool = False
    report_failure_reason: str | None = None
    report_usage: dict = Field(default_factory=dict)
    report_provider_requests: int = 0
    agent_execution_provider_requests: int = 0
    agent_execution_tokens: dict = Field(default_factory=dict)
    agent_report_tokens: dict = Field(default_factory=dict)
    total_agent_user_request_tokens: int = 0
    report_claim_count: int = 0
    report_citation_count: int = 0
    report_evidence_references: list[dict] = Field(default_factory=list)
    final_report_input_fields: list[str] = Field(default_factory=list)


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
    state_path = output_dir / f"{state['task_id']}.json"
    report = state.get("draft_report", "")
    report_available = bool(isinstance(report, str) and report.strip())
    report_path = output_dir / f"{state['task_id']}.md"
    if report_available:
        report_path.write_text(report, encoding="utf-8")
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    status = state.get("status", "PAUSED")
    succeeded = status == "COMPLETED" and report_available
    terminal = status != "PAUSED"
    error_code = status if status.startswith("FAILED_") else None
    return DeepResearchResponse(
        task_id=state["task_id"],
        status=status,
        succeeded=succeeded,
        terminal=terminal,
        error_code=error_code,
        stop_reason=state.get("stop_reason"),
        research_plan=state.get("research_plan", []),
        sub_questions=state.get("sub_questions", []),
        evidence_gaps=state.get("evidence_gaps", []),
        candidate_papers=state.get("candidate_papers", []),
        contradictions=state.get("contradictions", []),
        citation_results=state.get("citation_results", []),
        node_history=state.get("node_history", []),
        report_path=str(report_path) if report_available else "",
        report_available=report_available,
        report=report,
        report_quality=state.get("report_quality"),
        model_usage=state.get("model_usage"),
        llm_provider=state.get("llm_provider"),
        llm_model=state.get("llm_model"),
        request_attempt_count=state.get("request_attempt_count", 0),
        provider_completed_request_count=state.get("provider_completed_request_count", 0),
        usage_record_count=state.get("usage_record_count", 0),
        active_reserved_tokens=state.get("active_reserved_tokens", 0),
    )


def _failed_state(task_id: str | None, status: str, stop_reason: str) -> dict:
    return {
        "task_id": task_id or f"failed-{uuid.uuid4().hex[:12]}",
        "status": status,
        "stop_reason": stop_reason,
        "research_plan": [],
        "sub_questions": [],
        "evidence_gaps": [stop_reason],
        "candidate_papers": [],
        "contradictions": [],
        "citation_results": [],
        "node_history": [status.lower()],
        "draft_report": "",
        "report_quality": None,
        "model_usage": {},
        "llm_provider": None,
        "llm_model": None,
        "request_attempt_count": 0,
        "provider_completed_request_count": 0,
        "usage_record_count": 0,
        "active_reserved_tokens": 0,
    }


def _agent_response(
    state,
    report_result: AgentReportResult | None = None,
) -> ResearchAgentResponse:
    failure_code = None
    if state.status == AgentStatus.FAILED and state.stop_reason == StopReason.PROVIDER_FAILURE:
        failure_code = "AGENT_DECISION_PROVIDER_ERROR"
    report_result = report_result or AgentFinalReportStore().load(state.task_id)
    report_usage = dict(report_result.usage or {})
    execution_tokens = state.token_usage.model_dump()
    total_tokens = int(execution_tokens.get("total_tokens") or 0) + int(
        report_usage.get("total_tokens") or 0
    )
    return ResearchAgentResponse(
        task_id=state.task_id,
        status=state.status.value if hasattr(state.status, "value") else str(state.status),
        terminal=state.status != AgentStatus.PAUSED,
        stop_reason=state.stop_reason.value if state.stop_reason else None,
        plan_version=state.plan_version,
        subquestions=[item.model_dump() for item in state.subquestions],
        resolved_subquestions=state.resolved_subquestions,
        unresolved_subquestions=state.unresolved_subquestions,
        evidence_count=len(state.evidence_state.items),
        observations=[item.model_dump() for item in state.observations],
        tool_history=state.tool_history,
        step_count=state.step_count,
        tool_call_count=state.tool_call_count,
        provider_call_count=state.provider_call_count,
        token_usage=state.token_usage.model_dump(),
        estimated_cost=state.estimated_cost,
        remaining_budget={
            "steps": state.remaining_step_budget,
            "tools": state.remaining_tool_budget,
            "tokens": state.remaining_token_budget,
            "cost_usd": state.remaining_cost_budget,
        },
        verification_state=state.verification_state.model_dump()
        if state.verification_state
        else None,
        checkpoint_id=state.checkpoint_id,
        checkpoint_count=len(state.checkpoint_chain),
        failure_code=failure_code,
        report_status=report_result.status.value
        if hasattr(report_result.status, "value")
        else str(report_result.status),
        report_markdown=report_result.markdown,
        report_available=bool(report_result.markdown.strip()),
        report_failure_reason=report_result.failure_reason,
        report_usage=report_usage,
        report_provider_requests=report_result.provider_request_count,
        agent_execution_provider_requests=state.provider_call_count,
        agent_execution_tokens=execution_tokens,
        agent_report_tokens=report_usage,
        total_agent_user_request_tokens=total_tokens,
        report_claim_count=report_result.claim_count,
        report_citation_count=report_result.citation_count,
        report_evidence_references=report_result.evidence_references,
        final_report_input_fields=list(FINAL_REPORT_INPUT_FIELDS),
    )


def _synthesize_agent_report_if_ready(
    runner: ResearchAgentRunner,
    state,
    settings,
    store: AgentFinalReportStore | None = None,
) -> AgentReportResult:
    store = store or AgentFinalReportStore()
    existing = store.load(state.task_id)
    if existing.markdown.strip() or existing.status != AgentReportStatus.NOT_STARTED:
        return existing
    if state.status != AgentStatus.COMPLETED:
        return existing
    verification_status = getattr(getattr(state, "verification_state", None), "status", None)
    if getattr(verification_status, "value", verification_status) != "PASS":
        return existing
    if settings.app_profile != "production" or settings.llm_provider == "template":
        result = AgentReportResult(
            status=AgentReportStatus.NOT_STARTED,
            failure_reason="real report provider is not configured",
        )
        store.save(state.task_id, result)
        return result
    try:
        llm = build_llm_provider(settings)
    except ProviderConfigurationError as exc:
        result = AgentReportResult(
            status=AgentReportStatus.FAILED_PROVIDER,
            failure_reason=str(exc),
        )
        store.save(state.task_id, result)
        return result
    synthesizer = AgentFinalReportSynthesizer(llm)
    result = synthesizer.synthesize(state)
    store.save(state.task_id, result)
    runner.trace.append(
        state,
        phase="FINAL_REPORT_SYNTHESIS",
        extra={
            "report_status": result.status.value,
            "report_provider_requests": result.provider_request_count,
            "report_claim_count": result.claim_count,
            "report_citation_count": result.citation_count,
        },
    )
    return result


def _materialize_agent_decision_provider_failure(
    runner: ResearchAgentRunner,
    task_id: str,
    exc: AgentDecisionProviderError,
):
    try:
        state = runner.checkpoints.load(task_id)
    except KeyError:
        state = None
    if state is None:
        raise exc
    state.status = AgentStatus.FAILED
    state.stop_reason = StopReason.PROVIDER_FAILURE
    state.retry_state.last_error = type(exc).__name__
    state.retry_state.retryable = False
    state.refresh_remaining_budget()
    state.tool_history.append(
        {
            "phase": "TERMINAL_FAILURE",
            "failure_code": "AGENT_DECISION_PROVIDER_ERROR",
            "root_exception": type(exc).__name__,
            "stop_reason": StopReason.PROVIDER_FAILURE.value,
        }
    )
    runner.checkpoints.save(state, "PROVIDER_FAILURE")
    runner.trace.append(
        state,
        phase="PROVIDER_FAILURE",
        extra={
            "failure_code": "AGENT_DECISION_PROVIDER_ERROR",
            "root_exception": type(exc).__name__,
            "failure_materialized": True,
        },
    )
    return state


@router.post("/deep", response_model=DeepResearchResponse)
def run_deep_research(payload: DeepResearchRequest, db: DbSession) -> DeepResearchResponse:
    try:
        settings, external, import_provider = _providers(payload, db)
        if not settings.deep_research_enabled:
            return _response(
                _failed_state(
                    payload.task_id,
                    "FAILED_PROVIDER_CONFIGURATION",
                    "DEEP_RESEARCH_ENABLED=false",
                )
            )
        try:
            local_provider = HybridLocalResearchProvider(settings)
        except Exception as exc:
            if settings.app_profile == "production":
                return _response(
                    _failed_state(
                        payload.task_id,
                        "FAILED_RETRIEVAL",
                        f"production hybrid retrieval unavailable: {type(exc).__name__}",
                    )
                )
            local_provider = ArtifactLocalResearchProvider(settings.parsed_papers_dir)
        try:
            synthesis_provider = build_research_synthesis_provider(settings)
        except ProviderConfigurationError as exc:
            if settings.app_profile == "production":
                return _response(
                    _failed_state(
                        payload.task_id,
                        "FAILED_PROVIDER_CONFIGURATION",
                        str(exc),
                    )
                )
            synthesis_provider = None
        with checkpoint_saver(settings) as saver:
            state = DeepResearchGraph(
                local_provider,
                external,
                import_provider,
                synthesis_provider=synthesis_provider,
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
        if not settings.deep_research_enabled:
            return _response(
                _failed_state(
                    payload.task_id,
                    "FAILED_PROVIDER_CONFIGURATION",
                    "DEEP_RESEARCH_ENABLED=false",
                )
            )
        local_provider = HybridLocalResearchProvider(settings)
        synthesis_provider = build_research_synthesis_provider(settings)
        with checkpoint_saver(settings) as saver:
            state = DeepResearchGraph(
                local_provider,
                external,
                import_provider,
                synthesis_provider=synthesis_provider,
                checkpointer=saver,
            ).resume(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        detail = f"deep research resume failed: {type(exc).__name__}"
        raise HTTPException(status_code=503, detail=detail) from exc
    return _response(state)


@router.post("/agent", response_model=ResearchAgentResponse)
def run_research_agent(payload: ResearchAgentRequest) -> ResearchAgentResponse:
    try:
        settings = get_settings()
        try:
            local_provider = HybridLocalResearchProvider(settings)
        except Exception as exc:
            if settings.app_profile == "production":
                raise HTTPException(
                    status_code=503,
                    detail=f"production hybrid retrieval unavailable: {type(exc).__name__}",
                ) from exc
            local_provider = ArtifactLocalResearchProvider(settings.parsed_papers_dir)
        task_id = payload.task_id or f"agent-{uuid.uuid4().hex[:12]}"
        decision_provider = _agent_decision_provider(settings)
        runner = ResearchAgentRunner(local_provider, decision_provider=decision_provider)
        try:
            state = runner.run(
                payload.query,
                task_id=task_id,
                paper_ids=payload.paper_ids,
                budget=payload.budget,
                interrupt_after_phase=payload.pause_after_phase,
            )
        except AgentDecisionProviderError as exc:
            state = _materialize_agent_decision_provider_failure(runner, task_id, exc)
        report_result = _synthesize_agent_report_if_ready(runner, state, settings)
        return _agent_response(state, report_result)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"research agent failed: {type(exc).__name__}",
        ) from exc


@router.get("/agent/{task_id}", response_model=ResearchAgentResponse)
def get_research_agent(task_id: str) -> ResearchAgentResponse:
    try:
        state = ResearchAgentRunner(None).checkpoints.load(task_id)
        return _agent_response(state)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/agent/{task_id}/resume", response_model=ResearchAgentResponse)
def resume_research_agent(task_id: str) -> ResearchAgentResponse:
    try:
        settings = get_settings()
        try:
            local_provider = HybridLocalResearchProvider(settings)
        except Exception as exc:
            if settings.app_profile == "production":
                raise HTTPException(
                    status_code=503,
                    detail=f"production hybrid retrieval unavailable: {type(exc).__name__}",
                ) from exc
            local_provider = ArtifactLocalResearchProvider(settings.parsed_papers_dir)
        decision_provider = _agent_decision_provider(settings)
        runner = ResearchAgentRunner(
            local_provider,
            decision_provider=decision_provider,
        )
        try:
            state = runner.resume(task_id)
        except AgentDecisionProviderError as exc:
            state = _materialize_agent_decision_provider_failure(runner, task_id, exc)
        report_result = _synthesize_agent_report_if_ready(runner, state, settings)
        return _agent_response(state, report_result)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"research agent resume failed: {type(exc).__name__}",
        ) from exc


def _agent_decision_provider(settings):
    if settings.app_profile != "production" or settings.llm_provider == "template":
        return None
    llm = build_llm_provider(settings)
    if not hasattr(llm, "generate_structured_json"):
        raise ProviderConfigurationError("research agent requires structured JSON LLM provider")
    return LLMResearchAgentDecisionProvider(llm)
