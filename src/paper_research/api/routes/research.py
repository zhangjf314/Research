import json
import re
import secrets
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field, model_validator
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
    generate_unverified_draft,
    suppressed_report,
)
from paper_research.agents.research_agent.models import AgentStatus, StopReason, SubquestionStatus
from paper_research.agents.research_agent.checkpoint import (
    JsonResearchAgentCheckpointStore, checkpoint_kind, require_current_resume,
)
from paper_research.agents.research_agent.verifier import (
    REPORT_SAFETY_RECEIPT, ResearchSafetyError, validate_research_task_id,
)
from paper_research.api.routes.rag import HybridRetrievalEvidenceResponse
from paper_research.indexing.registry import IndexIdentityError
from paper_research.agents.state import ResearchBudget
from paper_research.config import get_settings
from paper_research.db import get_db
from paper_research.providers.factory import (
    ProviderConfigurationError,
    build_llm_provider,
    build_research_synthesis_provider,
)
from paper_research.providers.llm import LLMProviderError
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
    include_evidence_contract: bool = Field(default=False, strict=True)
    generate_draft: bool = Field(default=False, strict=True)
    multimodal_enabled: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def validate_readonly_opt_in(self):
        if self.multimodal_enabled and self.include_evidence_contract:
            raise ValueError("multimodal requires the ACTIVE Agent execution path")
        if self.generate_draft and (self.include_evidence_contract or self.pause_after_phase
                                    or not self.query.strip() or len(self.query) > 2000):
            raise ValueError("draft requires a nonblank query <=2000 chars and an unpaused ACTIVE run")
        if self.include_evidence_contract:
            if not self.query.strip() or len(self.query) > 2000:
                raise ValueError("readonly query must be nonblank and at most 2000 characters")
            if len(self.paper_ids or []) > 38:
                raise ValueError("readonly paper scope must contain at most 38 UUIDs")
            for paper_id in self.paper_ids or []:
                uuid.UUID(paper_id)
            if self.pause_after_phase is not None or self.budget.max_tool_calls < 1:
                raise ValueError("readonly requires no pause and at least one tool call")
            if self.task_id is not None and not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", self.task_id,
            ):
                raise ValueError("invalid readonly task_id")
        return self


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


class ResearchAgentEvidenceResponse(ResearchAgentResponse):
    retrieval_observation: HybridRetrievalEvidenceResponse


class ResearchAgentSafetyResponse(ResearchAgentResponse):
    evidence_trust: dict


class ResearchAgentActiveResponse(ResearchAgentSafetyResponse):
    active_provenance: dict


class ResearchAgentDraftResponse(ResearchAgentActiveResponse):
    research_draft: dict


def _readonly_snapshot(state) -> HybridRetrievalEvidenceResponse | None:
    keys = [key for key in state.evidence_provenance
            if key.startswith("p1_readonly_retrieval_bridge_")]
    if not keys:
        return None
    try:
        if keys != ["p1_readonly_retrieval_bridge_v1"]:
            raise ValueError("unknown readonly namespace")
        marker = state.evidence_provenance[keys[0]]
        if (set(marker) != {"mode", "schema_version", "retrieval_observation"}
                or marker["mode"] != "READONLY_RETRIEVAL_OBSERVATION"
                or type(marker["schema_version"]) is not int or marker["schema_version"] != 1):
            raise ValueError("invalid readonly marker")
        raw = marker["retrieval_observation"]
        snapshot = HybridRetrievalEvidenceResponse.model_validate(raw)
        # Require the complete serialized D1 wrapper, including nested default fields.
        # This rejects coercions, omissions and fields ignored by legacy nested models.
        if json.dumps(raw, sort_keys=True, allow_nan=False) != json.dumps(
            snapshot.model_dump(mode="json"), sort_keys=True, allow_nan=False,
        ):
            raise ValueError("noncanonical readonly wrapper")
        sidecar = snapshot.evidence_contract
        contract = sidecar.contract
        if (not re.fullmatch(r"EC-[a-f0-9]{16}", contract.contract_id)
                or sidecar.retrieval_identity.embedding_dimension < 1
                or any(not str(value) for value in sidecar.retrieval_identity.model_dump().values())
                or contract.created_by != "readonly_retrieve_adapter" or contract.contract_version != 1
                or contract.parent_contract_version is not None or contract.change_reason is not None
                or contract.added_requirements or contract.removed_requirements
                or contract.modified_requirements or len(contract.requirements) != 1):
            raise ValueError("invalid readonly contract")
        requirement = contract.requirements[0]
        if (requirement.requirement_id != "R1" or requirement.requirement_type != "SOURCE"
                or requirement.description != (
                    "Observe resolvable document/block/page provenance for this retrieval response."
                )
                or requirement.status != "UNSEARCHED" or requirement.priority != 1
                or not requirement.required or requirement.min_sources != 1):
            raise ValueError("invalid readonly requirement")
        for paper_id in requirement.source_scope:
            uuid.UUID(paper_id)
        for evidence in sidecar.evidence:
            item, provenance = evidence.item, evidence.provenance
            uuid.UUID(item.paper_id)
            uuid.UUID(provenance.paper_id)
            if (item.verification_status != "CANDIDATE" or item.supports or item.contradicts
                    or item.source_type != "local_retrieval"
                    or item.discovered_by_tool != "readonly_retrieve_adapter"
                    or item.discovered_at_step != 0 or not item.evidence_id or not item.chunk_id or not item.block_id
                    or item.page < 1 or item.page_end is None or item.page_end < 1
                    or not item.text_or_reference or not item.block_ids or not all(item.block_ids)
                    or not item.block_page_map
                    or len(item.block_ids) != len(set(item.block_ids))
                    or any(type(page) is not int or page < 1 for page in item.block_page_map.values())
                    or not all(provenance.block_ids)
                    or len(provenance.block_ids) != len(set(provenance.block_ids))
                    or any(page < 1 for page in provenance.block_page_map.values())):
                raise ValueError("invalid readonly evidence")
            for source in provenance.sources:
                uuid.UUID(source.paper_id)
        for gap in sidecar.gaps:
            if (gap.requirement_id != "R1" or gap.gap_type != "MISSING_EVIDENCE"
                    or gap.evidence_ids or gap.severity != 1 or not gap.gap_id):
                raise ValueError("invalid readonly gap")
        for coverage in sidecar.coverage:
            if len(coverage.evidence_ids) != len(set(coverage.evidence_ids)):
                raise ValueError("duplicate readonly evidence IDs")
        if any(error.item_index is not None and error.item_index < 0 for error in sidecar.errors):
            raise ValueError("invalid readonly error index")
        return snapshot
    except Exception as exc:
        raise HTTPException(status_code=503, detail="READONLY_OBSERVATION_INVALID") from exc


def _readonly_response(state, snapshot) -> ResearchAgentEvidenceResponse:
    response = _agent_response(state, AgentReportResult(status=AgentReportStatus.NOT_STARTED))
    return ResearchAgentEvidenceResponse(
        **{**response.model_dump(), "final_report_input_fields": []},
        retrieval_observation=snapshot,
    )


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
    kind = checkpoint_kind(state.model_dump(mode="json"))
    historical_status = state.status.value
    historical_verification = state.verification_state.status.value if state.verification_state else None
    archived = kind in {"LEGACY", "SCIENCE"}
    if archived:
        state = state.model_copy(deep=True)
        state.status = AgentStatus.PARTIAL
        state.stop_reason = StopReason.UNRESOLVABLE_GAP
        state.verification_state = None
        state.resolved_subquestions = []
        state.unresolved_subquestions = [item.id for item in state.subquestions]
        for item in state.subquestions:
            item.status = SubquestionStatus.OPEN
        state.observations = []
        state.tool_history = []
    failure_code = None
    if state.status == AgentStatus.FAILED and state.stop_reason == StopReason.PROVIDER_FAILURE:
        failure_code = "AGENT_DECISION_PROVIDER_ERROR"
    report_result = (AgentReportResult(status=AgentReportStatus.NOT_STARTED)
                     if kind == "READONLY" else suppressed_report())
    report_usage = dict(report_result.usage or {})
    execution_tokens = state.token_usage.model_dump()
    total_tokens = int(execution_tokens.get("total_tokens") or 0) + int(
        report_usage.get("total_tokens") or 0
    )
    response = ResearchAgentResponse(
        task_id=state.task_id,
        status=state.status.value if hasattr(state.status, "value") else str(state.status),
        terminal=state.status not in {AgentStatus.PAUSED, AgentStatus.RUNNING}
        if kind == "CURRENT_C2" else state.status != AgentStatus.PAUSED,
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
        final_report_input_fields=[],
    )
    if kind == "READONLY":
        return response
    trust = {
        "policy_id": REPORT_SAFETY_RECEIPT["policy_id"],
        "schema_version": 2 if kind == "CURRENT_C2" else 1,
        "checkpoint_view": "CURRENT_C2" if kind == "CURRENT_C2" else
            ("HISTORICAL_UNQUALIFIED" if archived else "CURRENT_C1"),
        "source_identity_schema": "canonical-agent-provenance-v2" if kind == "CURRENT_C2"
            else "legacy-opaque-v0",
        "structural_provenance_status": "VALID_AT_ADMISSION_SNAPSHOT" if kind == "CURRENT_C2"
            else "NOT_ASSESSED",
        "semantic_status": "SEMANTIC_SUPPORT_NOT_ASSESSED",
        "formal_report_eligibility": "DENIED_UNQUALIFIED_SEMANTIC_SUPPORT",
        "canonical_candidate_injection": kind == "CURRENT_C2",
        "resolution_semantics": "RETRIEVAL_PRESENCE_ONLY",
        "historical_agent_status": historical_status if archived else None,
        "historical_verification_status": historical_verification if archived else None,
        "historical_labels_qualified": False,
        "resume_allowed": not archived and state.status in {AgentStatus.PAUSED, AgentStatus.RUNNING},
    }
    if kind == "CURRENT_C2":
        receipt = state.evidence_provenance["p1_active_canonical_provenance_v2"]
        active = {
            "schema_version": "canonical-agent-provenance-v2",
            "snapshot_semantics": "VALIDATED_AT_ADMISSION_AS_OF_CHECKPOINT",
            "semantic_status": "SEMANTIC_SUPPORT_NOT_ASSESSED",
            "generation_binding": receipt["generation_binding"],
            "generation_binding_digest": receipt["generation_binding_digest"],
            "retrieval_identity": receipt["retrieval_identity"],
            "paper_scope": receipt["paper_scope"],
            "structural_errors": [error for observation in state.observations
                                  for error in getattr(observation, "structural_errors", [])],
            "structural_gaps": [gap for observation in state.observations
                                for gap in getattr(observation, "structural_gaps", [])],
            "evidence": [
                item.model_dump(mode="json") for item in state.evidence_state.items.values()
            ],
            "subquestion_evidence": state.evidence_state.subquestion_evidence,
        }
        return ResearchAgentActiveResponse(
            **response.model_dump(), evidence_trust=trust, active_provenance=active,
        )
    return ResearchAgentSafetyResponse(**response.model_dump(), evidence_trust=trust)


def _synthesize_agent_report_if_ready(
    runner: ResearchAgentRunner,
    state,
    settings,
    store: AgentFinalReportStore | None = None,
) -> AgentReportResult:
    return suppressed_report()


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
        except (IndexIdentityError, ProviderConfigurationError):
            raise
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
    except IndexIdentityError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
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
    except IndexIdentityError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except Exception as exc:
        detail = f"deep research resume failed: {type(exc).__name__}"
        raise HTTPException(status_code=503, detail=detail) from exc
    return _response(state)


def _authorize_research_draft(settings, token):
    if not settings.research_draft_enabled or not settings.live_model_calls_enabled:
        raise HTTPException(status_code=403, detail="RESEARCH_DRAFT_DISABLED")
    expected = settings.research_draft_token
    if (expected is None or not expected.get_secret_value() or not isinstance(token, str)
            or not secrets.compare_digest(token.encode(), expected.get_secret_value().encode())):
        raise HTTPException(status_code=403, detail="RESEARCH_DRAFT_AUTHORIZATION_REQUIRED")
    if settings.llm_provider == "template":
        raise HTTPException(status_code=503, detail="RESEARCH_DRAFT_REAL_LLM_REQUIRED")


def _agent_draft(state, local_provider, settings):
    # A draft opt-in authorizes one synthesis request, never automatic retries.
    provider = None
    try:
        provider = build_llm_provider(settings.model_copy(update={
            "llm_max_retries": 0, "llm_thinking_enabled": False, "llm_stream": False,
        }))
        semantic_verifier = None
        if settings.research_semantic_review_enabled:
            from paper_research.agents.research_agent.semantic_review import StructuredSemanticVerifier
            semantic_verifier = StructuredSemanticVerifier(provider, task_id=state.task_id,
                receipt_dir=settings.research_semantic_receipt_dir,
                max_output_tokens=min(settings.llm_max_output_tokens, 3500))
        draft = generate_unverified_draft(
            state, local_provider, provider,
            max_output_tokens=min(settings.llm_max_output_tokens, 4096),
            semantic_verifier=semantic_verifier,
            semantic_allowed_calls=1 if semantic_verifier is not None else 0,
            vision_enabled=settings.llm_vision_enabled,
        )
        if draft.get("status") == "UNVERIFIED" and semantic_verifier is None:
            from paper_research.agents.research_agent.semantic_review import prepare_semantic_review
            draft["semantic_review"] = prepare_semantic_review(state, local_provider, draft)
        return draft
    except LLMProviderError as exc:
        return {"status": "PROVIDER_ERROR", "reason": exc.error_code,
                "request_count": exc.api_request_count,
                "usage_records": [record.model_dump() for record in exc.usage_records]}
    except ProviderConfigurationError:
        return {"status": "CONFIGURATION_ERROR", "reason": "check LLM configuration"}
    finally:
        if provider is not None and hasattr(provider, "client"):
            provider.client.close()


@router.post("/agent", response_model=ResearchAgentDraftResponse | ResearchAgentEvidenceResponse | ResearchAgentActiveResponse | ResearchAgentSafetyResponse)
def run_research_agent(
    payload: ResearchAgentRequest,
    draft_token: str | None = Header(default=None, alias="X-Research-Draft-Token"),
    semantic_token: str | None = Header(default=None, alias="X-Research-Semantic-Token"),
) -> ResearchAgentResponse:
    try:
        settings = get_settings()
        if payload.generate_draft:
            _authorize_research_draft(settings, draft_token)
            if settings.research_semantic_review_enabled:
                expected = settings.research_semantic_review_token
                if (expected is None or not expected.get_secret_value()
                        or not isinstance(semantic_token, str)
                        or not secrets.compare_digest(expected.get_secret_value().encode(), semantic_token.encode())):
                    raise HTTPException(status_code=403, detail="INDEPENDENT_SEMANTIC_BUDGET_AUTHORIZATION_REQUIRED")
        task_id = payload.task_id or f"agent-{uuid.uuid4().hex[:12]}"
        validate_research_task_id(task_id)
        checkpoints = JsonResearchAgentCheckpointStore()
        reports = AgentFinalReportStore()
        if payload.include_evidence_contract and (checkpoints.path_for(task_id).exists()
                or (Path(".runtime/research-agent/traces") / f"{task_id}.jsonl").exists()
                or (reports.root / f"{task_id}-final-report.json").exists()
                or (reports.root / f"{task_id}-final-report.md").exists()):
            raise ResearchSafetyError("RESEARCH_TASK_ALREADY_EXISTS", 409)
        if payload.include_evidence_contract:
            if settings.retrieval_v2_enabled:
                raise HTTPException(status_code=422, detail="READONLY_RETRIEVAL_MODE_UNSUPPORTED")
        if payload.include_evidence_contract:
            local_provider = HybridLocalResearchProvider(settings, require_identity=True)
            runner = ResearchAgentRunner(local_provider, readonly=True)
            state = runner.run_readonly_retrieval(
                payload.query, task_id=task_id, paper_ids=payload.paper_ids, budget=payload.budget,
            )
            return _readonly_response(state, _readonly_snapshot(state))
        with checkpoints.task_write_lease(task_id):
            if (checkpoints.path_for(task_id).exists()
                    or (Path(".runtime/research-agent/traces") / f"{task_id}.jsonl").exists()
                    or (reports.root / f"{task_id}-final-report.json").exists()
                    or (reports.root / f"{task_id}-final-report.md").exists()):
                raise ResearchSafetyError("RESEARCH_TASK_ALREADY_EXISTS", 409)
            local_provider = HybridLocalResearchProvider(settings, require_identity=True)
            decision_provider = _agent_decision_provider(settings)
            if payload.multimodal_enabled:
                from paper_research.agents.research_agent.multimodal import configured_multimodal_provider
                local_provider = configured_multimodal_provider(local_provider, settings)
            runner = ResearchAgentRunner(local_provider, decision_provider=decision_provider)
            try:
                state = runner.run(
                    payload.query,
                    task_id=task_id,
                    paper_ids=payload.paper_ids,
                    budget=payload.budget,
                    interrupt_after_phase=payload.pause_after_phase,
                    multimodal_enabled=payload.multimodal_enabled,
                )
            except AgentDecisionProviderError as exc:
                state = _materialize_agent_decision_provider_failure(runner, task_id, exc)
            report_result = _synthesize_agent_report_if_ready(runner, state, settings)
            response = _agent_response(state, report_result)
            if payload.generate_draft:
                if not isinstance(response, ResearchAgentActiveResponse):
                    raise HTTPException(status_code=409, detail="RESEARCH_DRAFT_ACTIVE_REQUIRED")
                draft = _agent_draft(state, local_provider, settings)
                response.total_agent_user_request_tokens += int(
                    draft.get("usage", {}).get("total_tokens") or 0,
                )
                response.total_agent_user_request_tokens += int(
                    draft.get("semantic_review", {}).get("verifier", {}).get(
                        "usage", {}).get("usage", {}).get("total_tokens") or 0,
                )
                return ResearchAgentDraftResponse(
                    **response.model_dump(), research_draft=draft,
                )
            return response
    except HTTPException:
        raise
    except ResearchSafetyError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.code) from exc
    except IndexIdentityError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"research agent failed: {type(exc).__name__}",
        ) from exc


@router.get("/agent/{task_id}", response_model=ResearchAgentEvidenceResponse | ResearchAgentActiveResponse | ResearchAgentSafetyResponse)
def get_research_agent(task_id: str) -> ResearchAgentResponse:
    try:
        state = JsonResearchAgentCheckpointStore().load(task_id)
        snapshot = _readonly_snapshot(state)
        if snapshot is not None:
            return _readonly_response(state, snapshot)
        return _agent_response(state)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HTTPException:
        raise
    except ResearchSafetyError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.code) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="READONLY_OBSERVATION_INVALID") from exc


@router.post("/agent/{task_id}/resume", response_model=ResearchAgentEvidenceResponse | ResearchAgentActiveResponse | ResearchAgentSafetyResponse)
def resume_research_agent(task_id: str) -> ResearchAgentResponse:
    try:
        checkpoints = JsonResearchAgentCheckpointStore()
        with checkpoints.task_write_lease(task_id):
            previous = checkpoints.load(task_id)
            if _readonly_snapshot(previous) is not None:
                raise HTTPException(status_code=409, detail="READONLY_OBSERVATION_TERMINAL")
            require_current_resume(previous)
            settings = get_settings()
            local_provider = HybridLocalResearchProvider(settings, require_identity=True)
            decision_provider = None if previous.verification_state is not None else _agent_decision_provider(settings)
            if previous.multimodal_enabled:
                from paper_research.agents.research_agent.multimodal import configured_multimodal_provider
                local_provider = configured_multimodal_provider(local_provider, settings)
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
    except ResearchSafetyError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.code) from exc
    except IndexIdentityError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
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
