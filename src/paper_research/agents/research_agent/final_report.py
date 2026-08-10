from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field, ValidationError, model_validator

from paper_research.agents.research_agent.models import (
    AgentStatus,
    EvidenceItem,
    VerificationStatus,
)
from paper_research.agents.research_agent.state import AgentState
from paper_research.providers.llm import LLMProviderError, StructuredJSONResult

AGENT_FINAL_REPORT_SYNTHESIS = True
FINAL_REPORT_INPUT_FIELDS = [
    "original research query",
    "verified evidence blocks",
    "paper metadata",
    "evidence IDs",
    "page numbers",
    "verification status",
    "research gaps",
    "tool observations",
    "paper IDs",
]
SYSTEM_DERIVED_CONTROL_FIELDS = {
    "verification_pass",
    "report_valid",
    "citation_valid",
    "insufficient_evidence",
    "quality_gate_pass",
    "status",
    "completed",
}


class AgentReportStatus(StrEnum):
    NOT_STARTED = "NOT_STARTED"
    AVAILABLE = "AVAILABLE"
    FAILED_PROVIDER = "FAILED_PROVIDER"
    FAILED_SCHEMA = "FAILED_SCHEMA"
    FAILED_VALIDATION = "FAILED_VALIDATION"
    REPORT_INPUT_INVALID = "REPORT_INPUT_INVALID"


class StructuredReportProvider(Protocol):
    def generate_structured_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        request_context: dict[str, Any] | None = None,
        max_output_tokens: int | None = None,
    ) -> StructuredJSONResult:
        """Return a provider-authored JSON object."""


class AgentReportClaimDraft(BaseModel):
    text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)


class AgentReportSectionDraft(BaseModel):
    title: str = Field(min_length=1)
    claims: list[AgentReportClaimDraft] = Field(default_factory=list)


class AgentFinalReportDraft(BaseModel):
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    sections: list[AgentReportSectionDraft] = Field(min_length=1)
    research_gaps: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def reject_model_authored_control_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            found = _find_control_fields(data)
            if found:
                joined = ", ".join(sorted(found))
                raise ValueError(f"model-authored control fields are not allowed: {joined}")
        return data

    @model_validator(mode="after")
    def validate_user_facing_content(self) -> AgentFinalReportDraft:
        claim_count = sum(len(section.claims) for section in self.sections)
        if claim_count <= 0:
            raise ValueError("final report requires at least one evidence-linked claim")
        return self


class AgentReportEvidence(BaseModel):
    evidence_id: str
    paper_id: str
    block_id: str
    page: int
    section: str = ""
    text: str
    stable_key: str


class AgentReportResult(BaseModel):
    status: AgentReportStatus
    markdown: str = ""
    failure_reason: str | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    provider_request_count: int = 0
    claim_count: int = 0
    citation_count: int = 0
    evidence_references: list[dict[str, Any]] = Field(default_factory=list)


class AgentFinalReportStore:
    def __init__(self, root: Path = Path("data/reports/research-agent")) -> None:
        self.root = root

    def save(self, task_id: str, result: AgentReportResult) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = result.model_dump(mode="json")
        (self.root / f"{task_id}-final-report.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if result.markdown.strip():
            (self.root / f"{task_id}-final-report.md").write_text(
                result.markdown,
                encoding="utf-8",
            )

    def load(self, task_id: str) -> AgentReportResult:
        path = self.root / f"{task_id}-final-report.json"
        if not path.exists():
            return AgentReportResult(status=AgentReportStatus.NOT_STARTED)
        return AgentReportResult.model_validate_json(path.read_text(encoding="utf-8"))


class AgentFinalReportCompiler:
    def compile(
        self,
        draft_payload: dict[str, Any],
        *,
        evidence_by_id: dict[str, AgentReportEvidence],
    ) -> AgentReportResult:
        try:
            draft = AgentFinalReportDraft.model_validate(draft_payload)
        except ValidationError as exc:
            return AgentReportResult(
                status=AgentReportStatus.FAILED_SCHEMA,
                failure_reason=_validation_message(exc),
            )
        citation_count = 0
        lines = [f"# {draft.title.strip()}", "", "## Summary", "", draft.summary.strip(), ""]
        for section in draft.sections:
            lines.extend([f"## {section.title.strip()}", ""])
            if not section.claims:
                continue
            for claim in section.claims:
                unknown = [item for item in claim.evidence_ids if item not in evidence_by_id]
                if unknown:
                    return AgentReportResult(
                        status=AgentReportStatus.FAILED_VALIDATION,
                        failure_reason="AGENT_REPORT_INVALID_CITATION: "
                        + ", ".join(sorted(unknown)),
                    )
                if not claim.evidence_ids:
                    return AgentReportResult(
                        status=AgentReportStatus.FAILED_VALIDATION,
                        failure_reason="AGENT_REPORT_CLAIM_WITHOUT_CITATION",
                    )
                citation_count += len(claim.evidence_ids)
                citations = " ".join(f"[{item}]" for item in claim.evidence_ids)
                lines.append(f"- {claim.text.strip()} {citations}")
            lines.append("")
        if draft.research_gaps:
            lines.extend(["## Limitations / Research Gaps", ""])
            for gap in draft.research_gaps:
                if str(gap).strip():
                    lines.append(f"- {str(gap).strip()}")
            lines.append("")
        lines.extend(["## Evidence References", ""])
        used_ids = _ordered_unique(
            evidence_id
            for section in draft.sections
            for claim in section.claims
            for evidence_id in claim.evidence_ids
        )
        for evidence_id in used_ids:
            evidence = evidence_by_id[evidence_id]
            label = f"{evidence.paper_id}, p.{evidence.page}, {evidence.block_id}"
            lines.append(f"- [{evidence_id}] {label}")
        markdown = "\n".join(lines).strip() + "\n"
        return AgentReportResult(
            status=AgentReportStatus.AVAILABLE,
            markdown=markdown,
            claim_count=sum(len(section.claims) for section in draft.sections),
            citation_count=citation_count,
            evidence_references=[evidence_by_id[item].model_dump() for item in used_ids],
        )


class AgentFinalReportSynthesizer:
    def __init__(
        self,
        provider: StructuredReportProvider,
        *,
        compiler: AgentFinalReportCompiler | None = None,
        max_attempts: int = 2,
        max_output_tokens: int = 1800,
    ) -> None:
        self.provider = provider
        self.compiler = compiler or AgentFinalReportCompiler()
        self.max_attempts = max(1, min(max_attempts, 2))
        self.max_output_tokens = max_output_tokens

    def synthesize(self, state: AgentState) -> AgentReportResult:
        input_validation = validate_report_input_state(state)
        if input_validation is not None:
            return input_validation
        evidence_by_id = build_report_evidence(state)
        system_prompt = _system_prompt()
        user_prompt = _user_prompt(state, evidence_by_id)
        usage_records: list[dict[str, Any]] = []
        request_count = 0
        last_failure = "report synthesis did not run"
        for attempt in range(1, self.max_attempts + 1):
            try:
                result = self.provider.generate_structured_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt
                    if attempt == 1
                    else _repair_prompt(user_prompt, last_failure, evidence_by_id),
                    schema_name="agent_final_report_draft_v1",
                    request_context={
                        "task_id": state.task_id,
                        "stage": "AGENT_FINAL_REPORT_SYNTHESIS"
                        if attempt == 1
                        else "REPORT_SCHEMA_REPAIR",
                        "attempt_number": attempt,
                    },
                    max_output_tokens=self.max_output_tokens,
                )
            except LLMProviderError as exc:
                return AgentReportResult(
                    status=AgentReportStatus.FAILED_PROVIDER,
                    failure_reason=f"{type(exc).__name__}: {exc}",
                    provider_request_count=getattr(exc, "api_request_count", 0),
                )
            except Exception as exc:
                return AgentReportResult(
                    status=AgentReportStatus.FAILED_PROVIDER,
                    failure_reason=f"{type(exc).__name__}: {exc}",
                )
            request_count += max(result.request_attempt_count, 1)
            usage_records.append(result.usage.model_dump())
            compiled = self.compiler.compile(result.payload, evidence_by_id=evidence_by_id)
            compiled.usage = _merge_usage(usage_records)
            compiled.provider_request_count = request_count
            if compiled.status == AgentReportStatus.AVAILABLE:
                return compiled
            last_failure = compiled.failure_reason or compiled.status.value
            if compiled.status == AgentReportStatus.FAILED_VALIDATION:
                return compiled
        return AgentReportResult(
            status=AgentReportStatus.FAILED_SCHEMA,
            failure_reason=last_failure,
            usage=_merge_usage(usage_records),
            provider_request_count=request_count,
        )


def validate_report_input_state(state: AgentState) -> AgentReportResult | None:
    if state.status != AgentStatus.COMPLETED:
        return AgentReportResult(
            status=AgentReportStatus.NOT_STARTED,
            failure_reason="agent execution is not completed",
        )
    verification = state.verification_state
    if verification is None or verification.status != VerificationStatus.PASS:
        return AgentReportResult(
            status=AgentReportStatus.NOT_STARTED,
            failure_reason="agent verification is not PASS",
        )
    if not state.evidence_state.items:
        return AgentReportResult(
            status=AgentReportStatus.REPORT_INPUT_INVALID,
            failure_reason="REPORT_INPUT_INVALID: verified Evidence State is empty",
        )
    return None


def build_report_evidence(state: AgentState) -> dict[str, AgentReportEvidence]:
    items = sorted(state.evidence_state.items.values(), key=lambda item: item.stable_key)
    return {
        f"E{idx:02d}": _report_evidence(f"E{idx:02d}", item)
        for idx, item in enumerate(items, start=1)
    }


def _report_evidence(evidence_id: str, item: EvidenceItem) -> AgentReportEvidence:
    return AgentReportEvidence(
        evidence_id=evidence_id,
        paper_id=item.paper_id,
        block_id=item.block_id,
        page=item.page,
        section=item.section,
        text=item.text_or_reference,
        stable_key=item.stable_key,
    )


def _system_prompt() -> str:
    return (
        "You write a concise user-facing research report from supplied verified evidence. "
        "Use only supplied verified evidence. Do not introduce unsupported facts. "
        "Every factual research claim must reference one or more supplied evidence IDs. "
        "If evidence does not support a conclusion, state the limitation. "
        "Return only one JSON object matching the requested draft shape."
    )


def _user_prompt(state: AgentState, evidence_by_id: dict[str, AgentReportEvidence]) -> str:
    payload = {
        "research_question": state.research_question,
        "draft_shape": {
            "title": "string",
            "summary": "string",
            "sections": [
                {
                    "title": "string",
                    "claims": [{"text": "string", "evidence_ids": ["E01"]}],
                }
            ],
            "research_gaps": ["string"],
        },
        "valid_evidence_ids": list(evidence_by_id),
        "verified_evidence": [item.model_dump() for item in evidence_by_id.values()],
        "research_gaps": state.evidence_gaps,
        "forbidden_fields": sorted(SYSTEM_DERIVED_CONTROL_FIELDS),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _repair_prompt(
    original_prompt: str,
    validation_error: str,
    evidence_by_id: dict[str, AgentReportEvidence],
) -> str:
    payload = {
        "instruction": "Return only corrected JSON. Fix JSON/schema/citation ID contract errors.",
        "validation_error": validation_error,
        "valid_evidence_ids": list(evidence_by_id),
        "original_request": json.loads(original_prompt),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _find_control_fields(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in SYSTEM_DERIVED_CONTROL_FIELDS:
                found.add(key)
            found.update(_find_control_fields(nested))
    elif isinstance(value, list):
        for item in value:
            found.update(_find_control_fields(item))
    return found


def _merge_usage(items: list[dict[str, Any]]) -> dict[str, Any]:
    input_tokens = sum(int(item.get("input_tokens") or 0) for item in items)
    output_tokens = sum(int(item.get("output_tokens") or 0) for item in items)
    total_tokens = sum(int(item.get("total_tokens") or 0) for item in items)
    costs = [
        float(item["estimated_cost_usd"])
        for item in items
        if item.get("estimated_cost_usd") is not None
    ]
    usage_sources = [str(item.get("usage_source") or "unknown") for item in items]
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(sum(costs), 8) if costs else None,
        "usage_source": usage_sources[-1] if usage_sources else "not_used",
    }


def _ordered_unique(items) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _validation_message(exc: ValidationError) -> str:
    first = exc.errors()[0] if exc.errors() else {}
    loc = ".".join(str(part) for part in first.get("loc", ()))
    msg = str(first.get("msg", exc))
    return f"{loc}: {msg}" if loc else msg


__all__ = [
    "AGENT_FINAL_REPORT_SYNTHESIS",
    "AgentFinalReportCompiler",
    "AgentFinalReportDraft",
    "AgentFinalReportStore",
    "AgentFinalReportSynthesizer",
    "AgentReportStatus",
    "AgentReportResult",
    "FINAL_REPORT_INPUT_FIELDS",
    "build_report_evidence",
    "validate_report_input_state",
]
