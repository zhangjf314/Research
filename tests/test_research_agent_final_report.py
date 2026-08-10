from __future__ import annotations

from typing import Any

from paper_research.agents.research_agent.final_report import (
    AgentFinalReportCompiler,
    AgentFinalReportSynthesizer,
    AgentReportStatus,
    build_report_evidence,
)
from paper_research.agents.research_agent.models import (
    AgentStatus,
    EvidenceItem,
    VerificationResult,
    VerificationStatus,
)
from paper_research.agents.research_agent.state import AgentState
from paper_research.providers.llm import ModelUsage, StructuredJSONResult


class MockReportProvider:
    def __init__(self, payloads: list[dict[str, Any]] | None = None, *, fail: bool = False) -> None:
        self.payloads = list(payloads or [])
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    def generate_structured_json(self, **kwargs) -> StructuredJSONResult:
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("connect error")
        payload = self.payloads.pop(0)
        return StructuredJSONResult(
            payload=payload,
            provider="mock",
            model="mock-report",
            usage=ModelUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            request_attempt_count=1,
        )


def _completed_state(tmp_path) -> AgentState:
    del tmp_path
    state = AgentState(research_question="What did the paper find?")
    state.status = AgentStatus.COMPLETED
    state.verification_state = VerificationResult(
        status=VerificationStatus.PASS,
        verified_claims=["Evidence supports the research question."],
        recommended_next_action="FINISH",
    )
    state.evidence_state.add(
        EvidenceItem(
            evidence_id="a",
            paper_id="p1",
            block_id="a",
            page=1,
            section="Results",
            text_or_reference="Transformer evidence a",
            discovered_by_tool="tool-1",
            discovered_at_step=1,
        ),
        ["SQ1"],
    )
    return state


def test_agent_final_report_success_uses_verified_evidence_only(tmp_path) -> None:
    state = _completed_state(tmp_path)
    provider = MockReportProvider(
        [
            {
                "title": "Research answer",
                "summary": "A concise synthesis.",
                "sections": [
                    {
                        "title": "Findings",
                        "claims": [
                            {
                                "text": "The evidence supports the finding.",
                                "evidence_ids": ["E01"],
                            }
                        ],
                    }
                ],
                "research_gaps": ["Only retrieved evidence was available."],
            }
        ]
    )

    result = AgentFinalReportSynthesizer(provider).synthesize(state)

    assert result.status == AgentReportStatus.AVAILABLE
    assert "# Research answer" in result.markdown
    assert "[E01]" in result.markdown
    assert result.claim_count == 1
    assert result.citation_count == 1
    assert state.status == AgentStatus.COMPLETED
    assert len(provider.calls) == 1
    assert "Step 1" not in provider.calls[0]["user_prompt"]
    assert "system prompt" not in provider.calls[0]["user_prompt"].lower()


def test_agent_final_report_unknown_citation_fails_without_repair(tmp_path) -> None:
    state = _completed_state(tmp_path)
    provider = MockReportProvider(
        [
            {
                "title": "Bad report",
                "summary": "Uses an unknown citation.",
                "sections": [
                    {
                        "title": "Findings",
                        "claims": [{"text": "Unsupported", "evidence_ids": ["E999"]}],
                    }
                ],
                "research_gaps": [],
            }
        ]
    )

    result = AgentFinalReportSynthesizer(provider).synthesize(state)

    assert result.status == AgentReportStatus.FAILED_VALIDATION
    assert "AGENT_REPORT_INVALID_CITATION" in str(result.failure_reason)
    assert len(provider.calls) == 1


def test_agent_final_report_claim_without_citation_fails_schema() -> None:
    evidence_by_id = {
        "E01": next(iter(build_report_evidence(_minimal_state_with_evidence()).values()))
    }
    result = AgentFinalReportCompiler().compile(
        {
            "title": "Bad report",
            "summary": "No citation.",
            "sections": [{"title": "Findings", "claims": [{"text": "Claim", "evidence_ids": []}]}],
            "research_gaps": [],
        },
        evidence_by_id=evidence_by_id,
    )

    assert result.status == AgentReportStatus.FAILED_SCHEMA


def test_agent_final_report_provider_failure_preserves_completed_execution(tmp_path) -> None:
    state = _completed_state(tmp_path)

    result = AgentFinalReportSynthesizer(MockReportProvider(fail=True)).synthesize(state)

    assert state.status == AgentStatus.COMPLETED
    assert result.status == AgentReportStatus.FAILED_PROVIDER
    assert result.markdown == ""


def test_agent_final_report_schema_failure_after_repair_attempt(tmp_path) -> None:
    state = _completed_state(tmp_path)
    provider = MockReportProvider(
        [
            {"title": "Bad", "summary": "Missing sections"},
            {"title": "Still bad", "summary": "Missing sections"},
        ]
    )

    result = AgentFinalReportSynthesizer(provider).synthesize(state)

    assert result.status == AgentReportStatus.FAILED_SCHEMA
    assert len(provider.calls) == 2


def test_agent_final_report_rejects_model_authored_control_fields(tmp_path) -> None:
    state = _completed_state(tmp_path)
    provider = MockReportProvider(
        [
            {
                "title": "Bad",
                "summary": "Bad",
                "status": "COMPLETED",
                "sections": [
                    {
                        "title": "Findings",
                        "claims": [{"text": "Claim", "evidence_ids": ["E01"]}],
                    }
                ],
            },
            {
                "title": "Bad again",
                "summary": "Bad",
                "status": "COMPLETED",
                "sections": [
                    {
                        "title": "Findings",
                        "claims": [{"text": "Claim", "evidence_ids": ["E01"]}],
                    }
                ],
            },
        ]
    )

    result = AgentFinalReportSynthesizer(provider).synthesize(state)

    assert result.status == AgentReportStatus.FAILED_SCHEMA
    assert "control fields" in str(result.failure_reason)


def test_agent_final_report_no_verified_evidence_does_not_call_provider() -> None:
    state = AgentState(research_question="empty")
    state.status = AgentStatus.COMPLETED
    state.verification_state = VerificationResult(status=VerificationStatus.PASS)
    provider = MockReportProvider([])

    result = AgentFinalReportSynthesizer(provider).synthesize(state)

    assert result.status == AgentReportStatus.REPORT_INPUT_INVALID
    assert len(provider.calls) == 0


def _minimal_state_with_evidence() -> AgentState:
    state = AgentState(research_question="q")
    state.evidence_state.add(
        item=EvidenceItem(
            evidence_id="a",
            paper_id="p1",
            block_id="a",
            page=1,
            text_or_reference="evidence a",
            discovered_by_tool="tool-1",
            discovered_at_step=1,
        ),
        target_subquestions=["SQ1"],
    )
    return state
