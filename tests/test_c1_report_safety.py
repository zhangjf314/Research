"""S3 bounded supplements. HTTP/PDF/restart qualification lives in D4's harness."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from paper_research.agents.research_agent import final_report as report
from paper_research.agents.research_agent.checkpoint import (
    JsonResearchAgentCheckpointStore, checkpoint_kind, require_current_resume,
)
from paper_research.agents.research_agent.models import EvidenceItem, Subquestion
from paper_research.agents.research_agent.state import AgentState
from paper_research.agents.research_agent.verifier import (
    DeterministicResearchVerifier, REPORT_SAFETY_NAMESPACE, REPORT_SAFETY_RECEIPT,
    ResearchSafetyError,
)
from paper_research.api.routes import research


def state_current(task_id="bounded"):
    return AgentState(task_id=task_id, research_question="synthetic unrelated evidence",
        subquestions=[Subquestion(id="SQ1", question="synthetic claim")],
        evidence_provenance={REPORT_SAFETY_NAMESPACE: dict(REPORT_SAFETY_RECEIPT)})


def candidate(**changes):
    values = dict(evidence_id="chunk-token", paper_id="doc-token", block_id="chunk-token",
        page=1, text_or_reference="Unrelated or contradictory artificial text",
        discovered_by_tool="retrieve_evidence", discovered_at_step=1)
    return EvidenceItem(**{**values, **changes})


def test_presence_never_grants_semantic_support():
    state = state_current()
    state.evidence_state.add(candidate(), ["SQ1"])
    state.evidence_state.contradictions = ["unassessed artificial contradiction"]
    result = DeterministicResearchVerifier().verify(state).model_dump(mode="json")
    assert result == {"status": "PARTIAL", "verified_claims": [], "unsupported_claims": [],
        "unresolved_subquestions": ["SQ1"], "evidence_gaps": ["SEMANTIC_SUPPORT_NOT_ASSESSED"],
        "contradictions": [], "recommended_next_action": "FINISH"}
    assert all(item.verification_status == "CANDIDATE" for item in state.evidence_state.items.values())


def test_report_direct_entrypoints_always_closed(tmp_path, monkeypatch):
    state = state_current()
    state.status = "COMPLETED"
    state.evidence_state.add(candidate(verification_status="VERIFIED", supports=["SQ1"]), ["SQ1"])
    provider = SimpleNamespace(generate_structured_json=lambda **kw: pytest.fail("report provider called"))
    draft = {"title": "Artificial", "summary": "Unqualified assertion", "sections": [
        {"title": "Result", "claims": [{"text": "Unqualified claim", "evidence_ids": ["E01"]}]}]}
    evidence = {"E01": report.AgentReportEvidence(evidence_id="E01", paper_id="doc-token",
        block_id="fake", page=1, text="unrelated", stable_key="fake")}
    for payload in [draft, {}, {"status": "VERIFIED"}, {**draft, "summary": "E99"}]:
        assert report.AgentFinalReportCompiler().compile(payload, evidence_by_id=evidence) == report.suppressed_report()
    assert report.validate_report_input_state(state) == report.suppressed_report()
    assert report.build_report_evidence(state) == {}
    assert report.AgentFinalReportSynthesizer(provider).synthesize(state) == report.suppressed_report()
    store = report.AgentFinalReportStore(tmp_path / "reports")
    store.root.mkdir()
    cache = store.root / "bounded-final-report.json"
    available = report.AgentReportResult(status="AVAILABLE", markdown="Synthetic unsafe cache")
    cache.write_text(available.model_dump_json(), encoding="utf-8")
    original = cache.read_bytes()
    original_read = Path.read_text
    def forbidden_cache_read(path, *args, **kwargs):
        assert path != cache, "cached content read"
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", forbidden_cache_read)
    assert store.load("bounded") == report.suppressed_report()
    store.save("bounded", report.suppressed_report())
    for result in [available, report.AgentReportResult(status="NOT_STARTED"),
                   report.suppressed_report().model_copy(update={"usage": {"total_tokens": 1}})]:
        with pytest.raises(ResearchSafetyError, match="REPORT_SAFETY_WRITE_FORBIDDEN"):
            store.save("bounded", result)
    assert cache.read_bytes() == original


@pytest.mark.parametrize("mutate", [
    lambda s: s.update(status="COMPLETED"),
    lambda s: s.update(verified_claims=["unsupported asserted fact"]),
    lambda s: s.update(stop_reason="SUCCESS"),
    lambda s: s.update(verification_state={"status": "PASS"}),
    lambda s: s.update(evidence_assessments=[{}]),
    lambda s: s.update(requirement_coverage={"R1": {}}),
    lambda s: s.update(evidence_contract={"fake": True}),
    lambda s: s.update(step_count=True),
    lambda s: s.update(tool_call_count=-1),
    lambda s: s.update(completed_tool_call_ids=["same", "same"]),
])
def test_current_trust_state_rejected(mutate):
    raw = state_current().model_dump(mode="json")
    mutate(raw)
    with pytest.raises(ResearchSafetyError, match="RESEARCH_TRUST_STATE_INVALID"):
        checkpoint_kind(raw)


@pytest.mark.parametrize("mutate", [
    lambda s: s.update(schema_version=2),
    lambda s: s["evidence_provenance"][REPORT_SAFETY_NAMESPACE].update(schema_version=True),
    lambda s: s["evidence_provenance"][REPORT_SAFETY_NAMESPACE].update(schema_version=2),
    lambda s: s["evidence_provenance"][REPORT_SAFETY_NAMESPACE].update(qualification_id="fake"),
    lambda s: s["evidence_provenance"].update(p1_agent_report_safety_v2={}),
    lambda s: s["evidence_provenance"].update(p1_readonly_retrieval_bridge_v1={}),
])
def test_unknown_schema_has_no_legacy_fallback(mutate):
    raw = state_current().model_dump(mode="json")
    mutate(raw)
    with pytest.raises(ResearchSafetyError, match="RESEARCH_TRUST_SCHEMA_INVALID"):
        checkpoint_kind(raw)


@pytest.mark.parametrize("status", ["RUNNING", "PAUSED", "COMPLETED", "PARTIAL", "FAILED"])
def test_legacy_safe_view_no_resume(status, tmp_path):
    state = AgentState(task_id="legacy", research_question="synthetic", status=status,
                       subquestions=[Subquestion(id="SQ1", question="synthetic", status="RESOLVED")])
    state.evidence_state.add(candidate(verification_status="VERIFIED", supports=["unsafe"]), ["SQ1"])
    with pytest.raises(ResearchSafetyError, match="LEGACY_CHECKPOINT_RESUME_FORBIDDEN"):
        require_current_resume(state)
    result = research._agent_response(state, report.AgentReportResult(status="AVAILABLE", markdown="unsafe")).model_dump(mode="json")
    assert result["status"] == "PARTIAL" and result["stop_reason"] == "UNRESOLVABLE_GAP"
    assert result["verification_state"] is None and not result["resolved_subquestions"]
    assert result["unresolved_subquestions"] == ["SQ1"] and result["subquestions"][0]["status"] == "OPEN"
    assert result["report_markdown"] == "" and result["observations"] == result["tool_history"] == []
    assert result["evidence_trust"]["historical_agent_status"] == status
    assert state.status == status and state.subquestions[0].status == "RESOLVED"
    store = JsonResearchAgentCheckpointStore(tmp_path / "missing-root")
    with pytest.raises(KeyError): store.load("absent")
    assert not store.root.exists()
    with pytest.raises(ResearchSafetyError, match="LEGACY_CHECKPOINT_WRITE_FORBIDDEN"):
        store.save(state, "ATTEMPT")
    assert not store.root.exists()


def test_canonical_same_block_different_chunks_and_source_domains(tmp_path):
    import importlib.util
    helper_path = Path(__file__).with_name("test_readonly_evidence_contract.py")
    spec = importlib.util.spec_from_file_location("d4_sources", helper_path)
    helpers = importlib.util.module_from_spec(spec); spec.loader.exec_module(helpers)
    from paper_research.generation.evidence_contract_adapter import build_readonly_evidence_contract
    from paper_research.chunking.types import Chunk
    chunks = [Chunk(chunk_id=c, paper_id=helpers.PAPER, block_ids=["b1"], section_path=["Synthetic"],
        block_type="paragraph", page_start=1, page_end=1, block_page_map={"b1": 1},
        chunk_text="alpha synthetic", token_count=3) for c in ["c1", "c2"]]
    context = helpers.write_paper(tmp_path, chunks=chunks)
    other = [c.model_copy(update={"paper_id": helpers.OTHER}) for c in chunks]
    context += helpers.write_paper(tmp_path, helpers.OTHER, chunks=other)
    trace = helpers.trace_for(context, scope=[helpers.PAPER, helpers.OTHER])
    sidecar = build_readonly_evidence_contract("synthetic", context, trace, parsed_root=tmp_path)
    assert len(sidecar.evidence) == 4
    assert len({e.item.evidence_id for e in sidecar.evidence}) == 4
    for e in sidecar.evidence:
        assert e.item.verification_status == "CANDIDATE"
        assert e.provenance.sources[0].paper_id == e.item.paper_id
    # C1 deliberately does not repair the legacy stable key or duplicate-SQ bug.
    state = state_current()
    first = candidate(); second = candidate(chunk_id="different", evidence_id="different")
    assert state.evidence_state.add(first, ["SQ1"])
    assert not state.evidence_state.add(second, ["SQ2"])
    assert not state.evidence_state.evidence_for("SQ2")
    assert report.build_report_evidence(state) == {}
    assert research._agent_response(state).evidence_trust["canonical_candidate_injection"] is False
