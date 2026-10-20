"""Explicitly synthetic controls through the real frozen pipeline with fake transports.

The corpus, BM25, fusion, models/config and ContextBuilder are unchanged. Dense
responses and reranker HTTP responses are fixtures, never scientific quality data.
"""

import json
from types import SimpleNamespace

import httpx

from paper_research.agents.research_agent.evidence_contract import (
    ContractPlan,
    EvidenceAssessment,
    EvidenceContract,
    EvidenceRequirement,
    stable_contract_id,
)
from paper_research.agents.research_agent.models import ResearchPlan, Subquestion
from paper_research.retrieval.frozen_backend import (
    ROOT,
    load_frozen_retrieval_backend,
    read_manifest,
)

CASES = {
    "M1": {
        "description": "initial retrieval sufficient",
        "requirements": 1,
        "rounds": [[0]],
        "stop": "CONTRACT_SATISFIED",
        "replans": 0,
    },
    "M2": {
        "description": "missing evidence → targeted search",
        "requirements": 2,
        "rounds": [[0], [1]],
        "stop": "CONTRACT_SATISFIED",
        "replans": 1,
    },
    "M3": {
        "description": "retrieval failure → reformulation",
        "requirements": 1,
        "rounds": [[], [1]],
        "stop": "CONTRACT_SATISFIED",
        "replans": 1,
    },
    "M4": {
        "description": "only unsatisfied multipart requirement searched",
        "requirements": 2,
        "rounds": [[0], [1]],
        "stop": "CONTRACT_SATISFIED",
        "replans": 1,
    },
    "M5": {
        "description": "duplicate evidence gives zero gain",
        "requirements": 1,
        "rounds": [[0], [0], [0]],
        "stop": "NO_INFORMATION_GAIN",
        "replans": 2,
    },
    "M6": {
        "description": "two targeted calls without gain stop",
        "requirements": 1,
        "rounds": [[0], [0], [0]],
        "stop": "NO_INFORMATION_GAIN",
        "replans": 2,
    },
    "M7": {
        "description": "contradiction → targeted resolution attempt",
        "requirements": 1,
        "rounds": [[0, 1], [2]],
        "stop": "CONTRACT_SATISFIED",
        "replans": 1,
    },
    "M8": {
        "description": "hard tool/step budget stops unsatisfied contract",
        "requirements": 2,
        "rounds": [[0]],
        "stop": "BUDGET_EXHAUSTED",
        "replans": 0,
    },
}


class MechanismPlanner:
    def __init__(self, case):
        self.case = case

    def initial_contract_plan(self, question, paper_ids=None):
        reqs = [
            EvidenceRequirement(
                requirement_id=f"R{i}",
                description=f"zzrequirement{i}",
                source_scope=paper_ids or [],
            )
            for i in range(1, CASES[self.case]["requirements"] + 1)
        ]
        return ContractPlan(
            contract=EvidenceContract(
                contract_id=stable_contract_id(question),
                question=question,
                requirements=reqs,
                created_by="SYNTHETIC_MECHANISM_TEST",
            ),
            plan=ResearchPlan(
                objective=question,
                subquestions=[
                    Subquestion(id=r.requirement_id, question=r.description) for r in reqs
                ],
                completion_criteria=["synthetic fixture requirements supported"],
            ),
        )


class MechanismQdrant:
    def __init__(self, case, selected):
        self.case, self.selected, self.round = case, selected, 0
        self.calls = []

    def query_points(self, **kwargs):
        expected = read_manifest()["dense"]["collection"]
        # Frozen DenseRetriever oversamples fivefold when a filter object exists,
        # then returns at most 30 to RRF. This is the unchanged R2 implementation.
        assert kwargs["collection_name"] == expected and kwargs["limit"] == 150
        index = min(self.round, len(CASES[self.case]["rounds"]) - 1)
        rows = CASES[self.case]["rounds"][index]
        self.round += 1
        self.calls.append({"collection": kwargs["collection_name"], "limit": kwargs["limit"]})
        return SimpleNamespace(
            points=[
                SimpleNamespace(payload=self.selected[i].model_dump(), score=1 - j * 0.01)
                for j, i in enumerate(rows)
            ]
        )

    def close(self):
        pass


def fixture_backend(case):
    manifest = read_manifest()
    from paper_research.chunking.types import Chunk

    chunks = [
        Chunk.model_validate_json(line)
        for line in (ROOT / manifest["corpus"]["path"]).read_text(encoding="utf-8").splitlines()
    ]
    selected = []
    seen = set()
    for chunk in chunks:
        blocks = {(chunk.paper_id, b) for b in chunk.block_ids}
        if blocks and not blocks & seen:
            selected.append(chunk)
            seen |= blocks
        if len(selected) == 3:
            break
    qdrant = MechanismQdrant(case, selected)

    def embedding_response(request):
        data = json.loads(request.content)
        assert data["model"] == manifest["dense"]["model"]
        return httpx.Response(
            200,
            json={
                "data": [{"index": 0, "embedding": [0.0] * 1024}],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    def rerank_response(request):
        data = json.loads(request.content)
        assert data["model"] == manifest["reranker"]["model"]
        assert data["top_n"] == len(data["documents"]) <= 30
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": i, "relevance_score": 1 - i * 0.01} for i in range(data["top_n"])
                ]
            },
        )

    backend = load_frozen_retrieval_backend(
        client=qdrant,
        verify_live=False,
        embedding_client=httpx.Client(transport=httpx.MockTransport(embedding_response)),
        reranker_client=httpx.Client(transport=httpx.MockTransport(rerank_response)),
    )
    return backend, MechanismVerifier(case, qdrant, selected)


class MechanismVerifier:
    def __init__(self, case, qdrant, selected):
        self.case, self.qdrant, self.selected = case, qdrant, selected

    def assess(self, state, requirement_ids, evidence_ids, runner):
        assessments = []
        for rid in requirement_ids:
            for eid in evidence_ids:
                verdict = "SUPPORTS"
                if self.case in {"M2", "M4", "M8"}:
                    verdict = (
                        "SUPPORTS"
                        if eid == self.selected[int(rid[1:]) - 1].chunk_id
                        else "IRRELEVANT"
                    )
                elif self.case == "M5":
                    verdict = "PARTIAL"
                elif self.case == "M6":
                    verdict = "IRRELEVANT"
                elif self.case == "M7" and eid == self.selected[1].chunk_id:
                    verdict = "CONTRADICTS" if self.qdrant.round == 1 else "IRRELEVANT"
                assessments.append(
                    EvidenceAssessment(
                        requirement_id=rid,
                        evidence_id=eid,
                        verdict=verdict,
                        confidence=1,
                        rationale="SYNTHETIC_MECHANISM_TEST: registered verdict; not corpus truth",
                    )
                )
        return assessments


def validate_case(case, state):
    expected = CASES[case]
    assert state.stop_reason.value == expected["stop"]
    assert len(state.replan_history) == expected["replans"]
    if case in {"M2", "M4"}:
        assert state.retrieval_actions[1].requirement_ids == ["R2"]
        assert state.retrieval_actions[1].action_type.value == "TARGETED_SEARCH"
    if case == "M3":
        assert state.retrieval_actions[1].action_type.value == "REFORMULATE_QUERY"
    if case in {"M5", "M6"}:
        assert all(not rp.effective_replan for rp in state.replan_history)
        assert all(g.new_evidence_count == 0 for g in state.information_gain_history[1:])
    if case == "M7":
        assert any(
            g["gap_type"] == "CONTRADICTORY_EVIDENCE" for g in state.tool_history[0]["gap_after"]
        )
        assert state.retrieval_actions[1].reason_code.value == "CONTRADICTION"
    assert (
        state.step_count <= state.budget.max_steps
        and state.tool_call_count <= state.budget.max_tool_calls
    )
