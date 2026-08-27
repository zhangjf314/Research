from fastapi.testclient import TestClient

from paper_research.agents.research_agent.models import ToolAction
from paper_research.agents.research_agent.state import AgentState
from paper_research.agents.research_agent.tools import ResearchAgentToolRegistry
from paper_research.api.routes.research import ResearchAgentRequest
from paper_research.main import create_app


def test_library_ui_exposes_indexed_selection_direct_qa_and_error_states() -> None:
    html = TestClient(create_app()).get("/api/v1/ui/library").text

    assert "Direct QA" in html
    assert "paperResearchSelectedPaperIds" in html
    assert "Select at least one indexed paper" in html
    assert "QA running..." in html
    assert "Direct QA failed:" in html
    assert "fetch('/api/v1/qa'" in html
    assert "paper_ids:paperIds" in html
    assert "Canonical/block provenance" in html
    assert "Indexing paper..." in html


def test_research_ui_reuses_selected_paper_scope_for_workflow_and_agent() -> None:
    html = TestClient(create_app()).get("/api/v1/ui/research").text

    assert "paperResearchSelectedPaperIds" in html
    assert "paper_ids:selectedPaperIds()" in html
    assert "Choose papers in the Library" in html


def test_agent_request_and_retrieval_tool_preserve_selected_paper_scope() -> None:
    payload = ResearchAgentRequest(query="Explain the result", paper_ids=["paper-a", "paper-b"])
    assert payload.paper_ids == ["paper-a", "paper-b"]

    received: list[list[str] | None] = []

    class Provider:
        def search(self, query: str, paper_ids: list[str] | None, limit: int) -> list[dict]:
            received.append(paper_ids)
            return []

    state = AgentState(research_question="Explain the result", paper_ids=payload.paper_ids)
    action = ToolAction(
        action="retrieve_evidence",
        arguments={"query": "Explain the result"},
        target_subquestion_ids=["SQ1"],
        decision_reason="scope test",
    )
    ResearchAgentToolRegistry(Provider()).execute(state, action, "scope-test")

    assert received == [["paper-a", "paper-b"]]
