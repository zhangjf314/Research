from paper_research.agents.deep_research_graph import DeepResearchGraph
from paper_research.agents.providers import ExternalResearchProvider, LocalResearchProvider
from paper_research.agents.state import ResearchBudget


class FakeLocalProvider(LocalResearchProvider):
    def __init__(self, return_evidence: bool = True) -> None:
        self.return_evidence = return_evidence

    def search(self, query: str, paper_ids: list[str] | None, limit: int = 5) -> list[dict]:
        if not self.return_evidence:
            return []
        return [
            {
                "evidence_id": f"e-{abs(hash(query))}",
                "paper_id": "paper-1",
                "section_path": ["Method"],
                "page_start": 3,
                "page_end": 3,
                "quote": f"Evidence answering {query}",
                "score": 2.0,
                "source": "local",
            }
        ]


class FakeExternalProvider(ExternalResearchProvider):
    def search(self, query: str, limit: int = 10) -> list[dict]:
        return [
            {
                "source": "fake",
                "source_id": "candidate-1",
                "title": "External Candidate",
                "abstract": "Potential missing evidence",
                "pdf_url": "https://example.test/paper.pdf",
                "source_url": "https://example.test/paper",
            }
        ]


def test_graph_generates_plan_report_and_valid_citations() -> None:
    result = DeepResearchGraph(FakeLocalProvider()).run("How do long-context models work?")

    assert result["status"] == "COMPLETED"
    assert result["evidence_gaps"] == []
    assert len(result["sub_questions"]) == 4
    assert "# 深度研究报告" in result["draft_report"]
    assert result["citation_results"]
    assert all(item["valid"] for item in result["citation_results"])
    assert result["node_history"] == [
        "understand",
        "plan",
        "local_search",
        "assess",
        "synthesize",
        "report",
        "validate",
    ]


def test_graph_uses_external_search_for_evidence_gaps() -> None:
    result = DeepResearchGraph(
        FakeLocalProvider(return_evidence=False), FakeExternalProvider()
    ).run("What evidence is missing?")

    assert result["candidate_papers"][0]["source_id"] == "candidate-1"
    assert result["external_search_count"] == 1
    assert "external_search" in result["node_history"]
    assert result["evidence_gaps"]


def test_budget_stops_before_external_search() -> None:
    result = DeepResearchGraph(
        FakeLocalProvider(return_evidence=False), FakeExternalProvider()
    ).run("Budgeted research", budget=ResearchBudget(max_iterations=1))

    assert result["stop_reason"] == "max_iterations"
    assert "external_search" not in result["node_history"]
    assert result["status"] == "COMPLETED"


def test_external_candidate_can_be_imported_and_researched_again() -> None:
    local = FakeLocalProvider(return_evidence=False)

    def importer(_: dict) -> str:
        local.return_evidence = True
        return "imported-paper"

    result = DeepResearchGraph(local, FakeExternalProvider(), importer).run(
        "Research with automatic import",
        budget=ResearchBudget(max_iterations=3, max_external_searches=2),
    )

    assert result["iteration_count"] == 2
    assert result["selected_papers"][0]["paper_id"] == "imported-paper"
    assert result["evidence_gaps"] == []
    assert result["status"] == "COMPLETED"
