import json
from pathlib import Path

from paper_research.agents.deep_research_graph import DeepResearchGraph
from paper_research.agents.providers import ArtifactLocalResearchProvider
from paper_research.agents.state import ResearchBudget


def main() -> None:
    graph = DeepResearchGraph(
        ArtifactLocalResearchProvider(Path("data/reports/parsing-audit"))
    )
    state = graph.run(
        "long-context large language models: methods, experimental results, and limitations",
        budget=ResearchBudget(max_iterations=3, max_evidence_items=30),
    )
    output_dir = Path("data/reports/deep-research-audit")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "research_report.md").write_text(state["draft_report"], encoding="utf-8")
    (output_dir / "research_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    valid = sum(item["valid"] for item in state["citation_results"])
    audit = [
        "# Deep Research Workflow Audit",
        "",
        f"- Status: {state['status']}",
        f"- Stop reason: {state['stop_reason']}",
        f"- Sub-questions: {len(state['sub_questions'])}",
        f"- Evidence items: {len(state['local_evidence'])}",
        f"- Evidence gaps: {len(state['evidence_gaps'])}",
        f"- Valid citations: {valid}/{len(state['citation_results'])}",
        f"- Estimated tokens: {state['estimated_tokens']}",
        f"- Iterations: {state['iteration_count']}",
        f"- Node path: {' -> '.join(state['node_history'])}",
        "",
    ]
    (output_dir / "audit_report.md").write_text("\n".join(audit), encoding="utf-8")


if __name__ == "__main__":
    main()
