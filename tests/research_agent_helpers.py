from __future__ import annotations

from pathlib import Path

from paper_research.agents.research_agent.checkpoint import JsonResearchAgentCheckpointStore
from paper_research.agents.research_agent.runner import ResearchAgentRunner
from paper_research.agents.research_agent.state import AgentBudget
from paper_research.agents.research_agent.trace import ResearchAgentTraceWriter


class MockEvidenceProvider:
    def __init__(
        self,
        responses: list[list[dict]] | None = None,
        *,
        fail_first: bool = False,
    ) -> None:
        self.responses = list(responses or [])
        self.fail_first = fail_first
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, paper_ids=None, limit: int = 5) -> list[dict]:
        self.calls.append((query, limit))
        if self.fail_first and len(self.calls) == 1:
            raise RuntimeError("transient retrieval failure")
        if self.responses:
            return self.responses.pop(0)
        return []


def evidence(block: str, *, paper: str = "p1", page: int = 1, text: str = "evidence") -> dict:
    return {
        "evidence_id": block,
        "paper_id": paper,
        "page_start": page,
        "page_end": page,
        "text": f"{text} {block}",
        "retrieval_score": 0.9,
        "section_path": ["Results"],
    }


def runner(tmp_path: Path, provider: MockEvidenceProvider) -> ResearchAgentRunner:
    return ResearchAgentRunner(
        provider,
        checkpoint_store=JsonResearchAgentCheckpointStore(tmp_path / "checkpoints"),
        trace_writer=ResearchAgentTraceWriter(tmp_path / "traces"),
    )


def small_budget(**overrides) -> AgentBudget:
    values = {
        "max_steps": 12,
        "max_tool_calls": 16,
        "max_provider_requests": 4,
        "max_tokens": 40000,
        "max_cost_usd": 0.05,
        "max_no_progress_actions": 2,
    }
    values.update(overrides)
    return AgentBudget(**values)

