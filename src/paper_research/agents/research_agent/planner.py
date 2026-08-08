from __future__ import annotations

import re

from paper_research.agents.research_agent.models import ResearchPlan, Subquestion
from paper_research.agents.research_agent.state import AgentState


class RuleBasedResearchPlanner:
    """Deterministic planner used by v1 runtime and tests.

    It stores only structured decisions, not hidden reasoning.
    """

    def initial_plan(self, question: str) -> ResearchPlan:
        clauses = [
            item.strip(" .;:\n\t")
            for item in re.split(r"\band\b|[;；]", question, flags=re.I)
            if item.strip(" .;:\n\t")
        ]
        if len(clauses) < 2:
            clauses = [
                f"Find background evidence for: {question}",
                f"Find method or experiment evidence for: {question}",
            ]
        clauses = clauses[:6]
        return ResearchPlan(
            objective=question.strip(),
            subquestions=[
                Subquestion(id=f"SQ{index}", question=clause)
                for index, clause in enumerate(clauses, start=1)
            ],
            completion_criteria=[
                "Each open subquestion has verified evidence or an explicit insufficiency.",
                "No final claim cites evidence outside the evidence state.",
            ],
        )

    def replan(self, state: AgentState, reason: str) -> ResearchPlan:
        existing = list(state.subquestions)
        open_ids = set(state.unresolved_subquestions)
        additions: list[Subquestion] = []
        if "contradiction" in reason.lower():
            additions.append(
                Subquestion(
                    id=f"SQ{len(existing) + 1}",
                    question="Inspect additional evidence to resolve the contradiction",
                )
            )
        elif open_ids:
            for subquestion in existing:
                if subquestion.id in open_ids:
                    additions.append(
                        Subquestion(
                            id=f"SQ{len(existing) + len(additions) + 1}",
                            question=f"Find missing evidence for {subquestion.question}",
                        )
                    )
                    break
        return ResearchPlan(
            objective=state.research_question,
            subquestions=[*existing, *additions][:6],
            completion_criteria=state.current_plan.completion_criteria
            if state.current_plan
            else ["Resolve required subquestions with evidence."],
        )

