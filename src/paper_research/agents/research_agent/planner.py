from __future__ import annotations

import re

from paper_research.agents.research_agent.models import ResearchPlan, Subquestion
from paper_research.agents.research_agent.state import AgentState


class RuleBasedResearchPlanner:
    """Deterministic planner used by v1 runtime and tests.

    It stores only structured decisions, not hidden reasoning.
    """

    def initial_plan(self, question: str, *, paper_ids: list[str] | None = None) -> ResearchPlan:
        comparison = re.search(r"比较|对比|分别|\bcompare\b|\bversus\b|\bvs\b", question, re.I)
        if comparison and paper_ids and 2 <= len(paper_ids) <= 6:
            # Scope belongs to the existing SQ identity and ToolAction arguments;
            # no new checkpoint fields or independent planning system are needed.
            return ResearchPlan(
                objective=question.strip(),
                subquestions=[Subquestion(
                    id=f"paper:{paper_id}",
                    question=f"Describe this paper's own method, experiments and limitations. {question}",
                ) for paper_id in dict.fromkeys(paper_ids)],
                completion_criteria=["Give each scoped paper one method retrieval opportunity; record evidence gaps."],
            )
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

    def initial_contract_plan(self, question: str, paper_ids=None):
        from paper_research.agents.research_agent.evidence_contract import (
            ContractPlan,
            EvidenceContract,
            EvidenceRequirement,
            stable_contract_id,
        )

        comparison = re.search(
            r"compare\s+(.+?)\s+(?:and|versus|vs\.?|with)\s+(.+?)[.?]?$", question, re.I
        )
        if comparison:
            a, b = comparison.groups()
            specs = [
                (f"{a} mechanism", "METHOD"),
                (f"{b} mechanism", "METHOD"),
                (f"Comparable experimental setting for {a} and {b}", "COMPARISON"),
                (f"{a} result", "RESULT"),
                (f"{b} result", "RESULT"),
                (f"Limitations and caveats of {a} and {b}", "LIMITATION"),
            ]
        else:
            # A contract describes the question, not invented generic background needs.
            clauses = [
                part.strip(" .;\n")
                for part in re.split(r"\band\b|[;；]", question, flags=re.I)
                if part.strip(" .;\n")
            ]
            specs = [(part, "FACT") for part in clauses[:6]]
        contract = EvidenceContract(
            contract_id=stable_contract_id(question),
            question=question,
            created_by="RuleBasedResearchPlanner",
            requirements=[
                EvidenceRequirement(
                    requirement_id=f"R{i}",
                    description=text,
                    requirement_type=kind,
                    source_scope=paper_ids or [],
                    min_sources=2 if kind == "COMPARISON" and len(paper_ids or []) >= 2 else 1,
                )
                for i, (text, kind) in enumerate(specs, 1)
            ],
        )
        plan = ResearchPlan(
            objective=question,
            subquestions=[
                Subquestion(id=r.requirement_id, question=r.description)
                for r in contract.requirements
            ],
            completion_criteria=["All required evidence requirements have semantic support."],
        )
        return ContractPlan(plan=plan, contract=contract)

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
