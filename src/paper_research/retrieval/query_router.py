from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from paper_research.evidence.claims import ClaimUnit
from paper_research.retrieval.filters import RetrievalFilter

QuestionType = Literal[
    "definition",
    "method",
    "mechanism",
    "result",
    "comparison",
    "limitation",
    "multi_paper",
    "unanswerable",
    "unknown",
]


class RetrievalProfile(BaseModel):
    name: QuestionType
    dense_weight: float = Field(ge=0)
    lexical_weight: float = Field(ge=0)
    structural_boosts: dict[str, float] = Field(default_factory=dict)
    evidence_role_filters: list[str] = Field(default_factory=list)
    block_type_filters: list[str] = Field(default_factory=list)
    section_title_boosts: dict[str, float] = Field(default_factory=dict)
    numeric_term_boost: float = 0.0
    paper_diversity_minimum: int = 1
    retrieval_k: int = 20
    candidate_pool_k: int = 40
    exclude_roles: list[str] = Field(
        default_factory=lambda: ["metadata", "citation_only", "non_evidence"]
    )


class RoutingDecision(BaseModel):
    question_type: QuestionType
    profile: RetrievalProfile
    retrieval_filter: RetrievalFilter
    target_paper_ids: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    deterministic: bool = True
    fallback_used: bool = False


PROFILES: dict[QuestionType, RetrievalProfile] = {
    "definition": RetrievalProfile(
        name="definition",
        dense_weight=0.7,
        lexical_weight=0.3,
        evidence_role_filters=["definition"],
        section_title_boosts={"introduction": 0.08},
    ),
    "method": RetrievalProfile(
        name="method",
        dense_weight=0.65,
        lexical_weight=0.35,
        evidence_role_filters=["method", "setup"],
        section_title_boosts={"method": 0.12, "approach": 0.10},
    ),
    "mechanism": RetrievalProfile(
        name="mechanism",
        dense_weight=0.65,
        lexical_weight=0.35,
        evidence_role_filters=["mechanism", "method"],
        section_title_boosts={"method": 0.08, "model": 0.08},
    ),
    "result": RetrievalProfile(
        name="result",
        dense_weight=0.55,
        lexical_weight=0.45,
        evidence_role_filters=["result", "metric"],
        numeric_term_boost=0.15,
        section_title_boosts={"experiment": 0.12, "result": 0.12},
    ),
    "comparison": RetrievalProfile(
        name="comparison",
        dense_weight=0.55,
        lexical_weight=0.45,
        evidence_role_filters=["comparison", "result"],
        numeric_term_boost=0.10,
        section_title_boosts={"comparison": 0.12, "experiment": 0.08},
    ),
    "limitation": RetrievalProfile(
        name="limitation",
        dense_weight=0.6,
        lexical_weight=0.4,
        evidence_role_filters=["limitation"],
        section_title_boosts={"limitation": 0.15, "discussion": 0.10, "conclusion": 0.08},
    ),
    "multi_paper": RetrievalProfile(
        name="multi_paper",
        dense_weight=0.6,
        lexical_weight=0.4,
        evidence_role_filters=["comparison", "result", "method"],
        paper_diversity_minimum=2,
        candidate_pool_k=60,
    ),
    "unanswerable": RetrievalProfile(
        name="unanswerable",
        dense_weight=0.5,
        lexical_weight=0.5,
        evidence_role_filters=[],
        candidate_pool_k=40,
    ),
    "unknown": RetrievalProfile(
        name="unknown",
        dense_weight=0.5,
        lexical_weight=0.5,
        evidence_role_filters=[],
    ),
}


def route_query(
    question: str,
    claims: list[ClaimUnit],
    retrieval_filter: RetrievalFilter,
) -> RoutingDecision:
    del question
    types = {claim.question_type for claim in claims}
    selected: QuestionType = "unknown"
    reasons = []
    if "unanswerable" in types:
        selected = "unanswerable"
        reasons.append("ClaimUnit declares an unanswerable obligation")
    elif "multi_paper" in types or len(retrieval_filter.paper_ids or []) > 1:
        selected = "multi_paper"
        reasons.append("Multiple target papers require candidate quotas")
    else:
        role_map: dict[str, QuestionType] = {
            "define": "definition",
            "explain_method": "method",
            "explain_mechanism": "mechanism",
            "report_result": "result",
            "compare": "comparison",
            "report_limitation": "limitation",
        }
        routed = [role_map[claim.claim_role] for claim in claims if claim.claim_role in role_map]
        if routed:
            selected = routed[0]
            reasons.append(f"First stable claim role maps to {selected}")
        else:
            reasons.append("No stable type; safe baseline profile selected")
    profile = PROFILES[selected].model_copy(deep=True)
    target_papers = list(retrieval_filter.paper_ids or [])
    if selected == "multi_paper":
        profile.paper_diversity_minimum = max(2, len(target_papers))
    return RoutingDecision(
        question_type=selected,
        profile=profile,
        retrieval_filter=retrieval_filter.model_copy(deep=True),
        target_paper_ids=target_papers,
        reasons=reasons,
        fallback_used=selected == "unknown",
    )
