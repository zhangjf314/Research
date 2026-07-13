from typing import Literal

from pydantic import BaseModel, Field


class EvaluationItem(BaseModel):
    id: str
    question: str
    question_type: str
    relevant_paper_ids: list[str] = Field(min_length=1)
    relevant_block_ids: list[str] = Field(default_factory=list)
    relevant_pages: list[int] = Field(default_factory=list)
    expected_answer_points: list[str] = Field(default_factory=list)
    annotation_status: Literal["silver", "human_reviewed", "rejected"] = "silver"
    reviewer: str | None = None
    notes: str | None = None


class RetrievalPrediction(BaseModel):
    item_id: str
    ranked_paper_ids: list[str]
    ranked_block_ids: list[str] = Field(default_factory=list)
