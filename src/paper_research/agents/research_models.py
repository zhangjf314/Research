from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class ResearchEvidence(BaseModel):
    evidence_id: str
    paper_id: str
    title: str | None = None
    section_path: list[str] = Field(default_factory=list)
    page_start: int
    page_end: int
    text: str
    retrieval_score: float | None = None
    retrieval_sources: list[str] = Field(default_factory=list)
    target_sections: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _compatibility_aliases(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        if "text" not in normalized and "quote" in normalized:
            normalized["text"] = normalized["quote"]
        if "retrieval_score" not in normalized and "score" in normalized:
            normalized["retrieval_score"] = normalized["score"]
        if "retrieval_sources" not in normalized and "source" in normalized:
            normalized["retrieval_sources"] = [str(normalized["source"])]
        return normalized

    @property
    def global_key(self) -> tuple[str, str]:
        return (self.paper_id, self.evidence_id)
