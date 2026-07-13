from datetime import UTC, datetime

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    block_id: str
    section_path: list[str]
    page_start: int
    page_end: int
    quote: str


class EvidenceBackedField(BaseModel):
    value: str | list[str] | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0, le=1)


class ExperimentCard(BaseModel):
    datasets: EvidenceBackedField = Field(default_factory=EvidenceBackedField)
    baselines: EvidenceBackedField = Field(default_factory=EvidenceBackedField)
    metrics: EvidenceBackedField = Field(default_factory=EvidenceBackedField)
    hyperparameters: EvidenceBackedField = Field(default_factory=EvidenceBackedField)
    hardware: EvidenceBackedField = Field(default_factory=EvidenceBackedField)
    software: EvidenceBackedField = Field(default_factory=EvidenceBackedField)
    training_details: EvidenceBackedField = Field(default_factory=EvidenceBackedField)
    ablation_settings: EvidenceBackedField = Field(default_factory=EvidenceBackedField)
    code_url: EvidenceBackedField = Field(default_factory=EvidenceBackedField)
    data_url: EvidenceBackedField = Field(default_factory=EvidenceBackedField)


class PaperAnalysis(BaseModel):
    paper_id: str
    title: str | None = None
    research_background: EvidenceBackedField
    research_problem: EvidenceBackedField
    main_contributions: EvidenceBackedField
    method_summary: EvidenceBackedField
    experiment_summary: EvidenceBackedField
    main_results: EvidenceBackedField
    limitations: EvidenceBackedField
    future_work: EvidenceBackedField
    experiment_card: ExperimentCard
    analyzer: str = "section-aware-extractive-v1"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
