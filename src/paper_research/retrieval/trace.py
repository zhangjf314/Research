import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from paper_research.retrieval.context_builder import ContextItem


class TraceResult(BaseModel):
    chunk_id: str
    paper_id: str
    score: float
    rank: int
    dense_rank: int | None = None
    sparse_rank: int | None = None


class RetrievalTrace(BaseModel):
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    filters: dict[str, object]
    dense_results: list[TraceResult]
    sparse_results: list[TraceResult]
    fusion_results: list[TraceResult]
    rerank_results: list[TraceResult]
    final_context: list[ContextItem]
    latency_ms: float
    retrieval_latency_ms: float = 0
    rerank_latency_ms: float = 0
    context_build_latency_ms: float = 0
    embedding_provider: str = "unknown"
    embedding_model: str = "unknown"
    embedding_revision: str = "unknown"
    embedding_dimension: int = 0
    rerank_provider: str = "unknown"
    rerank_model: str = "unknown"
    llm_provider: str = "unknown"
    llm_model: str = "unknown"
    prompt_version: str = "unknown"
    index_version: str = "unknown"
    dataset_version: str = "unknown"
    collection: str = "unknown"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class JsonlTraceRepository:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, trace: RetrievalTrace) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(trace.model_dump(mode="json"), ensure_ascii=False) + "\n")
