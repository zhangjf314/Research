import json
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from paper_research.chunking.types import Chunk
from paper_research.config import get_settings
from paper_research.generation.qa_service import Answer, QAService
from paper_research.indexing.registry import IndexRegistry
from paper_research.indexing.vector_store import QdrantVectorStore
from paper_research.providers.factory import ProviderConfigurationError, build_provider_bundle
from paper_research.retrieval.context_builder import ContextBuilder, ContextItem
from paper_research.retrieval.dense import DenseRetriever
from paper_research.retrieval.filters import RetrievalFilter
from paper_research.retrieval.hybrid import HybridRetrievalResult, HybridRetriever
from paper_research.retrieval.sparse import BM25Retriever
from paper_research.retrieval.trace import JsonlTraceRepository, RetrievalTrace

router = APIRouter()


class QARequest(BaseModel):
    question: str = Field(min_length=1)
    paper_ids: list[uuid.UUID] | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class HybridRetrievalRequest(BaseModel):
    query: str = Field(min_length=1)
    filters: RetrievalFilter = Field(default_factory=RetrievalFilter)
    recall_k: int = Field(default=20, ge=1, le=100)
    top_k: int = Field(default=5, ge=1, le=20)


class HybridRetrievalResponse(BaseModel):
    context: list[ContextItem]
    trace: RetrievalTrace


def _run_hybrid(payload: HybridRetrievalRequest) -> HybridRetrievalResult:
    settings = get_settings()
    chunks = _load_chunks(
        settings.parsed_papers_dir,
        payload.filters.paper_ids,
        settings.index_version,
    )
    if not chunks:
        raise HTTPException(status_code=409, detail="no indexed chunk files found")
    providers = build_provider_bundle(settings)
    store = QdrantVectorStore(
        QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key),
        IndexRegistry(settings.data_dir / "index_registry.json").resolve(
            settings.active_collection
        ),
        settings.embedding_dimensions,
    )
    return HybridRetriever(
        DenseRetriever(providers.embedding, store),
        BM25Retriever(chunks),
        providers.reranker,
        ContextBuilder(include_neighbors=True),
        JsonlTraceRepository(settings.retrieval_trace_path),
        provider_metadata=settings.provider_metadata,
    ).retrieve(
        payload.query,
        payload.filters,
        recall_k=payload.recall_k,
        top_k=payload.top_k,
    )


@router.post("/qa", response_model=Answer)
def ask_question(payload: QARequest) -> Answer:
    settings = get_settings()
    started = time.perf_counter()
    try:
        filters = RetrievalFilter(
            paper_ids=[str(paper_id) for paper_id in payload.paper_ids]
            if payload.paper_ids
            else None
        )
        result = _run_hybrid(
            HybridRetrievalRequest(
                query=payload.question,
                filters=filters,
                recall_k=settings.retrieval_recall_k,
                top_k=payload.top_k,
            )
        )
        providers = build_provider_bundle(settings)
        return QAService(
            llm=providers.llm,
            prompt_version=settings.prompt_version,
        ).answer_from_context(
            payload.question,
            result.context,
            retrieval_latency_ms=result.trace.retrieval_latency_ms,
            rerank_latency_ms=result.trace.rerank_latency_ms,
            context_build_latency_ms=result.trace.context_build_latency_ms,
            total_started=started,
        )
    except ProviderConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"claim QA unavailable: {type(exc).__name__}"
        ) from exc


@router.post("/retrieve", response_model=HybridRetrievalResponse)
def hybrid_retrieve(payload: HybridRetrievalRequest) -> HybridRetrievalResponse:
    try:
        result = _run_hybrid(payload)
        return HybridRetrievalResponse(context=result.context, trace=result.trace)
    except ProviderConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"hybrid retrieval unavailable: {type(exc).__name__}"
        ) from exc


def _load_chunks(root: Path, paper_ids: list[str] | None, index_version: str) -> list[Chunk]:
    filename = f"paper_chunks.{index_version}.jsonl"
    paths = (
        [root / paper_id / filename for paper_id in paper_ids]
        if paper_ids
        else list(root.glob(f"*/{filename}"))
    )
    # Compatibility with RC1 artifacts before versioned chunk files existed.
    if not any(path.exists() for path in paths):
        paths = (
            [root / paper_id / "paper_chunks.jsonl" for paper_id in paper_ids]
            if paper_ids
            else list(root.glob("*/paper_chunks.jsonl"))
        )
    chunks: list[Chunk] = []
    for path in paths:
        if path.exists():
            chunks.extend(
                Chunk.model_validate(json.loads(line))
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
    return chunks
