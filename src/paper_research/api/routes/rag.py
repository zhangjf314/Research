import json
import time
import traceback
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
from paper_research.providers.factory import (
    ProviderConfigurationError,
    build_embedding_provider,
    build_provider_bundle,
    build_reranker,
)
from paper_research.providers.llm import LLMProviderError
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
    sample_id: str | None = None
    run_id: str | None = None


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
    embedding = build_embedding_provider(settings)
    reranker = build_reranker(settings)
    store = QdrantVectorStore(
        QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key),
        IndexRegistry(settings.data_dir / "index_registry.json").resolve(
            settings.active_collection
        ),
        settings.embedding_dimensions,
    )
    return HybridRetriever(
        DenseRetriever(embedding, store),
        BM25Retriever(chunks),
        reranker,
        ContextBuilder(
            include_neighbors=True,
            max_tokens=settings.qa_context_token_budget,
        ),
        JsonlTraceRepository(settings.retrieval_trace_path),
        provider_metadata=settings.provider_metadata,
        rerank_input_k=settings.rerank_input_k,
        rerank_output_k=settings.rerank_output_k,
    ).retrieve(
        payload.query,
        payload.filters,
        recall_k=(
            max(payload.recall_k, settings.rerank_input_k)
            if settings.rerank_enabled
            else payload.recall_k
        ),
        top_k=payload.top_k,
        retrieval_scope="paper" if payload.filters.paper_ids else "global",
    )


@router.post("/qa", response_model=Answer)
def ask_question(payload: QARequest) -> Answer:
    settings = get_settings()
    started = time.perf_counter()
    run_id = payload.run_id or f"qa-{uuid.uuid4().hex[:12]}"
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
            audit_metadata={
                "sample_id": payload.sample_id,
                "run_id": run_id,
                "request_id": run_id,
            },
        )
    except ProviderConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMProviderError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "code": exc.error_code,
                "message": str(exc),
                "stage": exc.stage,
                "api_request_count": exc.api_request_count,
                "retry_reasons": exc.retry_reasons,
                "rate_limit_events": exc.rate_limit_events,
                "response_audit_path": exc.response_audit_path,
            },
        ) from exc
    except ValueError as exc:
        message = str(exc)
        if "unsupported production QA prompt version" in message:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "CLAIM_QA_CONFIGURATION_ERROR",
                    "message": "The configured QA prompt version is unsupported.",
                    "stage": "LLM_REQUEST_BUILD",
                },
            ) from exc
        raise HTTPException(
            status_code=500,
            detail={
                "code": "CLAIM_QA_INTERNAL_ERROR",
                "message": "Claim QA failed with an internal validation error.",
                "stage": "UNKNOWN",
            },
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        audit_path = _persist_qa_exception_audit(exc, payload.sample_id, run_id)
        raise HTTPException(
            status_code=503,
            detail={
                "code": "CLAIM_QA_UNEXPECTED_ERROR",
                "message": f"claim QA unavailable: {type(exc).__name__}",
                "stage": "API_QA_UNEXPECTED",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "exception_audit_path": str(audit_path),
            },
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
            block_pages = _load_block_pages(path.parent)
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                chunk = Chunk.model_validate(json.loads(line))
                chunks.append(_with_block_page_map(chunk, block_pages))
    return chunks


def _load_block_pages(paper_dir: Path) -> dict[str, int]:
    path = paper_dir / "paper_blocks.jsonl"
    if not path.exists():
        return {}
    pages: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        block = json.loads(line)
        block_id = block.get("block_id") or block.get("id")
        page = block.get("page_start") or block.get("source_page") or block.get("page")
        if block_id and page is not None:
            pages[str(block_id)] = int(page)
    return pages


def _with_block_page_map(chunk: Chunk, block_pages: dict[str, int]) -> Chunk:
    authoritative_map = {
        block_id: block_pages[block_id]
        for block_id in chunk.block_ids
        if block_id in block_pages
        and chunk.page_start <= block_pages[block_id] <= chunk.page_end
    }
    if set(authoritative_map) == set(chunk.block_ids):
        if chunk.block_page_map == authoritative_map:
            return chunk
        return chunk.model_copy(update={"block_page_map": authoritative_map})
    if chunk.block_page_map:
        return chunk
    return chunk


def _persist_qa_exception_audit(
    exc: BaseException, sample_id: str | None, run_id: str
) -> Path:
    settings = get_settings()
    audit_dir = settings.qa_response_audit_dir
    audit_dir.mkdir(parents=True, exist_ok=True)
    safe_sample_id = sample_id or "unknown"
    target = audit_dir / f"{safe_sample_id}-{run_id}-api-exception.json"
    payload = {
        "schema_version": "qa-api-exception-audit-v1",
        "sample_id": sample_id,
        "run_id": run_id,
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "traceback": traceback.format_exception(type(exc), exc, exc.__traceback__)[-20:],
        "api_key_persisted": False,
        "authorization_header_persisted": False,
        "request_payload_persisted": False,
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target
