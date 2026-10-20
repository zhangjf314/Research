import json
import time
import traceback
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator
from qdrant_client import QdrantClient

from paper_research.chunking.types import Chunk
from paper_research.config import get_settings
from paper_research.generation.evidence_contract_adapter import ReadOnlyEvidenceContractResult
from paper_research.generation.qa_service import Answer, QAService
from paper_research.indexing.registry import (
    IndexIdentityError,
    IndexRegistry,
    runtime_index_identity,
)
from paper_research.indexing.vector_store import QdrantVectorStore
from paper_research.providers.factory import (
    ProviderConfigurationError,
    build_embedding_provider,
    build_provider_bundle,
    build_reranker,
)
from paper_research.providers.llm import LLMProviderError
from paper_research.retrieval.context_builder import ContextBuilder, ContextItem
from paper_research.retrieval.contextual import ContextualChunk
from paper_research.retrieval.dense import DenseRetriever
from paper_research.retrieval.filters import RetrievalFilter
from paper_research.retrieval.hybrid import HybridRetrievalResult, HybridRetriever
from paper_research.retrieval.pipeline import build_v2_retriever, retrieval_collection
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
    include_evidence_contract: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def validate_evidence_opt_in(self):
        if self.include_evidence_contract:
            if not self.query.strip() or len(self.query) > 2000:
                raise ValueError(
                    "evidence opt-in query must be nonblank and at most 2000 characters"
                )
            scope = self.filters.paper_ids or []
            if len(scope) > 38:
                raise ValueError("evidence opt-in paper scope must contain at most 38 UUIDs")
            for paper_id in scope:
                uuid.UUID(paper_id)
        return self


class HybridRetrievalResponse(BaseModel):
    context: list[ContextItem]
    trace: RetrievalTrace


class HybridRetrievalEvidenceResponse(HybridRetrievalResponse):
    evidence_contract: ReadOnlyEvidenceContractResult


def _run_hybrid(payload: HybridRetrievalRequest, observer=None):
    settings = get_settings()
    registry = IndexRegistry(settings.data_dir / "index_registry.json")
    collection = retrieval_collection(settings, registry)
    store = QdrantVectorStore(
        QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key),
        collection,
        settings.embedding_dimensions,
        chunk_model=ContextualChunk if settings.retrieval_v2_enabled else Chunk,
    )
    expected = runtime_index_identity(settings.provider_metadata, collection)
    registry.verify_identity(collection, expected, store.client)
    embedding = build_embedding_provider(settings)
    actual = runtime_index_identity(
        {**settings.provider_metadata,
         "embedding_provider": embedding.provider_name,
         "embedding_model": embedding.model_name,
         "embedding_revision": embedding.revision,
         "embedding_dimension": embedding.dimensions}, collection,
    )
    differences = [key for key in actual if actual[key] != expected[key]]
    if differences:
        raise IndexIdentityError("INDEX_IDENTITY_CONFLICT", differences, status_code=409)
    def check_object():
        actual_identity = runtime_index_identity(
            {**settings.provider_metadata, "embedding_provider": embedding.provider_name,
             "embedding_model": embedding.model_name, "embedding_revision": embedding.revision,
             "embedding_dimension": embedding.dimensions}, collection)
        if actual_identity != expected or store.collection != collection:
            raise IndexIdentityError("INDEX_IDENTITY_CONFLICT", status_code=409)
    with registry.query_admission(settings.active_collection, collection, expected, store.client,
                                  object_check=check_object,
                                  supported=not settings.retrieval_v2_enabled) as (verified, _):
        chunks = _load_chunks(settings.parsed_papers_dir, payload.filters.paper_ids,
                              settings.index_version, verified_chunks=verified)
        if not chunks:
            raise HTTPException(status_code=409, detail="no indexed chunk files found")
        reranker = build_reranker(settings)
        result = HybridRetriever(
            DenseRetriever(embedding, store), BM25Retriever(chunks), reranker,
            ContextBuilder(include_neighbors=False,
                max_characters=settings.qa_context_token_budget * 4,
                max_tokens=settings.qa_context_token_budget),
            JsonlTraceRepository(settings.retrieval_trace_path),
            provider_metadata=settings.provider_metadata,
            rerank_input_k=settings.rerank_input_k, rerank_output_k=settings.rerank_output_k,
        ).retrieve(payload.query, payload.filters,
            recall_k=max(payload.recall_k, settings.rerank_input_k) if settings.rerank_enabled else payload.recall_k,
            top_k=payload.top_k, retrieval_scope="paper" if payload.filters.paper_ids else "global")
        return observer(result) if observer is not None else result


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
                top_k=(
                    settings.retrieval_v2_final_top_k
                    if settings.retrieval_v2_enabled and "top_k" not in payload.model_fields_set
                    else payload.top_k
                ),
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
    except IndexIdentityError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
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


@router.post(
    "/retrieve", response_model=HybridRetrievalEvidenceResponse | HybridRetrievalResponse,
)
def hybrid_retrieve(
    payload: HybridRetrievalRequest,
) -> HybridRetrievalEvidenceResponse | HybridRetrievalResponse:
    try:
        def observe(result):
            if payload.include_evidence_contract:
                from paper_research.generation.evidence_contract_adapter import (
                    build_readonly_evidence_contract,
                    evidence_contract_failure,
                )

                try:
                    sidecar = build_readonly_evidence_contract(
                        payload.query, result.context, result.trace,
                        parsed_root=get_settings().parsed_papers_dir,
                    )
                except Exception:
                    sidecar = evidence_contract_failure(payload.query, result.context, result.trace)
                return HybridRetrievalEvidenceResponse(
                    context=result.context, trace=result.trace, evidence_contract=sidecar,
                )
            return HybridRetrievalResponse(context=result.context, trace=result.trace)
        return _run_hybrid(payload, observer=observe)
    except IndexIdentityError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except ProviderConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"hybrid retrieval unavailable: {type(exc).__name__}"
        ) from exc


def _load_chunks(root: Path, paper_ids: list[str] | None, index_version: str,
                 *, verified_chunks: list[Chunk] | None = None) -> list[Chunk]:
    if verified_chunks is not None:
        pages = {}
        result = []
        ordered = ([chunk for paper_id in paper_ids for chunk in verified_chunks
                    if chunk.paper_id == paper_id] if paper_ids else verified_chunks)
        for chunk in ordered:
            if paper_ids and chunk.paper_id not in paper_ids:
                continue
            if chunk.paper_id not in pages:
                try:
                    pages[chunk.paper_id] = _load_block_pages(root / chunk.paper_id)
                except (OSError, ValueError, TypeError, AttributeError):
                    pages[chunk.paper_id] = {}
            result.append(_with_block_page_map(chunk, pages[chunk.paper_id]))
        return result
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
        if block_id in block_pages and chunk.page_start <= block_pages[block_id] <= chunk.page_end
    }
    if set(authoritative_map) == set(chunk.block_ids):
        if chunk.block_page_map == authoritative_map:
            return chunk
        return chunk.model_copy(update={"block_page_map": authoritative_map})
    if chunk.block_page_map:
        return chunk
    return chunk


def _persist_qa_exception_audit(exc: BaseException, sample_id: str | None, run_id: str) -> Path:
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
