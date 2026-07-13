from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from paper_research.config import get_settings
from paper_research.db import get_db
from paper_research.infrastructure.redis_service import get_redis_service
from paper_research.schemas.paper import PaperRead
from paper_research.search.clients import ArxivClient, SemanticScholarClient
from paper_research.search.http import CachedRetryClient
from paper_research.search.import_service import PaperImportService
from paper_research.search.models import PaperCandidate, SearchRequest, SearchResponse
from paper_research.search.service import PaperSearchService

router = APIRouter()
DbSession = Annotated[Session, Depends(get_db)]


def get_http_client() -> CachedRetryClient:
    settings = get_settings()
    return CachedRetryClient(
        settings.search_cache_dir,
        ttl_seconds=settings.search_cache_ttl_seconds,
        retries=settings.external_request_retries,
        redis_cache=get_redis_service(),
    )


@router.post("/papers", response_model=SearchResponse)
def search_papers(payload: SearchRequest) -> SearchResponse:
    settings = get_settings()
    http = get_http_client()
    service = PaperSearchService(
        [ArxivClient(http), SemanticScholarClient(http, settings.semantic_scholar_api_key)]
    )
    response = service.search(payload)
    response.telemetry = {
        "fallback_used": bool(
            response.telemetry.get("fallback_used")
            or http.telemetry.get("fallback_used")
        ),
        "rate_limited": bool(
            response.telemetry.get("rate_limited")
            or http.telemetry.get("rate_limited")
        ),
        "cache_hit": bool(http.telemetry.get("cache_hit")),
        "retry_count": int(http.telemetry.get("retry_count", 0)),
    }
    if not response.candidates and response.source_errors:
        raise HTTPException(status_code=503, detail=response.source_errors)
    return response


@router.post("/import", response_model=PaperRead)
def import_paper(candidate: PaperCandidate, db: DbSession) -> PaperRead:
    try:
        lock_name = candidate.arxiv_id or candidate.doi or candidate.source_id
        redis_service = get_redis_service()
        redis_available = redis_service.ping()
        with redis_service.lock(f"import:{lock_name}") as acquired:
            if redis_available and not acquired:
                raise HTTPException(status_code=409, detail="paper import is already running")
            # Redis unavailable degrades to the database/file-hash idempotency path.
            service = PaperImportService(db, get_settings(), get_http_client())
            result = service.import_candidate(candidate)
        return PaperRead.model_validate(result.paper)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        detail = f"paper import failed: {type(exc).__name__}"
        raise HTTPException(status_code=503, detail=detail) from exc
