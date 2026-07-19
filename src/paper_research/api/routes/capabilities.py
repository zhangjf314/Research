import importlib.util
import os
import shutil

import fitz
import httpx
import psycopg
from fastapi import APIRouter
from pydantic import BaseModel

from paper_research.config import get_settings
from paper_research.infrastructure.redis_service import get_redis_service
from paper_research.version import __display_version__, __version__

router = APIRouter()


class Capability(BaseModel):
    status: str
    configured: bool
    verified: bool
    detail: str | None = None


class CapabilitiesResponse(BaseModel):
    overall: str
    version: str
    display_version: str
    profile: str
    capabilities: dict[str, Capability]
    production_configuration_issues: list[str]
    stage13_30_budget: dict[str, object]


FULL_QA_BUDGET_VARS = [
    "LIVE_MODEL_CALLS_ENABLED",
    "FULL_QA_MAX_ITEMS",
    "FULL_QA_MAX_INPUT_TOKENS",
    "FULL_QA_MAX_OUTPUT_TOKENS",
    "FULL_QA_MAX_COST_USD",
    "FULL_QA_MAX_TOTAL_SECONDS",
]


DEEP_RESEARCH_BUDGET_VARS = [
    "DEEP_RESEARCH_ENABLED",
    "DEEP_RESEARCH_MAX_INPUT_TOKENS",
    "DEEP_RESEARCH_MAX_OUTPUT_TOKENS",
    "DEEP_RESEARCH_MAX_COST_USD",
    "DEEP_RESEARCH_MAX_TOTAL_SECONDS",
    "DEEP_RESEARCH_MAX_ITERATIONS",
    "DEEP_RESEARCH_MAX_PAPERS",
]


def _budget_presence() -> dict[str, object]:
    full_present = {name: bool(os.getenv(name)) for name in FULL_QA_BUDGET_VARS}
    deep_present = {name: bool(os.getenv(name)) for name in DEEP_RESEARCH_BUDGET_VARS}
    live_enabled = os.getenv("LIVE_MODEL_CALLS_ENABLED", "").lower() == "true"
    missing_full = [name for name, present in full_present.items() if not present]
    full_ready = live_enabled and not missing_full
    if full_ready:
        status = "FULL_QA_BUDGET_READY"
    elif live_enabled:
        status = "SMOKE_ONLY_BUDGET_INCOMPLETE"
    else:
        status = "BLOCKED_BY_LIVE_MODEL_CALLS_DISABLED"
    return {
        "status": status,
        "live_model_calls_enabled": live_enabled,
        "full_qa_budget_ready": full_ready,
        "missing_full_qa_budget_vars": missing_full,
        "full_qa_budget_vars_present": full_present,
        "deep_research_budget_vars_present": deep_present,
    }


@router.get("/capabilities", response_model=CapabilitiesResponse)
def capabilities() -> CapabilitiesResponse:
    settings = get_settings()
    redis_service = get_redis_service()
    redis_up = redis_service.ping()
    grobid_verified = False
    if settings.grobid_url:
        try:
            response = httpx.get(f"{settings.grobid_url.rstrip('/')}/api/isalive", timeout=1)
            grobid_verified = response.is_success
        except httpx.HTTPError:
            pass
    tesseract = shutil.which("tesseract")
    checkpoint_verified = settings.checkpoint_provider == "memory"
    checkpoint_detail = settings.checkpoint_provider
    if settings.checkpoint_provider == "postgres" and settings.checkpoint_database_url:
        try:
            with psycopg.connect(settings.checkpoint_database_url, connect_timeout=1) as connection:
                checkpoint_verified = bool(
                    connection.execute(
                        "SELECT to_regclass('public.checkpoints') IS NOT NULL"
                    ).fetchone()[0]
                )
        except psycopg.Error as exc:
            checkpoint_detail = f"postgres: {type(exc).__name__}"
    embedding_ready = not settings.embedding_configuration_issues
    reranker_ready = not settings.rerank_configuration_issues
    items = {
        "pymupdf": Capability(
            status="available", configured=True, verified=True, detail=fitz.VersionBind
        ),
        "ocr_engine": Capability(
            status="available" if tesseract else "degraded",
            configured=bool(tesseract),
            verified=bool(tesseract),
            detail=tesseract or "Tesseract executable not found",
        ),
        "tesseract": Capability(
            status="available" if tesseract else "unavailable",
            configured=bool(tesseract),
            verified=bool(tesseract),
            detail=tesseract,
        ),
        "docling": Capability(
            status="available" if importlib.util.find_spec("docling") else "unavailable",
            configured=importlib.util.find_spec("docling") is not None,
            verified=False,
            detail="not exercised in this process",
        ),
        "grobid": Capability(
            status="available" if grobid_verified else "degraded",
            configured=bool(settings.grobid_url),
            verified=grobid_verified,
            detail=settings.grobid_url and "configured; live check failed" or "not configured",
        ),
        "embedding": Capability(
            status=(
                "available"
                if settings.app_profile == "baseline" or embedding_ready
                else "degraded"
            ),
            configured=settings.embedding_provider != "hash" or settings.app_profile == "baseline",
            verified=settings.embedding_provider == "hash",
            detail=f"{settings.embedding_provider}/{settings.embedding_model}",
        ),
        "reranker": Capability(
            status=(
                "disabled"
                if not settings.rerank_enabled
                else "configured" if reranker_ready else "degraded"
            ),
            configured=settings.rerank_enabled and reranker_ready,
            verified=settings.rerank_provider == "lexical" and settings.rerank_enabled,
            detail=f"{settings.rerank_provider}/{settings.rerank_model}",
        ),
        "llm": Capability(
            status=(
                "available"
                if settings.llm_provider == "template" and settings.app_profile == "baseline"
                else "configured" if not settings.llm_configuration_issues else "degraded"
            ),
            configured=not settings.llm_configuration_issues,
            verified=settings.llm_provider == "template",
            detail=f"{settings.llm_provider}/{settings.llm_model}",
        ),
        "redis": Capability(
            status="available" if redis_up else "degraded",
            configured=bool(settings.redis_url),
            verified=redis_up,
            detail=str(redis_service.stats()),
        ),
        "langgraph_checkpoint": Capability(
            status="available" if checkpoint_verified else "degraded",
            configured=settings.checkpoint_provider in {"memory", "postgres"},
            verified=checkpoint_verified,
            detail=checkpoint_detail,
        ),
    }
    degraded = any(item.status in {"degraded", "unavailable"} for item in items.values())
    return CapabilitiesResponse(
        overall="degraded" if degraded else "available",
        version=__version__,
        display_version=__display_version__,
        profile=settings.app_profile,
        capabilities=items,
        production_configuration_issues=settings.production_configuration_issues,
        stage13_30_budget=_budget_presence(),
    )
