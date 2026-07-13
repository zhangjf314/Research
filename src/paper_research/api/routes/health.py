from typing import Annotated, Literal

import httpx
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from paper_research.config import get_settings
from paper_research.db import get_db
from paper_research.infrastructure.redis_service import get_redis_service

router = APIRouter()
DbSession = Annotated[Session, Depends(get_db)]


class ComponentHealth(BaseModel):
    status: Literal["up", "down"]
    detail: str | None = None


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded"]
    components: dict[str, ComponentHealth]


@router.get("/health", response_model=HealthResponse)
def health(db: DbSession) -> HealthResponse:
    components: dict[str, ComponentHealth] = {}
    try:
        db.execute(text("SELECT 1"))
        components["postgres"] = ComponentHealth(status="up")
    except Exception as exc:  # health endpoint must report dependency failure
        components["postgres"] = ComponentHealth(status="down", detail=type(exc).__name__)

    settings = get_settings()
    try:
        response = httpx.get(f"{settings.qdrant_url.rstrip('/')}/healthz", timeout=2.0)
        response.raise_for_status()
        components["qdrant"] = ComponentHealth(status="up")
    except Exception as exc:  # health endpoint must report dependency failure
        components["qdrant"] = ComponentHealth(status="down", detail=type(exc).__name__)

    redis_service = get_redis_service()
    redis_up = redis_service.ping()
    stats = redis_service.stats()
    components["redis"] = ComponentHealth(
        status="up" if redis_up else "down",
        detail=(
            f"used={stats.get('used', False)}; keys={stats.get('key_count', 0)}; "
            f"cache_hit_rate={stats.get('cache_hit_rate', 0)}"
            if redis_up
            else redis_service.last_error
        ),
    )

    overall = "healthy" if all(item.status == "up" for item in components.values()) else "degraded"
    return HealthResponse(status=overall, components=components)
