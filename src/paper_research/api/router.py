from fastapi import APIRouter

from paper_research.api.routes import (
    capabilities,
    evaluation,
    health,
    indexes,
    papers,
    rag,
    research,
    search,
    ui,
)

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(capabilities.router, tags=["capabilities"])
api_router.include_router(indexes.router, prefix="/indexes", tags=["indexes"])
api_router.include_router(evaluation.router, prefix="/evaluation", tags=["evaluation"])
api_router.include_router(papers.router, prefix="/papers", tags=["papers"])
api_router.include_router(rag.router, tags=["rag"])
api_router.include_router(ui.router, prefix="/ui", tags=["ui"])
api_router.include_router(search.router, prefix="/search", tags=["search"])
api_router.include_router(research.router, prefix="/research", tags=["research"])
