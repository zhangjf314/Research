from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from paper_research.config import get_settings
from paper_research.indexing.rebuild import IndexRebuildService

router = APIRouter()


class IndexSwitchRequest(BaseModel):
    collection: str


@router.get("")
def index_registry() -> dict:
    settings = get_settings()
    return IndexRebuildService(settings).registry.read()


@router.post("/rebuild")
def rebuild_index() -> dict:
    return IndexRebuildService(get_settings()).rebuild()


@router.get("/rebuild/{rebuild_id}")
def rebuild_status(rebuild_id: str) -> dict:
    status = IndexRebuildService(get_settings()).get_status(rebuild_id)
    if status is None:
        raise HTTPException(status_code=404, detail="rebuild not found")
    return status


@router.post("/switch")
def switch_index(payload: IndexSwitchRequest) -> dict:
    try:
        return IndexRebuildService(get_settings()).switch(payload.collection)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
