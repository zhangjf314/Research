from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from paper_research.evaluation.review_store import GoldReviewStore

router = APIRouter()


class ReviewAction(BaseModel):
    action: str
    reviewer: str = Field(min_length=2)
    review_notes: str | None = None
    updates: dict | None = None


def store() -> GoldReviewStore:
    return GoldReviewStore(Path("data/evaluation/gold-set-v1.jsonl"))


@router.get("/review")
def list_review_items(
    status: str | None = None,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
) -> dict:
    items = store().list()
    if status:
        items = [item for item in items if item["review_status"] == status]
    return {"total": len(items), "items": items[offset : offset + limit]}


@router.get("/review/papers/{paper_id}/pdf", response_class=FileResponse)
def review_pdf(paper_id: str) -> FileResponse:
    if not paper_id.replace(".", "").isdigit():
        raise HTTPException(status_code=404, detail="PDF not found")
    path = Path("data/raw/audit") / f"{paper_id}.pdf"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(path, media_type="application/pdf", filename=path.name)


@router.get("/review/{question_id}")
def get_review_item(question_id: str) -> dict:
    review_store = store()
    item = review_store.get(question_id)
    if item is None:
        raise HTTPException(status_code=404, detail="review item not found")
    return {"item": item, "evidence": review_store.evidence(item)}


@router.post("/review/{question_id}")
def review_item(question_id: str, payload: ReviewAction) -> dict:
    try:
        return store().review(
            question_id,
            action=payload.action,
            reviewer=payload.reviewer,
            review_notes=payload.review_notes,
            updates=payload.updates,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="review item not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
