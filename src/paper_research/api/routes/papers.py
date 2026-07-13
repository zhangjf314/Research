import json
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from qdrant_client import QdrantClient
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from paper_research.analysis.service import AnalysisService
from paper_research.analysis.types import PaperAnalysis
from paper_research.chunking.structural_chunker import StructuralChunker
from paper_research.config import get_settings
from paper_research.db import get_db
from paper_research.indexing.registry import IndexRegistry
from paper_research.indexing.service import IndexingService
from paper_research.indexing.vector_store import QdrantVectorStore
from paper_research.ingestion.upload_service import UploadService, UploadValidationError
from paper_research.models.paper import PaperStatus
from paper_research.providers.factory import build_embedding_provider
from paper_research.repositories.paper import PaperRepository
from paper_research.schemas.paper import PaperCreate, PaperRead, PaperUploadResponse

router = APIRouter()
DbSession = Annotated[Session, Depends(get_db)]


@router.post("/upload", response_model=PaperUploadResponse, status_code=status.HTTP_201_CREATED)
def upload_paper(file: Annotated[UploadFile, File()], db: DbSession) -> PaperUploadResponse:
    try:
        result = UploadService(db, get_settings()).ingest(file)
    except UploadValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        detail = f"PDF parsing failed: {type(exc).__name__}"
        raise HTTPException(status_code=500, detail=detail) from exc
    return PaperUploadResponse(
        paper=PaperRead.model_validate(result.paper),
        duplicate=result.duplicate,
        artifacts={name: str(path) for name, path in result.artifact_paths.items()},
    )


@router.post("/{paper_id}/index")
def index_paper(paper_id: uuid.UUID, db: DbSession) -> dict[str, int | str]:
    repository = PaperRepository(db)
    paper = repository.get(paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="paper not found")
    settings = get_settings()
    blocks_path = settings.parsed_papers_dir / str(paper_id) / "paper_blocks.jsonl"
    if not blocks_path.exists():
        raise HTTPException(status_code=409, detail="paper has not been parsed")
    paper.parse_status = PaperStatus.indexing
    repository.save(paper)
    try:
        embedding = build_embedding_provider(settings)
        store = QdrantVectorStore(
            QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key),
            IndexRegistry(settings.data_dir / "index_registry.json").resolve(
                settings.active_collection
            ),
            settings.embedding_dimensions,
        )
        chunks_path = settings.parsed_papers_dir / str(paper_id) / (
            f"paper_chunks.{settings.index_version}.jsonl"
        )
        chunks = IndexingService(
            StructuralChunker(settings.chunk_max_tokens, settings.chunk_overlap_tokens),
            embedding,
            store,
        ).index(
            str(paper_id),
            blocks_path,
            chunks_path,
            metadata=settings.provider_metadata,
        )
        # Baseline compatibility file used by existing local audit scripts.
        compatibility_path = settings.parsed_papers_dir / str(paper_id) / "paper_chunks.jsonl"
        compatibility_path.write_text(chunks_path.read_text(encoding="utf-8"), encoding="utf-8")
        paper.parse_status = PaperStatus.ready
        paper.index_status = "READY"
        repository.save(paper)
        return {"paper_id": str(paper_id), "status": "READY", "chunk_count": len(chunks)}
    except Exception as exc:
        paper.index_status = "FAILED"
        repository.save(paper)
        detail = f"indexing failed: {type(exc).__name__}"
        raise HTTPException(status_code=503, detail=detail) from exc


@router.get("/{paper_id}/pdf", response_class=FileResponse)
def get_paper_pdf(paper_id: uuid.UUID, db: DbSession) -> FileResponse:
    paper = PaperRepository(db).get(paper_id)
    if paper is None or not paper.pdf_path:
        raise HTTPException(status_code=404, detail="PDF not found")
    path = Path(paper.pdf_path).resolve()
    raw_root = get_settings().raw_papers_dir.resolve()
    if raw_root not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(path, media_type="application/pdf", filename=f"{paper_id}.pdf")


@router.post("/{paper_id}/analyze", response_model=PaperAnalysis)
def analyze_paper(paper_id: uuid.UUID, db: DbSession) -> PaperAnalysis:
    if PaperRepository(db).get(paper_id) is None:
        raise HTTPException(status_code=404, detail="paper not found")
    parsed_dir = get_settings().parsed_papers_dir / str(paper_id)
    if not (parsed_dir / "paper_blocks.jsonl").exists():
        raise HTTPException(status_code=409, detail="paper has not been parsed")
    return AnalysisService().analyze_artifacts(str(paper_id), parsed_dir)


@router.get("/{paper_id}/analysis", response_model=PaperAnalysis)
def get_paper_analysis(paper_id: uuid.UUID, db: DbSession) -> PaperAnalysis:
    if PaperRepository(db).get(paper_id) is None:
        raise HTTPException(status_code=404, detail="paper not found")
    path = get_settings().parsed_papers_dir / str(paper_id) / "paper_analysis.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="paper analysis not found")
    return PaperAnalysis.model_validate(json.loads(path.read_text(encoding="utf-8")))


@router.post("", response_model=PaperRead, status_code=status.HTTP_201_CREATED)
def create_paper(payload: PaperCreate, db: DbSession) -> PaperRead:
    try:
        return PaperRepository(db).create(payload)
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="paper already exists") from exc


@router.get("", response_model=list[PaperRead])
def list_papers(
    db: DbSession,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[PaperRead]:
    return PaperRepository(db).list(limit=limit, offset=offset)


@router.get("/{paper_id}", response_model=PaperRead)
def get_paper(paper_id: uuid.UUID, db: DbSession) -> PaperRead:
    paper = PaperRepository(db).get(paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="paper not found")
    return paper
