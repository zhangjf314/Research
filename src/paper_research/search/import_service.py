from io import BytesIO

from fastapi import UploadFile
from sqlalchemy.orm import Session

from paper_research.config import Settings
from paper_research.ingestion.upload_service import UploadResult, UploadService
from paper_research.search.http import CachedRetryClient
from paper_research.search.models import PaperCandidate


class PaperImportService:
    def __init__(self, session: Session, settings: Settings, http: CachedRetryClient) -> None:
        self.session = session
        self.settings = settings
        self.http = http

    def import_candidate(self, candidate: PaperCandidate) -> UploadResult:
        if not candidate.pdf_url:
            raise ValueError("candidate has no downloadable PDF")
        payload = self.http.get_bytes(candidate.pdf_url)
        upload = UploadFile(
            filename=f"{candidate.arxiv_id or candidate.source_id}.pdf",
            file=BytesIO(payload),
        )
        result = UploadService(self.session, self.settings).ingest(upload)
        paper = result.paper
        paper.title = candidate.title
        paper.authors = candidate.authors
        paper.abstract = candidate.abstract
        paper.year = candidate.year
        paper.venue = candidate.venue
        paper.doi = candidate.doi
        paper.arxiv_id = candidate.arxiv_id
        paper.source_type = candidate.source
        paper.source_url = candidate.source_url
        self.session.add(paper)
        self.session.commit()
        self.session.refresh(paper)
        return UploadResult(paper, result.duplicate, result.artifact_paths)
