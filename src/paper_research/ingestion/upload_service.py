import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path

import fitz
from fastapi import UploadFile
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from paper_research.analysis.service import AnalysisService
from paper_research.config import Settings
from paper_research.ingestion.artifacts import write_parse_artifacts
from paper_research.models.paper import Paper, PaperStatus
from paper_research.parsing.page_assets import render_page_assets
from paper_research.parsing.router import ParserRouter
from paper_research.repositories.paper import PaperRepository


class UploadValidationError(ValueError):
    pass


@dataclass(frozen=True)
class UploadResult:
    paper: Paper
    duplicate: bool
    artifact_paths: dict[str, Path]


class UploadService:
    def __init__(
        self, session: Session, settings: Settings, parser_router: ParserRouter | None = None
    ) -> None:
        self.session = session
        self.settings = settings
        self.parser_router = parser_router or ParserRouter(settings=settings)
        self.repository = PaperRepository(session)

    def ingest(self, upload: UploadFile) -> UploadResult:
        self._validate_filename(upload.filename)
        raw_dir = self.settings.raw_papers_dir
        raw_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = raw_dir / f".{uuid.uuid4().hex}.part"
        try:
            file_hash, size = self._copy_and_hash(upload, temporary_path)
            if size == 0:
                raise UploadValidationError("uploaded PDF is empty")
            duplicate = self.repository.get_by_hash(file_hash)
            if duplicate is not None:
                return UploadResult(duplicate, True, self._artifact_paths(duplicate.id))

            final_path = raw_dir / f"{file_hash}.pdf"
            temporary_path.replace(final_path)
            self._validate_pdf(final_path)
            paper = Paper(
                title=Path(upload.filename or "paper.pdf").stem,
                authors=[],
                keywords=[],
                file_hash=file_hash,
                pdf_path=str(final_path),
                parse_status=PaperStatus.parsing,
            )
            try:
                self.session.add(paper)
                self.session.commit()
                self.session.refresh(paper)
            except IntegrityError:
                self.session.rollback()
                duplicate = self.repository.get_by_hash(file_hash)
                if duplicate is None:
                    raise
                return UploadResult(duplicate, True, self._artifact_paths(duplicate.id))

            try:
                parsed = self.parser_router.parse(final_path)
                output_dir = self.settings.parsed_papers_dir / str(paper.id)
                paths = write_parse_artifacts(parsed, output_dir)
                AnalysisService().analyze_artifacts(str(paper.id), output_dir)
                paths["analysis"] = output_dir / "paper_analysis.json"
                page_assets = render_page_assets(
                    final_path, output_dir / "page_assets", dpi=self.settings.page_asset_dpi
                )
                paths["page_assets"] = output_dir / "page_assets"
                paths["first_page"] = page_assets[0]
                paper.title = parsed.metadata.title or paper.title
                paper.authors = parsed.metadata.authors
                paper.parse_status = PaperStatus.parsed
                self.repository.save(paper)
                return UploadResult(paper, False, paths)
            except Exception:
                paper.parse_status = PaperStatus.failed
                self.repository.save(paper)
                raise
        finally:
            temporary_path.unlink(missing_ok=True)

    def _copy_and_hash(self, upload: UploadFile, destination: Path) -> tuple[str, int]:
        digest = hashlib.sha256()
        size = 0
        upload.file.seek(0)
        with destination.open("wb") as stream:
            while chunk := upload.file.read(1024 * 1024):
                size += len(chunk)
                if size > self.settings.max_upload_bytes:
                    raise UploadValidationError("uploaded PDF exceeds the configured size limit")
                digest.update(chunk)
                stream.write(chunk)
        return digest.hexdigest(), size

    @staticmethod
    def _validate_filename(filename: str | None) -> None:
        if not filename or Path(filename).suffix.lower() != ".pdf":
            raise UploadValidationError("only .pdf files are accepted")

    @staticmethod
    def _validate_pdf(path: Path) -> None:
        if path.read_bytes()[:5] != b"%PDF-":
            path.unlink(missing_ok=True)
            raise UploadValidationError("file content is not a PDF")
        try:
            with fitz.open(path) as document:
                if document.page_count < 1:
                    raise UploadValidationError("PDF has no pages")
        except fitz.FileDataError as exc:
            path.unlink(missing_ok=True)
            raise UploadValidationError("PDF is corrupted") from exc

    def _artifact_paths(self, paper_id: uuid.UUID) -> dict[str, Path]:
        output_dir = self.settings.parsed_papers_dir / str(paper_id)
        return {
            "metadata": output_dir / "paper_metadata.json",
            "blocks": output_dir / "paper_blocks.jsonl",
            "report": output_dir / "parse_report.md",
            "page_assets": output_dir / "page_assets",
        }
