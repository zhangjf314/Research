from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from qdrant_client import QdrantClient
from sqlalchemy.orm import Session

from paper_research.config import Settings
from paper_research.indexing.vector_store import QdrantVectorStore
from paper_research.repositories.paper import PaperRepository


@dataclass
class CleanupStep:
    name: str
    status: str
    detail: str = ""
    count: int = 0


@dataclass
class PaperCleanupResult:
    paper_id: str
    dry_run: bool
    deleted: bool
    steps: list[CleanupStep] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def qdrant_points_removed(self) -> int:
        return sum(step.count for step in self.steps if step.name.startswith("qdrant:"))


class PaperCleanupService:
    def __init__(self, session: Session, settings: Settings) -> None:
        self.session = session
        self.settings = settings
        self.repository = PaperRepository(session)

    def purge(self, paper_id: uuid.UUID, *, dry_run: bool = True) -> PaperCleanupResult:
        paper = self.repository.get(paper_id)
        result = PaperCleanupResult(paper_id=str(paper_id), dry_run=dry_run, deleted=False)
        if paper is None:
            result.steps.append(CleanupStep("database_record", "missing"))
            return result

        client = QdrantClient(url=self.settings.qdrant_url, api_key=self.settings.qdrant_api_key)
        for collection in self._known_collections():
            store = QdrantVectorStore(client, collection, self.settings.embedding_dimensions)
            before = store.count_by_paper_id(str(paper_id))
            if dry_run:
                result.steps.append(
                    CleanupStep(f"qdrant:{collection}", "would_delete", count=before)
                )
            else:
                removed = store.delete_by_paper_id(str(paper_id))
                result.steps.append(CleanupStep(f"qdrant:{collection}", "deleted", count=removed))

        if paper.pdf_path:
            self._remove_path(Path(paper.pdf_path), result, "raw_pdf", dry_run=dry_run)
        else:
            result.steps.append(CleanupStep("raw_pdf", "missing"))
        self._remove_path(
            self.settings.parsed_papers_dir / str(paper_id),
            result,
            "parsed_directory",
            dry_run=dry_run,
        )

        if dry_run:
            result.steps.append(CleanupStep("database_record", "would_delete"))
            return result

        self.session.delete(paper)
        self.session.commit()
        result.steps.append(CleanupStep("database_record", "deleted"))
        result.deleted = True
        return result

    def _known_collections(self) -> list[str]:
        names = {
            self.settings.active_collection,
            self.settings.baseline_collection,
            self.settings.production_collection,
        }
        registry_path = self.settings.data_dir / "index_registry.json"
        if registry_path.exists():
            try:
                data = json.loads(registry_path.read_text(encoding="utf-8") or "{}")
                values = data.values() if isinstance(data, dict) else []
                for value in values:
                    if isinstance(value, str):
                        names.add(value)
                    elif isinstance(value, dict):
                        for key in ("collection", "collection_name", "name"):
                            if isinstance(value.get(key), str):
                                names.add(value[key])
            except json.JSONDecodeError:
                pass
        return sorted(name for name in names if name)

    @staticmethod
    def _remove_path(
        path: Path,
        result: PaperCleanupResult,
        name: str,
        *,
        dry_run: bool,
    ) -> None:
        if not path.exists():
            result.steps.append(CleanupStep(name, "missing", detail=str(path)))
            return
        if dry_run:
            result.steps.append(CleanupStep(name, "would_delete", detail=str(path)))
            return
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            result.steps.append(CleanupStep(name, "deleted", detail=str(path)))
        except OSError as exc:
            result.steps.append(CleanupStep(name, "failed", detail=f"{path}: {exc}"))
            result.warnings.append(f"{name} cleanup failed: {exc}")
