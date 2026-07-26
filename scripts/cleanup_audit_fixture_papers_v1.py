from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path

from paper_research.config import get_settings
from paper_research.db import SessionLocal
from paper_research.ingestion.cleanup_service import PaperCleanupService
from paper_research.models.paper import Paper
from paper_research.repositories.paper import PaperRepository

FIXTURE_TITLE = re.compile(
    r"^(fully-scanned|mixed-native-scanned|text-native)(-\d{14})?$", re.IGNORECASE
)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _known_audit_ids(root: Path) -> set[str]:
    ids: set[str] = set()
    for path in (
        root / "data/evaluation/docker-ocr-production-v2.json",
        root / "artifacts/soak-test-portfolio-v1.json",
    ):
        data = _load_json(path)
        stack = [data]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                for key, item in value.items():
                    if key == "paper_id" and isinstance(item, str):
                        ids.add(item)
                    else:
                        stack.append(item)
            elif isinstance(value, list):
                stack.extend(value)
    return ids


def classify(paper: Paper, audit_ids: set[str]) -> tuple[str, str]:
    paper_id = str(paper.id)
    title = paper.title or ""
    source_type = paper.source_type or ""
    if paper_id in audit_ids:
        return "CONFIRMED_AUDIT_FIXTURE", "paper_id appears in OCR/soak audit artifact"
    if source_type == "audit_fixture" and FIXTURE_TITLE.match(title):
        return (
            "CONFIRMED_AUDIT_FIXTURE",
            "source_type=audit_fixture and title matches fixture pattern",
        )
    if FIXTURE_TITLE.match(title):
        return "CONFIRMED_AUDIT_FIXTURE", "title matches deterministic OCR fixture pattern"
    if any(token in title.lower() for token in ("docker ocr", "portfolio soak", "synthetic")):
        return "LIKELY_AUDIT_FIXTURE", "title contains audit fixture marker"
    return "REAL_USER_PAPER", "not classified as fixture"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="actually delete confirmed fixtures")
    parser.add_argument("--dry-run", action="store_true", help="audit without deleting")
    args = parser.parse_args()
    dry_run = not args.apply
    settings = get_settings()
    root = settings.data_dir.parent.resolve()
    audit_ids = _known_audit_ids(root)
    output_json = root / "data/evaluation/audit-fixture-cleanup-v1.json"
    output_md = root / "docs/audit-fixture-cleanup-v1.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    with SessionLocal() as session:
        repo = PaperRepository(session)
        papers = repo.list(limit=1000, include_fixtures=True)
        before_count = len(papers)
        service = PaperCleanupService(session, settings)
        records = []
        removed_ids = []
        for paper in papers:
            classification, reason = classify(paper, audit_ids)
            if classification == "REAL_USER_PAPER":
                continue
            result = None
            if classification == "CONFIRMED_AUDIT_FIXTURE":
                result = service.purge(paper.id, dry_run=dry_run)
                if not dry_run and result.deleted:
                    removed_ids.append(str(paper.id))
            records.append(
                {
                    "paper_id": str(paper.id),
                    "title": paper.title,
                    "source_type": paper.source_type,
                    "created_at": paper.created_at.isoformat() if paper.created_at else None,
                    "raw_pdf_exists": bool(paper.pdf_path and Path(paper.pdf_path).exists()),
                    "parsed_directory_exists": bool(
                        (settings.parsed_papers_dir / str(paper.id)).exists()
                    ),
                    "formal_evaluation_reference_count": 0,
                    "classification": classification,
                    "classification_reason": reason,
                    "cleanup_result": asdict(result) if result else None,
                }
            )
        after_count = len(repo.list(limit=1000, include_fixtures=True))

    summary = {
        "schema_version": "audit-fixture-cleanup-v1",
        "dry_run": dry_run,
        "library_before_count": before_count,
        "library_after_count": after_count,
        "confirmed_fixture_count": sum(
            1 for row in records if row["classification"] == "CONFIRMED_AUDIT_FIXTURE"
        ),
        "fixture_ids_removed": removed_ids,
        "records": records,
    }
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Audit Fixture Cleanup v1",
        "",
        f"- Dry run: {dry_run}",
        f"- Library before count: {before_count}",
        f"- Library after count: {after_count}",
        f"- Confirmed fixture count: {summary['confirmed_fixture_count']}",
        f"- Fixture IDs removed: {', '.join(removed_ids) if removed_ids else 'none'}",
        "",
        "## Records",
        "",
    ]
    for row in records:
        lines.append(f"- `{row['paper_id']}` · {row['classification']} · {row['title']}")
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
