from __future__ import annotations

import json
import threading
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path


class GoldReviewStore:
    _lock = threading.Lock()

    def __init__(self, dataset_path: Path, project_root: Path = Path(".")) -> None:
        self.dataset_path = dataset_path
        self.project_root = project_root

    def list(self) -> list[dict]:
        return [
            json.loads(line)
            for line in self.dataset_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def get(self, question_id: str) -> dict | None:
        return next((item for item in self.list() if item["question_id"] == question_id), None)

    def evidence(self, item: dict) -> list[dict]:
        evidence = []
        wanted = set(item.get("gold_block_ids") or [])
        for paper_id in item.get("gold_paper_ids") or []:
            path = (
                self.project_root
                / "data"
                / "reports"
                / "parsing-audit"
                / paper_id
                / "paper_blocks.jsonl"
            )
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                block = json.loads(line)
                if not wanted or block["block_id"] in wanted:
                    evidence.append({"paper_id": paper_id, **block})
        return evidence

    def review(
        self,
        question_id: str,
        *,
        action: str,
        reviewer: str,
        review_notes: str | None,
        updates: dict | None = None,
    ) -> dict:
        allowed = {"approve", "modify_approve", "unanswerable", "invalid", "defer"}
        if action not in allowed:
            raise ValueError(f"unsupported review action: {action}")
        with self._lock:
            items = self.list()
            target = next((item for item in items if item["question_id"] == question_id), None)
            if target is None:
                raise KeyError(question_id)
            if updates:
                editable = {
                    "question",
                    "scope",
                    "category",
                    "difficulty",
                    "answerable",
                    "gold_answer",
                    "required_claims",
                    "gold_paper_ids",
                    "gold_block_ids",
                    "gold_pages",
                    "citation_notes",
                }
                for key, value in updates.items():
                    if key in editable:
                        target[key] = value
            if action == "unanswerable":
                target.update(
                    {
                        "answerable": False,
                        "gold_answer": None,
                        "required_claims": [],
                        "gold_paper_ids": [],
                        "gold_block_ids": [],
                        "gold_pages": [],
                    }
                )
            target["review_status"] = {
                "approve": "approved",
                "modify_approve": "approved",
                "unanswerable": "approved",
                "invalid": "invalid",
                "defer": "pending",
            }[action]
            target["reviewer"] = reviewer
            target["reviewed_at"] = datetime.now(UTC).isoformat()
            target["review_notes"] = review_notes
            target["dataset_version"] = "gold-set-v1-human-review"
            self._write(items)
            self.write_reports(items)
            return target

    def write_reports(self, items: list[dict] | None = None) -> None:
        items = items or self.list()
        status = Counter(item["review_status"] for item in items)
        categories = Counter(item["category"] for item in items)
        progress = [
            "# Gold Set v1 Review Progress",
            "",
            f"- Total: {len(items)}",
            f"- Approved: {status['approved']}",
            f"- Pending: {status['pending']}",
            f"- Invalid: {status['invalid']}",
            "",
            "## Category coverage",
            "",
            *[f"- {key}: {value}" for key, value in sorted(categories.items())],
            "",
            "Only explicit human workbench actions can set `review_status=approved`.",
            "",
        ]
        audit = [
            "# Gold Set v1 Quality Audit",
            "",
            f"- Approved items eligible for formal metrics: {status['approved']}/{len(items)}",
            f"- Pending items excluded from formal metrics: {status['pending']}",
            f"- Items marked invalid: {status['invalid']}",
            "",
            "## Approval integrity",
            "",
            "Every approved record must contain reviewer, reviewed_at, review_notes, and "
            "dataset_version. Automated scripts never approve records.",
            "",
        ]
        docs = self.project_root / "docs"
        docs.mkdir(parents=True, exist_ok=True)
        (docs / "gold-set-review-progress.md").write_text("\n".join(progress), encoding="utf-8")
        (docs / "gold-set-quality-audit.md").write_text("\n".join(audit), encoding="utf-8")

    def _write(self, items: list[dict]) -> None:
        temporary = self.dataset_path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for item in items:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        temporary.replace(self.dataset_path)
