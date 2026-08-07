from __future__ import annotations

# ruff: noqa: E501
import argparse
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from paper_research.evaluation.rag_benchmark import read_jsonl
from paper_research.evaluation.rag_gold import (
    corpus_coverage,
    dataset_hash,
    normalize_gold_record,
    stratified_split,
    validate_gold_records,
    write_json_artifact,
    write_jsonl,
)

ROOT = Path("data/evaluation/rag-benchmark")
DOCS = Path("docs/rag-benchmark")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/evaluation/gold-set-v1.jsonl"))
    parser.add_argument("--min-approved", type=int, default=140)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--allow-incomplete-preview", action="store_true")
    args = parser.parse_args()

    raw_records = read_jsonl(args.input)
    approved = [normalize_gold_record(row) for row in raw_records if row.get("review_status") == "approved"]
    validation = validate_gold_records(approved, strict_structured_claims=True)
    status = "READY_TO_FREEZE" if len(approved) >= args.min_approved and validation["valid"] else "BLOCKED"

    if status == "BLOCKED" and not args.allow_incomplete_preview:
        manifest = {
            "schema_version": "rag-gold-manifest-v1",
            "dataset_version": "rag-gold-v1",
            "status": status,
            "blocked_reason": "insufficient approved Gold records or validation errors",
            "created_at": datetime.now(UTC).isoformat(),
            "approved_count": len(approved),
            "min_approved": args.min_approved,
            "validation": validation,
            "freeze_performed": False,
        }
        write_json_artifact(ROOT / "gold-manifest-v1.json", manifest)
        DOCS.mkdir(parents=True, exist_ok=True)
        (DOCS / "gold-manifest-v1.md").write_text(
            "\n".join(
                [
                    "# RAG Gold manifest v1",
                    "",
                    f"- status: `{status}`",
                    f"- approved_count: {len(approved)}",
                    f"- min_approved: {args.min_approved}",
                    "- freeze_performed: `false`",
                    "",
                    "The final `rag-gold-v1` dev/test split was not created because the reviewed Gold count does not meet the Stage 1B quality gate or validation failed.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return 2

    dev, test = stratified_split(approved, seed=args.seed)
    for row in dev:
        row["split"] = "dev"
    for row in test:
        row["split"] = "test"
    full = sorted(dev + test, key=lambda row: str(row.get("question_id")))

    write_jsonl(ROOT / "gold-full-v1.jsonl", full)
    write_jsonl(ROOT / "gold-dev-v1.jsonl", dev)
    write_jsonl(ROOT / "gold-test-v1.jsonl", test)

    manifest = {
        "schema_version": "rag-gold-manifest-v1",
        "dataset_version": "rag-gold-v1",
        "status": "PREVIEW_INCOMPLETE" if args.allow_incomplete_preview and len(approved) < args.min_approved else status,
        "created_at": datetime.now(UTC).isoformat(),
        "seed": args.seed,
        "total": len(full),
        "dev_count": len(dev),
        "test_count": len(test),
        "answerable_count": sum(1 for row in full if row.get("answerable")),
        "unanswerable_count": sum(1 for row in full if not row.get("answerable")),
        "category_distribution": dict(sorted(Counter(str(row.get("category")) for row in full).items())),
        "difficulty_distribution": dict(sorted(Counter(str(row.get("difficulty")) for row in full).items())),
        "full_hash": dataset_hash(full),
        "dev_hash": dataset_hash(dev),
        "test_hash": dataset_hash(test),
        "reviewed_count": len(approved),
        "pending_count": sum(1 for row in raw_records if row.get("review_status") == "pending"),
        "validation": validation,
        "corpus_coverage": corpus_coverage(full),
        "freeze_performed": len(approved) >= args.min_approved and validation["valid"],
    }
    write_json_artifact(ROOT / "gold-manifest-v1.json", manifest)

    lines = [
        "# RAG Gold manifest v1",
        "",
        f"- status: `{manifest['status']}`",
        f"- dataset_version: `{manifest['dataset_version']}`",
        f"- total: {manifest['total']}",
        f"- dev_count: {manifest['dev_count']}",
        f"- test_count: {manifest['test_count']}",
        f"- full_hash: `{manifest['full_hash']}`",
        f"- dev_hash: `{manifest['dev_hash']}`",
        f"- test_hash: `{manifest['test_hash']}`",
        f"- freeze_performed: `{manifest['freeze_performed']}`",
    ]
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "gold-manifest-v1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if manifest["freeze_performed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
