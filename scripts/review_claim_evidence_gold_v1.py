"""Human-only CLI for reviewing claim/evidence mappings."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "data/evaluation/claim-evidence-gold-v1.jsonl"
GOLD = ROOT / "data/evaluation/gold-set-v1.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=DEFAULT)
    parser.add_argument("--list-pending", action="store_true")
    parser.add_argument("--claim-id")
    parser.add_argument("--question-id")
    parser.add_argument("--category")
    parser.add_argument("--difficulty")
    parser.add_argument("--reviewer")
    parser.add_argument("--notes")
    parser.add_argument("--evidence-set", action="append", default=[])
    parser.add_argument("--alternative-set", action="append", default=[])
    parser.add_argument("--approve", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _parse_set(value: str) -> list[dict]:
    triples = []
    for item in value.split(","):
        paper_id, page, block_id = item.split(":", 2)
        triples.append({"paper_id": paper_id, "page": int(page), "block_id": block_id})
    return triples


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line) for line in args.path.read_text(encoding="utf-8").splitlines() if line
    ]
    gold = {
        item["question_id"]: item
        for item in (
            json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines() if line
        )
    }

    def matches_filters(row: dict) -> bool:
        source = gold.get(row["question_id"], {})
        return (
            (not args.question_id or row["question_id"] == args.question_id)
            and (not args.category or source.get("category") == args.category)
            and (not args.difficulty or source.get("difficulty") == args.difficulty)
        )

    if args.list_pending:
        for row in rows:
            if row["annotation_status"] == "pending" and matches_filters(row):
                print(f"{row['question_id']}\t{row['claim_id']}\t{row['claim_text']}")
        return
    if not args.claim_id:
        raise SystemExit("--claim-id is required for a review update")
    matches = [row for row in rows if row["claim_id"] == args.claim_id]
    if len(matches) != 1:
        raise SystemExit(f"expected one claim, found {len(matches)}")
    row = matches[0]
    if row["annotation_status"] == "approved" and not args.overwrite:
        raise SystemExit("reviewed row requires --overwrite")
    if not args.approve:
        raise SystemExit("only explicit --approve writes an approved status")
    if not args.reviewer or not args.notes:
        raise SystemExit("--reviewer and --notes are required")
    evidence_sets = [_parse_set(value) for value in args.evidence_set]
    if row["answerable"] and not evidence_sets:
        raise SystemExit("answerable claims require at least one explicit evidence set")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup = args.path.with_suffix(args.path.suffix + f".{timestamp}.bak")
    shutil.copy2(args.path, backup)
    row.update(
        evidence_sets=evidence_sets,
        acceptable_alternative_evidence=[
            _parse_set(value) for value in args.alternative_set
        ],
        annotation_status="approved",
        reviewer=args.reviewer,
        reviewed_at=datetime.now(UTC).isoformat(),
        review_notes=args.notes,
    )
    args.path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows),
        encoding="utf-8",
    )
    print(json.dumps({"updated": args.claim_id, "backup": str(backup)}))


if __name__ == "__main__":
    main()
