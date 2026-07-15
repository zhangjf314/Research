"""Human-only review tool for the Stage 13.1 Claim-Evidence Pilot."""

# ruff: noqa: E501 -- CLI diagnostics are intentionally explicit.

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from paper_research.evidence.schema import EvidenceUnit

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/evaluation"
PILOT = DATA / "claim-evidence-gold-pilot-v1.jsonl"
CLAIMS = DATA / "claim-units-v1.jsonl"
EVIDENCE = DATA / "evidence-corpus-v1.jsonl"
GOLD = DATA / "gold-set-v1.jsonl"
DECISIONS = {
    "approved",
    "rejected",
    "needs_source_inspection",
    "claim_ambiguous",
    "gold_incomplete",
    "parsing_error",
}
CLAIM_ROLES = {
    "identify",
    "define",
    "explain_method",
    "explain_mechanism",
    "compare",
    "report_result",
    "report_limitation",
    "synthesize",
    "verify_absence",
    "ambiguous",
}


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_set(value: str | None) -> list[list[dict[str, Any]]]:
    """Parse `paper:page:block,paper:page:block;...` into stable sets."""
    if not value:
        return []
    output = []
    for group in value.split(";"):
        triples = []
        for raw in group.split(","):
            parts = raw.strip().split(":", 2)
            if len(parts) != 3:
                raise ValueError("evidence triples must use paper_id:page:block_id")
            paper, page, block = parts
            triples.append({"paper_id": paper, "page": int(page), "block_id": block})
        output.append(sorted(triples, key=lambda x: (x["paper_id"], x["page"], x["block_id"])))
    return output


def source_hashes() -> dict[str, str]:
    return {
        "claim_units_sha256": digest(CLAIMS),
        "evidence_corpus_sha256": digest(EVIDENCE),
        "gold_sha256": digest(GOLD),
    }


def backup() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    target = PILOT.with_name(f"{PILOT.name}.{stamp}.bak")
    shutil.copy2(PILOT, target)
    return target


def write(rows: list[dict[str, Any]]) -> None:
    PILOT.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def validate_sets(
    evidence_sets: list[list[dict[str, Any]]],
    allowed_papers: set[str],
    evidence: dict[tuple[str, int, str], EvidenceUnit],
) -> None:
    for group in evidence_sets:
        if not group:
            raise ValueError("an evidence set cannot be empty")
        for item in group:
            triple = (item["paper_id"], item["page"], item["block_id"])
            if item["paper_id"] not in allowed_papers:
                raise ValueError(f"evidence paper is outside target papers: {item['paper_id']}")
            if triple not in evidence:
                raise ValueError(f"evidence triple does not exist: {triple}")


def invalidate_stale(rows: list[dict[str, Any]]) -> bool:
    current = source_hashes()
    changed = False
    for row in rows:
        if row["source_hashes"] == current:
            continue
        if row["annotation_status"] != "pending":
            row["annotation_status"] = "pending"
            row["decision"] = None
            row["reviewer"] = None
            row["reviewed_at"] = None
            row["review_notes"] = None
            row["approved_evidence_sets"] = []
            row["approved_alternative_evidence_sets"] = []
            row["claim_role_after_review"] = None
        row["source_hashes"] = current
        changed = True
    return changed


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--sample-id")
    value.add_argument("--decision", choices=sorted(DECISIONS))
    value.add_argument("--evidence-set")
    value.add_argument("--alternative-evidence-set")
    value.add_argument("--multi-block-required", choices=["true", "false"])
    value.add_argument("--claim-role", choices=sorted(CLAIM_ROLES))
    value.add_argument("--reviewer")
    value.add_argument("--notes")
    value.add_argument("--overwrite", action="store_true")
    value.add_argument("--list-pending", action="store_true")
    value.add_argument("--export", type=Path)
    return value


def main() -> None:
    args = parser().parse_args()
    rows = jsonl(PILOT)
    if invalidate_stale(rows):
        prior = backup()
        write(rows)
        raise SystemExit(
            f"Source hashes changed. Existing reviews were invalidated; backup={prior}. Re-run after inspection."
        )
    if args.list_pending:
        for row in rows:
            if row["annotation_status"] == "pending":
                print(
                    f"{row['pilot_sample_id']}\t{row['question_id']}\t{row['claim_role']}\t{row['claim_text']}"
                )
        return
    if not args.sample_id:
        raise SystemExit("--sample-id is required unless --list-pending is used")
    matches = [row for row in rows if row["pilot_sample_id"] == args.sample_id]
    if len(matches) != 1:
        raise SystemExit(f"sample not found or not unique: {args.sample_id}")
    row = matches[0]
    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        args.export.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        print(args.export)
        return
    if not args.decision or not args.reviewer or not args.notes:
        raise SystemExit("--decision, --reviewer, and --notes are required for review")
    if row["annotation_status"] != "pending" and not args.overwrite:
        raise SystemExit("sample is already reviewed; pass --overwrite after human confirmation")
    evidence_sets = parse_set(args.evidence_set)
    alternatives = parse_set(args.alternative_evidence_set)
    evidence_rows = [EvidenceUnit.model_validate(item) for item in jsonl(EVIDENCE)]
    evidence = {(item.paper_id, item.page, item.block_id): item for item in evidence_rows}
    allowed = set(row["target_papers"])
    validate_sets(evidence_sets, allowed, evidence)
    validate_sets(alternatives, allowed, evidence)
    role = args.claim_role or row["claim_role"]
    if args.decision == "approved" and role != "verify_absence" and not evidence_sets:
        raise SystemExit("approved requires at least one valid --evidence-set")
    prior = backup()
    row["annotation_status"] = args.decision
    row["decision"] = args.decision
    row["approved_evidence_sets"] = evidence_sets
    row["approved_alternative_evidence_sets"] = alternatives
    row["multi_block_required"] = (
        args.multi_block_required == "true"
        if args.multi_block_required is not None
        else row["multi_block_required"]
    )
    row["claim_role_after_review"] = role
    row["reviewer"] = args.reviewer
    row["reviewed_at"] = datetime.now(UTC).isoformat()
    row["review_notes"] = args.notes
    row["review_history"] = [
        *row.get("review_history", []),
        {
            "claim_role_before": row["claim_role"],
            "claim_role_after": role,
            "decision": args.decision,
            "reviewer": args.reviewer,
            "reviewed_at": row["reviewed_at"],
            "notes": args.notes,
        },
    ]
    write(rows)
    print(
        json.dumps({"sample_id": args.sample_id, "decision": args.decision, "backup": str(prior)})
    )


if __name__ == "__main__":
    main()
