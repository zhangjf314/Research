"""Create a pending-only claim/evidence annotation draft."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLAIMS = ROOT / "data/evaluation/claim-units-v1.jsonl"
OUTPUT = ROOT / "data/evaluation/claim-evidence-gold-v1.jsonl"


def main() -> None:
    claims = [
        json.loads(line) for line in CLAIMS.read_text(encoding="utf-8").splitlines() if line
    ]
    rows = []
    for claim in claims:
        rows.append(
            {
                "question_id": claim["question_id"],
                "claim_id": claim["claim_id"],
                "claim_text": claim["claim_text"],
                "answerable": claim["expected_answerability"],
                "evidence_sets": [],
                "acceptable_alternative_evidence": [],
                "negative_evidence": [],
                "annotation_status": "pending",
                "reviewer": None,
                "reviewed_at": None,
                "review_notes": None,
                "source_version": "claim-unit-v1+evidence-unit-v1",
                "source_claim_fingerprint": claim["claim_id"],
                "candidate_gold_block_ids": claim["gold_block_ids"],
                "candidate_gold_pages": claim["gold_pages"],
                "page_level_only_insufficient": True,
            }
        )
    if OUTPUT.exists():
        existing = [
            json.loads(line)
            for line in OUTPUT.read_text(encoding="utf-8").splitlines()
            if line
        ]
        existing_by_id = {row["claim_id"]: row for row in existing}
        for index, row in enumerate(rows):
            previous = existing_by_id.get(row["claim_id"])
            if not previous or previous.get("annotation_status") != "approved":
                continue
            unchanged = (
                previous.get("source_claim_fingerprint") == row["source_claim_fingerprint"]
                and previous.get("source_version") == row["source_version"]
            )
            if unchanged:
                rows[index] = previous
    OUTPUT.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "records": len(rows),
                "pending": sum(row["annotation_status"] == "pending" for row in rows),
                "approved": sum(row["annotation_status"] == "approved" for row in rows),
            }
        )
    )


if __name__ == "__main__":
    main()
