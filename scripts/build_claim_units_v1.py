"""Derive ClaimUnit v1 records without modifying approved Gold claims."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from paper_research.evidence.claims import build_claim_units

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data/evaluation/gold-set-v1.jsonl"
RETRIEVAL = ROOT / "data/evaluation/retrieval-gold-v2.jsonl"
OUTPUT = ROOT / "data/evaluation/claim-units-v1.jsonl"
MANIFEST = ROOT / "data/evaluation/claim-units-v1-manifest.json"
REPORT = ROOT / "docs/claim-units-v1.md"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    gold = _jsonl(GOLD)
    retrieval = {row["question_id"]: row for row in _jsonl(RETRIEVAL)}
    units = build_claim_units(gold, retrieval)
    text = "".join(
        json.dumps(unit.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n"
        for unit in units
    )
    OUTPUT.write_text(text, encoding="utf-8")
    roles = Counter(unit.claim_role for unit in units)
    types = Counter(unit.question_type for unit in units)
    signature = hashlib.sha256(text.encode()).hexdigest()
    manifest = {
        "schema_version": "claim-units-manifest-v1",
        "source_gold": "data/evaluation/gold-set-v1.jsonl",
        "source_retrieval_protocol": "data/evaluation/retrieval-gold-v2.jsonl",
        "question_count": len(gold),
        "claim_count": len(units),
        "answerable_claim_count": sum(unit.expected_answerability for unit in units),
        "unanswerable_claim_count": sum(not unit.expected_answerability for unit in units),
        "claim_role_distribution": dict(sorted(roles.items())),
        "question_type_distribution": dict(sorted(types.items())),
        "claim_level_gold_mapping_status": "pending_human_review",
        "build_signature": signature,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    REPORT.write_text(
        "\n".join(
            [
                "# Claim Units v1",
                "",
                "Derived from verbatim approved required claims. Unanswerable records receive "
                "one explicit verify-absence obligation derived from the original question.",
                "",
                f"- Questions: {len(gold)}",
                f"- Claim units: {len(units)}",
                f"- Answerable claims: {manifest['answerable_claim_count']}",
                f"- Unanswerable obligations: {manifest['unanswerable_claim_count']}",
                f"- Signature: `{signature}`",
                "",
                "Question-level Gold blocks are preserved as candidates, not asserted as "
                "approved claim-level mappings. `multi_block_required` remains null until review.",
                "",
                f"Claim roles: `{dict(sorted(roles.items()))}`",
                "",
                f"Question types: `{dict(sorted(types.items()))}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"claims": len(units), "build_signature": signature}))


if __name__ == "__main__":
    main()
