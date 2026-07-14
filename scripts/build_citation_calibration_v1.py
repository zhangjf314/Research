"""Build a representative pending-only citation calibration sample."""

from __future__ import annotations

import json
from collections import Counter, defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS = ROOT / "data/evaluation/qa-context-diagnostics-v1.json"
GOLD = ROOT / "data/evaluation/gold-set-v1.jsonl"
EVIDENCE = ROOT / "data/evaluation/evidence-corpus-v1.jsonl"
OUTPUT = ROOT / "data/evaluation/citation-calibration-v1.jsonl"
GUIDE = ROOT / "docs/citation-calibration-v1-guide.md"

STRATUM_MAP = {
    "exact_gold_block": "exact_gold",
    "same_gold_page": "same_page_non_exact",
    "semantic_support_non_gold": "semantic_support_signal",
    "weakly_related": "automated_unsupported_or_weakly_related",
    "unsupported": "automated_unsupported_or_weakly_related",
    "adjacent_to_gold_block": "automated_unsupported_or_weakly_related",
}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _balanced_take(rows: list[dict], count: int) -> list[dict]:
    buckets: dict[tuple[str, str], deque[dict]] = defaultdict(deque)
    for row in sorted(
        rows,
        key=lambda item: (
            item["category"], item["difficulty"], item["question_id"],
            item["claim_id"], item["cited_evidence"]["block_id"],
        ),
    ):
        buckets[(row["category"], row["difficulty"])].append(row)
    output = []
    keys = sorted(buckets)
    while len(output) < count and any(buckets.values()):
        for key in keys:
            if buckets[key] and len(output) < count:
                output.append(buckets[key].popleft())
    return output


def main() -> None:
    diagnostics = json.loads(DIAGNOSTICS.read_text(encoding="utf-8"))
    gold = {row["question_id"]: row for row in _jsonl(GOLD)}
    evidence_rows = _jsonl(EVIDENCE)
    evidence = {(row["paper_id"], row["block_id"]): row for row in evidence_rows}
    grouped = defaultdict(list)
    for run in diagnostics["runs"]:
        if run.get("context_mode") != "retrieved" or run.get("oracle"):
            continue
        question_id = run["question_id"]
        gold_row = gold[question_id]
        claims = {row["claim_id"]: row for row in run["answer"].get("claims", [])}
        context_rank = {
            block_id: rank
            for rank, context in enumerate(run["context"], 1)
            for block_id in context.get("block_ids", [])
        }
        for detail in run["diagnostics"]["citation_details"]:
            claim = claims.get(detail["claim_id"], {})
            for citation in detail["citations"]:
                classification = citation["classification"]
                stratum = STRATUM_MAP.get(classification)
                if not stratum:
                    continue
                key = (citation["paper_id"], citation["block_id"])
                unit = evidence.get(key)
                previous = (
                    evidence.get((citation["paper_id"], unit.get("previous_block_id")))
                    if unit and unit.get("previous_block_id") else None
                )
                following = (
                    evidence.get((citation["paper_id"], unit.get("next_block_id")))
                    if unit and unit.get("next_block_id") else None
                )
                grouped[stratum].append(
                    {
                        "question_id": question_id,
                        "claim_id": detail["claim_id"],
                        "claim_text": claim.get("text", ""),
                        "cited_evidence": {
                            "paper_id": citation["paper_id"],
                            "page": citation["page"],
                            "block_id": citation["block_id"],
                            "text": unit.get("text", "") if unit else "",
                            "block_type": unit.get("block_type") if unit else None,
                            "evidence_roles": unit.get("evidence_roles", []) if unit else [],
                        },
                        "adjacent_context": {
                            "previous": previous.get("text") if previous else None,
                            "next": following.get("text") if following else None,
                        },
                        "gold_evidence": {
                            "paper_ids": gold_row["gold_paper_ids"],
                            "pages": gold_row["gold_pages"],
                            "block_ids": gold_row["gold_block_ids"],
                        },
                        "automated_labels": {
                            "classification": classification,
                            "semantic_score": citation.get("semantic_score"),
                            "exact_gold": classification == "exact_gold_block",
                            "same_gold_page": classification == "same_gold_page",
                            "semantic_signal": classification == "semantic_support_non_gold",
                            "unsupported_or_weak": classification
                            in {"unsupported", "weakly_related", "adjacent_to_gold_block"},
                        },
                        "stratum": stratum,
                        "category": gold_row["category"],
                        "difficulty": gold_row["difficulty"],
                        "answerable": gold_row["answerable"],
                        "retrieval_scope": run.get("retrieval_scope", "paper"),
                        "context_rank": context_rank.get(citation["block_id"]),
                        "block_type": unit.get("block_type") if unit else None,
                        "human_review_status": "pending",
                        "human_label": None,
                        "reviewer": None,
                        "reviewed_at": None,
                        "review_notes": None,
                    }
                )
    selected = []
    for stratum in (
        "exact_gold", "same_page_non_exact", "semantic_support_signal",
        "automated_unsupported_or_weakly_related",
    ):
        unique = {}
        for row in grouped[stratum]:
            key = (
                row["question_id"], row["claim_id"],
                row["cited_evidence"]["paper_id"], row["cited_evidence"]["block_id"],
            )
            unique.setdefault(key, row)
        if len(unique) < 15:
            raise RuntimeError(f"insufficient unique calibration rows for {stratum}: {len(unique)}")
        selected.extend(_balanced_take(list(unique.values()), 15))
    selected.sort(key=lambda row: (row["stratum"], row["question_id"], row["claim_id"]))
    for index, row in enumerate(selected, 1):
        row["sample_id"] = f"citation-calibration-v1-{index:03d}"
    OUTPUT.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    distribution = Counter(row["stratum"] for row in selected)
    GUIDE.write_text(
        """# Citation Calibration v1 Guide

This is a deterministic 60-pair representative calibration draft with 15 rows in each of four
automated strata. Category and difficulty are round-robin balanced where candidates permit.
All human fields are pending/null. No automated label is a human conclusion.

Review the atomic claim against the exact cited block and its adjacent context. Page proximity,
adjacency and token overlap are not sufficient support. Use one label:

- `fully_supported`
- `partially_supported`
- `related_but_insufficient`
- `unsupported`
- `gold_annotation_too_narrow`
- `ambiguous_claim`
- `malformed_evidence`

Reviewer, reviewed_at and review_notes are mandatory for approval. This is not an independent
double-blind review unless a separately documented review protocol actually establishes that fact.
Until all 60 rows are reviewed, no formal human citation precision or v1 citation pass may be
reported.
""",
        encoding="utf-8",
    )
    print(json.dumps({"records": len(selected), "pending": len(selected), "strata": distribution}))


if __name__ == "__main__":
    main()
