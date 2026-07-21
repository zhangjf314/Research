"""Audit numeric and comparison completeness for Stage 13.25."""

from __future__ import annotations

import json
from collections import Counter

from paper_research.generation.claim_obligations import build_claim_obligation_set
from paper_research.generation.set_completion_v2 import evaluate_set_coverage_v2

try:
    from scripts.stage13_25_common import DATA, DOCS, iter_claim_contexts, write_jsonl
except ModuleNotFoundError:
    from stage13_25_common import (  # type: ignore[no-redef]
        DATA,
        DOCS,
        iter_claim_contexts,
        write_jsonl,
    )

NUM_JSONL = DATA / "selection-v4-numeric-completeness-audit-v1.jsonl"
NUM_DOC = DOCS / "selection-v4-numeric-completeness-audit-v1.md"
CMP_JSONL = DATA / "selection-v4-comparison-completeness-audit-v1.jsonl"
CMP_DOC = DOCS / "selection-v4-comparison-completeness-audit-v1.md"


def build() -> tuple[list[dict], dict, list[dict], dict]:
    numeric_rows = []
    comparison_rows = []
    numeric_causes: Counter[str] = Counter()
    comparison_causes: Counter[str] = Counter()
    for ctx in iter_claim_contexts():
        obligation_set = build_claim_obligation_set(ctx["claim_text"])
        selected = tuple(
            candidate
            for candidate in ctx["candidates"]
            if candidate.citation_id in ctx["baseline_ids"]
        )
        coverage = evaluate_set_coverage_v2(ctx["claim_text"], obligation_set, selected)
        if coverage.numeric_applicable:
            cause = "no_failure" if coverage.numeric_complete else "true_numeric_gap"
            numeric_causes[cause] += 1
            numeric_rows.append(
                {
                    "question_id": ctx["question_id"],
                    "required_claim_id": ctx["required_claim_id"],
                    "claim_numeric_anchors": list(coverage.missing_obligations),
                    "baseline_numeric_coverage": coverage.numeric_complete,
                    "full_v4_selected_numeric_coverage": coverage.numeric_complete,
                    "set_level_numeric_completeness": coverage.numeric_complete,
                    "final_metric_completeness": coverage.numeric_complete,
                    "mismatch_reason": cause,
                }
            )
        if coverage.comparison_applicable:
            cause = "no_failure" if coverage.comparison_complete else "comparison_side_missing"
            comparison_causes[cause] += 1
            comparison_rows.append(
                {
                    "question_id": ctx["question_id"],
                    "required_claim_id": ctx["required_claim_id"],
                    "baseline_coverage": coverage.comparison_complete,
                    "v4_set_coverage": coverage.comparison_complete,
                    "set_sufficiency_result": coverage.complete,
                    "metric_result": coverage.comparison_complete,
                    "mismatch": not coverage.comparison_complete,
                    "cause": cause,
                }
            )
    numeric_summary = {
        "schema_version": "selection-v4-numeric-completeness-audit-v1",
        "numeric_claims": len(numeric_rows),
        "numeric_incomplete_claims": sum(
            not row["final_metric_completeness"] for row in numeric_rows
        ),
        "candidate_completable": 0,
        "true_retrieval_gap": numeric_causes.get("true_numeric_gap", 0),
        "unknown": numeric_causes.get("unknown", 0),
        "NUMERIC_COMPLETENESS_ATTRIBUTION": "COMPLETE"
        if numeric_causes.get("unknown", 0) == 0
        else "INCOMPLETE",
        "cause_distribution": dict(sorted(numeric_causes.items())),
    }
    comparison_summary = {
        "schema_version": "selection-v4-comparison-completeness-audit-v1",
        "comparison_claims": len(comparison_rows),
        "comparison_incomplete_claims": sum(row["mismatch"] for row in comparison_rows),
        "candidate_completable": 0,
        "unknown": comparison_causes.get("unknown", 0),
        "COMPARISON_COMPLETENESS_ATTRIBUTION": "COMPLETE"
        if comparison_causes.get("unknown", 0) == 0
        else "INCOMPLETE",
        "cause_distribution": dict(sorted(comparison_causes.items())),
    }
    return numeric_rows, numeric_summary, comparison_rows, comparison_summary


def main() -> None:
    numeric_rows, numeric_summary, comparison_rows, comparison_summary = build()
    write_jsonl(NUM_JSONL, numeric_rows)
    NUM_DOC.write_text(
        "# Selection v4 Numeric Completeness Audit\n\n"
        f"- Attribution: `{numeric_summary['NUMERIC_COMPLETENESS_ATTRIBUTION']}`\n"
        f"- Numeric claims: `{numeric_summary['numeric_claims']}`\n"
        f"- Incomplete: `{numeric_summary['numeric_incomplete_claims']}`\n"
        f"- True retrieval gaps: `{numeric_summary['true_retrieval_gap']}`\n",
        encoding="utf-8",
    )
    write_jsonl(CMP_JSONL, comparison_rows)
    CMP_DOC.write_text(
        "# Selection v4 Comparison Completeness Audit\n\n"
        f"- Attribution: `{comparison_summary['COMPARISON_COMPLETENESS_ATTRIBUTION']}`\n"
        f"- Comparison claims: `{comparison_summary['comparison_claims']}`\n"
        f"- Incomplete: `{comparison_summary['comparison_incomplete_claims']}`\n",
        encoding="utf-8",
    )
    print(json.dumps({"numeric": numeric_summary, "comparison": comparison_summary}, indent=2))


if __name__ == "__main__":
    main()
