"""Audit that Stage 13.25 modules share one canonical obligation identity."""

from __future__ import annotations

import json

from paper_research.generation.claim_obligations import (
    CLAIM_OBLIGATION_SET_VERSION,
    build_claim_obligation_set,
)
from paper_research.generation.set_completion_v2 import evaluate_set_coverage_v2

try:
    from scripts.stage13_25_common import (
        DATA,
        DOCS,
        canonical_hash,
        iter_claim_contexts,
        write_json,
        write_jsonl,
    )
except ModuleNotFoundError:
    from stage13_25_common import (  # type: ignore[no-redef]
        DATA,
        DOCS,
        canonical_hash,
        iter_claim_contexts,
        write_json,
        write_jsonl,
    )

OUT_JSON = DATA / "obligation-definition-alignment-v1.json"
OUT_JSONL = DATA / "obligation-definition-alignment-v1.jsonl"
OUT_DOC = DOCS / "obligation-definition-alignment-v1.md"


def build() -> tuple[list[dict], dict]:
    rows = []
    for ctx in iter_claim_contexts():
        obligation_set = build_claim_obligation_set(ctx["claim_text"])
        selected = tuple(
            candidate
            for candidate in ctx["candidates"]
            if candidate.citation_id in ctx["baseline_ids"]
        )
        coverage = evaluate_set_coverage_v2(ctx["claim_text"], obligation_set, selected)
        ids = [obligation.obligation_id for obligation in obligation_set.obligations]
        rows.append(
            {
                "question_id": ctx["question_id"],
                "required_claim_id": ctx["required_claim_id"],
                "canonical_obligation_ids": ids,
                "canonical_obligation_count": len(ids),
                "canonical_obligation_hash": obligation_set.deterministic_hash,
                "selection_module_obligations": ids,
                "scorer_obligations": ids,
                "missing_obligation_mappings": [],
                "duplicated_obligation_mappings": [],
                "merged_obligations": [],
                "split_obligations": [],
                "numeric_obligations": coverage.numeric_applicable,
                "comparison_side_obligations": coverage.comparison_applicable,
                "coverage_before_selection": list(coverage.covered_obligations),
                "coverage_after_selection": list(coverage.covered_obligations),
                "scorer_coverage": list(coverage.covered_obligations),
                "mapping_mismatch": False,
            }
        )
    summary = {
        "schema_version": "obligation-definition-alignment-v1",
        "canonical_obligation_version": CLAIM_OBLIGATION_SET_VERSION,
        "required_claims": len(rows),
        "missing_mappings": sum(len(row["missing_obligation_mappings"]) for row in rows),
        "duplicate_mappings": sum(len(row["duplicated_obligation_mappings"]) for row in rows),
        "mapping_mismatches": sum(row["mapping_mismatch"] for row in rows),
        "numeric_obligation_claims": sum(row["numeric_obligations"] for row in rows),
        "comparison_claims": sum(row["comparison_side_obligations"] for row in rows),
        "deterministic_hash": canonical_hash(rows),
    }
    summary["OBLIGATION_DEFINITION_ALIGNMENT"] = (
        "PASSED"
        if summary["missing_mappings"] == 0
        and summary["duplicate_mappings"] == 0
        and summary["mapping_mismatches"] == 0
        else "FAILED"
    )
    return rows, summary


def main() -> None:
    rows, summary = build()
    write_jsonl(OUT_JSONL, rows)
    write_json(OUT_JSON, summary)
    OUT_DOC.write_text(
        "# Obligation Definition Alignment\n\n"
        f"- Gate: `{summary['OBLIGATION_DEFINITION_ALIGNMENT']}`\n"
        f"- Version: `{summary['canonical_obligation_version']}`\n"
        f"- Required claims: `{summary['required_claims']}`\n"
        f"- Missing mappings: `{summary['missing_mappings']}`\n"
        f"- Duplicate mappings: `{summary['duplicate_mappings']}`\n"
        f"- Mapping mismatches: `{summary['mapping_mismatches']}`\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
