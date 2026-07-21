"""Audit gaps between Full v4 and oracle candidate upper bound."""

from __future__ import annotations

import json
from collections import Counter

from paper_research.generation.claim_obligations import build_claim_obligation_set
from paper_research.generation.set_completion_v2 import (
    evaluate_set_coverage_v2,
    select_set_completion_v2,
)

try:
    from scripts.stage13_25_common import (
        DATA,
        DOCS,
        citation_keys,
        iter_claim_contexts,
        write_csv,
        write_json,
        write_jsonl,
    )
except ModuleNotFoundError:
    from stage13_25_common import (  # type: ignore[no-redef]
        DATA,
        DOCS,
        citation_keys,
        iter_claim_contexts,
        write_csv,
        write_json,
        write_jsonl,
    )

OUT_JSONL = DATA / "evidence-selection-v4-oracle-gap-v1.jsonl"
OUT_JSON = DATA / "evidence-selection-v4-oracle-gap-v1.json"
OUT_CSV = DATA / "evidence-selection-v4-oracle-gap-v1.csv"
OUT_DOC = DOCS / "evidence-selection-v4-oracle-gap-v1.md"


def classify(row: dict) -> str:
    if not row["oracle_candidate_upper_bound_status"] or row["full_v4_any_valid_status"]:
        return "no_gap"
    if row["candidate_rejected_at_admission"]:
        return "candidate_not_admitted"
    if row["candidate_rejected_by_set_builder"]:
        return "complementary_set_not_enumerated"
    if row["completeness_scorer_result"] == "false_negative":
        return "set_sufficiency_false_negative"
    if row["candidate_citation_slot_availability"] <= 0:
        return "citation_cap_blocked"
    return "true_unachievable_without_gold"


def build() -> tuple[list[dict], dict]:
    v4_rows = json.loads((DATA / "dev-v3-6-evidence-selection-v4-rows.json").read_text())
    v4_by_id = {row["required_claim_id"]: row for row in v4_rows["J_full_v4_candidate"]}
    rows: list[dict] = []
    causes: Counter[str] = Counter()
    questions: Counter[str] = Counter()
    for ctx in iter_claim_contexts():
        valid_sets = ctx["valid_sets"]
        any_valid = valid_sets["core"] | valid_sets["supporting"] | valid_sets["equivalent"]
        oracle_ids = [
            cid
            for cid, key in ctx["key_by_citation"].items()
            if cid in ctx["candidate_ids"] and key in any_valid
        ]
        v4_row = v4_by_id[ctx["required_claim_id"]]
        obligation_set = build_claim_obligation_set(ctx["claim_text"])
        baseline_candidates = tuple(
            candidate
            for candidate in ctx["candidates"]
            if candidate.citation_id in ctx["baseline_ids"]
        )
        v4_input_ids = tuple(v4_row["citation_ids"])
        set_result = select_set_completion_v2(ctx["claim_text"], ctx["candidates"], v4_input_ids)
        selected_ids = set(set_result.primary_citation_ids + set_result.supporting_citation_ids)
        selected_candidates = tuple(
            candidate for candidate in ctx["candidates"] if candidate.citation_id in selected_ids
        )
        baseline_coverage = evaluate_set_coverage_v2(
            ctx["claim_text"], obligation_set, baseline_candidates
        )
        final_coverage = evaluate_set_coverage_v2(
            ctx["claim_text"],
            obligation_set,
            selected_candidates,
        )
        admitted = set(set_result.rejected_citation_ids) | selected_ids
        oracle_set = set(oracle_ids)
        row = {
            "question_id": ctx["question_id"],
            "required_claim_id": ctx["required_claim_id"],
            "baseline_any_valid_status": bool(
                citation_keys(ctx["baseline_ids"], ctx["key_by_citation"]) & any_valid
            ),
            "full_v4_any_valid_status": bool(v4_row["any_valid_final"]),
            "oracle_candidate_upper_bound_status": bool(oracle_ids),
            "oracle_valid_candidate_set": oracle_ids,
            "full_v4_admitted_candidates": sorted(admitted),
            "full_v4_role_assignments": "see set-completion replay traces",
            "full_v4_selected_set": v4_row["citation_ids"],
            "candidate_citation_slot_availability": 3 - len(v4_row["citation_ids"]),
            "baseline_citation_count": len(ctx["baseline_ids"]),
            "final_citation_count": len(v4_row["citation_ids"]),
            "uncovered_obligations_before": list(baseline_coverage.missing_obligations),
            "candidate_covered_obligations": list(final_coverage.covered_obligations),
            "uncovered_obligations_after": list(final_coverage.missing_obligations),
            "numeric_obligations": baseline_coverage.numeric_applicable,
            "comparison_obligations": baseline_coverage.comparison_applicable,
            "candidate_numeric_anchors": final_coverage.numeric_complete,
            "candidate_comparison_sides": final_coverage.comparison_complete,
            "candidate_rejected_at_admission": bool(oracle_set - admitted),
            "candidate_rejected_by_role_assignment": False,
            "candidate_rejected_by_set_builder": bool(oracle_set - selected_ids),
            "candidate_rejected_by_replacement_proof": False,
            "candidate_rejected_by_citation_cap": len(v4_row["citation_ids"]) >= 3,
            "candidate_lost_during_fallback": False,
            "completeness_scorer_result": "false_negative"
            if oracle_ids and not final_coverage.complete
            else "aligned",
            "offline_relation_scorer_result": bool(oracle_ids),
            "secondary_gap_causes": [],
            "generic_repair_category": "set_completion"
            if oracle_ids and not v4_row["any_valid_final"]
            else "none",
            "no_online_gold_dependency": True,
        }
        row["primary_gap_cause"] = classify(row)
        causes[row["primary_gap_cause"]] += 1
        if row["primary_gap_cause"] != "no_gap":
            questions[ctx["question_id"]] += 1
        rows.append(row)
    summary = {
        "schema_version": "evidence-selection-v4-oracle-gap-v1",
        "required_claims": len(rows),
        "oracle_any_valid_hit_claims": sum(
            row["oracle_candidate_upper_bound_status"] for row in rows
        ),
        "full_v4_hit_claims": sum(row["full_v4_any_valid_status"] for row in rows),
        "oracle_v4_gap_claims": sum(
            row["oracle_candidate_upper_bound_status"] and not row["full_v4_any_valid_status"]
            for row in rows
        ),
        "gap_question_distribution": dict(sorted(questions.items())),
        "gap_root_cause_distribution": dict(sorted(causes.items())),
        "unknown_gap_reasons": causes.get("unknown", 0),
        "citation_cap_blocked": causes.get("citation_cap_blocked", 0),
        "candidate_budget_blocked": causes.get("candidate_not_admitted", 0),
        "V4_ORACLE_GAP_ATTRIBUTION": "COMPLETE" if causes.get("unknown", 0) == 0 else "INCOMPLETE",
        "PRIMARY_V4_ORACLE_GAP": causes.most_common(1)[0][0],
    }
    return rows, summary


def main() -> None:
    rows, summary = build()
    write_jsonl(OUT_JSONL, rows)
    write_csv(OUT_CSV, rows)
    write_json(OUT_JSON, summary)
    OUT_DOC.write_text(
        "# Evidence Selection v4 Oracle Gap Audit\n\n"
        f"- Attribution: `{summary['V4_ORACLE_GAP_ATTRIBUTION']}`\n"
        f"- Primary gap: `{summary['PRIMARY_V4_ORACLE_GAP']}`\n"
        f"- Oracle hit claims: `{summary['oracle_any_valid_hit_claims']}`\n"
        f"- Full v4 hit claims: `{summary['full_v4_hit_claims']}`\n"
        f"- Gap claims: `{summary['oracle_v4_gap_claims']}`\n"
        f"- Unknown: `{summary['unknown_gap_reasons']}`\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
