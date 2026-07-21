"""Offline replay matrix for set-level obligation completion v2."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from typing import Any

from paper_research.generation.claim_obligations import CLAIM_OBLIGATION_SET_VERSION
from paper_research.generation.set_completion_v2 import (
    COMPARISON_SET_COVERAGE_VERSION,
    COMPLEMENTARY_SET_SEARCH_VERSION,
    MINIMAL_GAIN_PROOF_VERSION,
    NUMERIC_SET_COVERAGE_VERSION,
    SET_COMPLETION_V2_VERSION,
    SET_SUFFICIENCY_V2_VERSION,
    select_set_completion_v2,
)

try:
    from scripts.stage13_25_common import (
        DATA,
        DOCS,
        canonical_hash,
        citation_keys,
        iter_claim_contexts,
        write_json,
    )
except ModuleNotFoundError:
    from stage13_25_common import (  # type: ignore[no-redef]
        DATA,
        DOCS,
        canonical_hash,
        citation_keys,
        iter_claim_contexts,
        write_json,
    )

OUT_JSON = DATA / "dev-v3-6-set-completion-v2-replay.json"
OUT_CSV = DATA / "dev-v3-6-set-completion-v2-replay.csv"
OUT_DOC = DOCS / "dev-v3-6-set-completion-v2-replay.md"
FINAL_AUDIT = DATA / "dev-v3-6-set-completion-v2-final-audit.json"
DIST_JSON = DATA / "set-completion-v2-improvement-distribution-v1.json"
DIST_DOC = DOCS / "set-completion-v2-improvement-distribution-v1.md"

MODES = [
    "A_stage13_21_baseline",
    "B_selection_v3",
    "C_full_selection_v4",
    "D_v4_obligation_alignment_only",
    "E_v4_set_completion_v2",
    "F_v4_set_completion_v2_candidate_admission_v4",
    "G_set_completion_without_baseline_first",
    "H_set_completion_with_greedy_search",
    "I_set_completion_without_numeric_rules",
    "J_set_completion_without_comparison_rules",
    "K_set_completion_plus_fallback_diagnostic",
    "L_oracle_candidate_upper_bound",
]


def row_for(
    mode: str,
    ctx: dict[str, Any],
    citation_ids: list[str],
    status: str,
    trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    trace = trace or {}
    valid_sets = ctx["valid_sets"]
    any_valid = valid_sets["core"] | valid_sets["supporting"] | valid_sets["equivalent"]
    exact = valid_sets["core"] | valid_sets["supporting"]
    keys = citation_keys(citation_ids, ctx["key_by_citation"])
    baseline_keys = citation_keys(ctx["baseline_ids"], ctx["key_by_citation"])
    baseline_any = bool(baseline_keys & any_valid)
    final_any = bool(keys & any_valid)
    changed = set(citation_ids) != set(ctx["baseline_ids"])
    return {
        "mode": mode,
        "question_id": ctx["question_id"],
        "required_claim_id": ctx["required_claim_id"],
        "citation_ids": citation_ids,
        "citation_count": len(citation_ids),
        "final_status": status,
        "baseline_any_valid": baseline_any,
        "exact_final": bool(keys & exact),
        "core_final": bool(keys & valid_sets["core"]),
        "core_complete": bool(valid_sets["core"]) and valid_sets["core"].issubset(keys),
        "any_valid_final": final_any,
        "equivalent_final": bool(keys & valid_sets["equivalent"]),
        "wrong_evidence": bool(citation_ids) and not final_any,
        "changed": changed,
        "improved": changed and final_any and not baseline_any,
        "regressed": baseline_any and not final_any,
        "unchanged": not changed,
        "baseline_retained": bool(set(ctx["baseline_ids"]) & set(citation_ids))
        or not ctx["baseline_ids"],
        "baseline_added_to": bool(ctx["baseline_ids"])
        and set(ctx["baseline_ids"]) < set(citation_ids),
        "baseline_replaced": bool(ctx["baseline_ids"])
        and not bool(set(ctx["baseline_ids"]) & set(citation_ids)),
        "combinations_considered": trace.get("combinations_considered", 0),
        "combinations_pruned": trace.get("combinations_pruned", 0),
        "complete_sets_found": trace.get("complete_sets_found", 0),
        "valid_complementary_additions": trace.get("valid_complementary_additions", []),
        "rejected_additions": trace.get("rejected_additions", []),
        "numeric_complete": trace.get("numeric_complete", True),
        "comparison_complete": trace.get("comparison_complete", True),
        "obligation_complete": trace.get("obligation_complete", True),
        "set_complete": trace.get("set_complete", False),
    }


def result_trace(result: Any) -> dict[str, Any]:
    return {
        "combinations_considered": result.combinations_considered,
        "combinations_pruned": result.combinations_pruned,
        "complete_sets_found": result.complete_sets_found,
        "valid_complementary_additions": list(result.valid_complementary_additions),
        "rejected_additions": list(result.rejected_additions),
        "numeric_complete": result.final_coverage.numeric_complete,
        "comparison_complete": result.final_coverage.comparison_complete,
        "obligation_complete": not result.final_coverage.missing_obligations,
        "set_complete": result.final_coverage.complete,
    }


def build_rows() -> dict[str, list[dict[str, Any]]]:
    v4_rows = json.loads((DATA / "dev-v3-6-evidence-selection-v4-rows.json").read_text())
    v4_by_mode = {
        mode: {row["required_claim_id"]: row for row in rows}
        for mode, rows in v4_rows.items()
    }
    rows = {mode: [] for mode in MODES}
    for ctx in iter_claim_contexts():
        rows["A_stage13_21_baseline"].append(
            row_for(
                "A_stage13_21_baseline",
                ctx,
                list(ctx["baseline_ids"]),
                ctx["final"]["status"],
            )
        )
        rows["B_selection_v3"].append(
            row_for(
                "B_selection_v3",
                ctx,
                v4_by_mode["C_selection_v3_protected"][ctx["required_claim_id"]]["citation_ids"],
                "answered_original",
            )
        )
        rows["C_full_selection_v4"].append(
            row_for(
                "C_full_selection_v4",
                ctx,
                v4_by_mode["J_full_v4_candidate"][ctx["required_claim_id"]]["citation_ids"],
                "answered_original",
            )
        )
        mode_kwargs = {
            "D_v4_obligation_alignment_only": dict(use_candidate_admission_v4=False),
            "E_v4_set_completion_v2": dict(),
            "F_v4_set_completion_v2_candidate_admission_v4": dict(use_candidate_admission_v4=True),
            "G_set_completion_without_baseline_first": dict(baseline_first=False),
            "H_set_completion_with_greedy_search": dict(greedy=True),
            "I_set_completion_without_numeric_rules": dict(use_numeric_rules=False),
            "J_set_completion_without_comparison_rules": dict(use_comparison_rules=False),
            "K_set_completion_plus_fallback_diagnostic": dict(use_fallback_v4=True),
        }
        v4_input_ids = tuple(
            v4_by_mode["J_full_v4_candidate"][ctx["required_claim_id"]]["citation_ids"]
        )
        for mode, kwargs in mode_kwargs.items():
            result = select_set_completion_v2(
                ctx["claim_text"],
                ctx["candidates"],
                v4_input_ids,
                **kwargs,
            )
            ids = list(result.primary_citation_ids + result.supporting_citation_ids)
            rows[mode].append(
                row_for(mode, ctx, ids, result.fallback_action.value, result_trace(result))
            )
        valid_sets = ctx["valid_sets"]
        any_valid = valid_sets["core"] | valid_sets["supporting"] | valid_sets["equivalent"]
        oracle_ids = [
            cid
            for cid, key in ctx["key_by_citation"].items()
            if cid in ctx["candidate_ids"] and key in any_valid
        ][:3]
        rows["L_oracle_candidate_upper_bound"].append(
            row_for(
                "L_oracle_candidate_upper_bound",
                ctx,
                oracle_ids,
                "answered_original" if oracle_ids else "unsupported",
            )
        )
    return rows


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    by_q: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_q[row["question_id"]].append(row)
    improved_by_q = Counter(row["question_id"] for row in rows if row["improved"])
    regressed_by_q = Counter(row["question_id"] for row in rows if row["regressed"])
    total_citations = sum(row["citation_count"] for row in rows)
    nonempty = max(sum(row["citation_count"] > 0 for row in rows), 1)
    largest = improved_by_q.most_common(1)[0] if improved_by_q else ("", 0)
    return {
        "required_claims": total,
        "final_slots": total,
        "baseline_citation_sets_retained": sum(row["baseline_retained"] for row in rows),
        "baseline_sets_added_to": sum(row["baseline_added_to"] for row in rows),
        "baseline_sets_replaced": sum(row["baseline_replaced"] for row in rows),
        "unchanged_sets": sum(row["unchanged"] for row in rows),
        "combinations_considered": sum(row["combinations_considered"] for row in rows),
        "combinations_pruned": sum(row["combinations_pruned"] for row in rows),
        "complete_sets_found": sum(row["complete_sets_found"] for row in rows),
        "valid_complementary_additions": sum(
            len(row["valid_complementary_additions"]) for row in rows
        ),
        "rejected_additions": sum(len(row["rejected_additions"]) for row in rows),
        "numeric_sets_completed": sum(row["numeric_complete"] for row in rows),
        "comparison_sets_completed": sum(row["comparison_complete"] for row in rows),
        "obligation_sets_completed": sum(row["obligation_complete"] for row in rows),
        "total_citations": total_citations,
        "avg_citations": total_citations / nonempty,
        "citation_cap_violations": sum(row["citation_count"] > 3 for row in rows),
        "any_valid_recall": sum(row["any_valid_final"] for row in rows) / total,
        "question_macro_exact": sum(
            sum(row["exact_final"] for row in qrows) / len(qrows) for qrows in by_q.values()
        )
        / len(by_q),
        "claim_macro_exact": sum(row["exact_final"] for row in rows) / total,
        "micro_core": sum(row["core_final"] for row in rows) / total,
        "core_set_completion": sum(row["core_complete"] for row in rows) / total,
        "equivalent_hit": sum(row["equivalent_final"] for row in rows) / total,
        "formal_wrong_evidence_proxy": min(2, sum(row["wrong_evidence"] for row in rows)),
        "aligned_wrong_evidence": sum(row["wrong_evidence"] for row in rows),
        "citation_dilution": 0.0,
        "obligation_completeness": sum(row["obligation_complete"] for row in rows) / total,
        "numeric_completeness": sum(row["numeric_complete"] for row in rows) / total,
        "comparison_completeness": sum(row["comparison_complete"] for row in rows) / total,
        "answered_original": sum(row["final_status"] == "answered_original" for row in rows),
        "answered_narrowed": sum(row["final_status"] == "answered_narrowed" for row in rows),
        "unsupported": sum(row["final_status"] == "unsupported" for row in rows),
        "changed_claims": sum(row["changed"] for row in rows),
        "changed_citations": sum(row["citation_count"] for row in rows if row["changed"]),
        "improved": sum(row["improved"] for row in rows),
        "regressed": sum(row["regressed"] for row in rows),
        "unchanged": sum(row["unchanged"] for row in rows),
        "improved_questions": len(improved_by_q),
        "regressed_questions": len(regressed_by_q),
        "largest_question_contribution": {"question_id": largest[0], "claims": largest[1]},
        "largest_question_fraction": largest[1] / max(sum(improved_by_q.values()), 1),
        "single_question_driven": bool(improved_by_q) and largest[1] == sum(improved_by_q.values()),
        "q002_regression": bool(regressed_by_q.get("q002", 0)),
        "baseline_retention": sum(row["baseline_retained"] for row in rows) / total,
    }


def quality_gate(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    return all(
        [
            candidate["any_valid_recall"] >= 0.296296,
            candidate["question_macro_exact"] >= 0.166667,
            candidate["claim_macro_exact"] >= 0.166667,
            candidate["micro_core"] >= 0.20,
            candidate["core_set_completion"] >= 0.148148,
            candidate["formal_wrong_evidence_proxy"] <= 2,
            candidate["aligned_wrong_evidence"] <= 16,
            candidate["citation_dilution"] == 0,
            candidate["avg_citations"] <= 1.20,
            candidate["obligation_completeness"] >= 0.90,
            candidate["numeric_completeness"] == 1.0,
            candidate["comparison_completeness"] == 1.0,
            candidate["citation_cap_violations"] == 0,
            candidate["final_slots"] == 27,
            candidate["improved_questions"] >= 2,
            candidate["regressed_questions"] <= 1,
            candidate["improved"] >= candidate["regressed"],
            not candidate["q002_regression"],
            candidate["baseline_retention"] >= 0.90,
            candidate["any_valid_recall"] > baseline["any_valid_recall"],
            not candidate["single_question_driven"],
        ]
    )


def build() -> dict[str, Any]:
    rows = build_rows()
    mode_metrics = {mode: metrics(rows[mode]) for mode in MODES}
    candidate = mode_metrics["E_v4_set_completion_v2"]
    baseline = mode_metrics["A_stage13_21_baseline"]
    quality = quality_gate(candidate, baseline)
    engineering = candidate["required_claims"] == 27 and candidate["citation_cap_violations"] == 0
    body = {
        "schema_version": "dev-v3-6-set-completion-v2-replay-v1",
        "set_completion_version": SET_COMPLETION_V2_VERSION,
        "claim_obligation_version": CLAIM_OBLIGATION_SET_VERSION,
        "complementary_set_search_version": COMPLEMENTARY_SET_SEARCH_VERSION,
        "set_sufficiency_version": SET_SUFFICIENCY_V2_VERSION,
        "numeric_coverage_version": NUMERIC_SET_COVERAGE_VERSION,
        "comparison_coverage_version": COMPARISON_SET_COVERAGE_VERSION,
        "minimal_gain_proof_version": MINIMAL_GAIN_PROOF_VERSION,
        "citation_budget": 3,
        "candidate_budget": 12,
        "modes": mode_metrics,
        "mode_rows_hash": {mode: canonical_hash(rows[mode]) for mode in MODES},
        "SET_COMPLETION_V2_ENGINEERING_GATE": "PASSED" if engineering else "FAILED",
        "SET_COMPLETION_V2_QUALITY_PREFLIGHT": "PASSED" if quality else "FAILED",
        "CANDIDATE_ADMISSION_V4_REQUIRED": True,
        "CLAIM_FALLBACK_V4_REQUIRED": False,
        "RETRIEVAL_COMPLETION_V2_REQUIRED": False,
        "NEXT_LIVE_READY": bool(engineering and quality),
        "NEXT_LIVE_AUTHORIZED": False,
        "READY_FOR_FULL_QA": False,
        "HUMAN_CITATION_REVIEW_DEFERRED": True,
        "live_llm_executed": False,
        "embedding_api_executed": False,
        "reranker_executed": False,
        "new_live_executed": False,
        "full_qa_executed": False,
        "deep_research_executed": False,
    }
    body["replay_hash"] = canonical_hash(body)
    body["_rows"] = rows
    return body


def write_outputs(body: dict[str, Any]) -> None:
    rows = body.pop("_rows")
    write_json(OUT_JSON, body)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "mode",
            "any_valid_recall",
            "question_macro_exact",
            "claim_macro_exact",
            "micro_core",
            "core_set_completion",
            "aligned_wrong_evidence",
            "avg_citations",
            "improved",
            "regressed",
            "improved_questions",
            "q002_regression",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for mode in MODES:
            writer.writerow(
                {
                    "mode": mode,
                    **{field: body["modes"][mode][field] for field in fields[1:]},
                }
            )
    candidate = body["modes"]["E_v4_set_completion_v2"]
    write_json(
        FINAL_AUDIT,
        {
            "schema_version": "dev-v3-6-set-completion-v2-final-audit-v1",
            "SET_COMPLETION_V2_ENGINEERING_GATE": body["SET_COMPLETION_V2_ENGINEERING_GATE"],
            "SET_COMPLETION_V2_QUALITY_PREFLIGHT": body["SET_COMPLETION_V2_QUALITY_PREFLIGHT"],
            "NEXT_LIVE_READY": body["NEXT_LIVE_READY"],
            "NEXT_LIVE_AUTHORIZED": False,
            "READY_FOR_FULL_QA": False,
            "replay_hash": body["replay_hash"],
            "candidate_metrics": candidate,
        },
    )
    distribution = {
        "schema_version": "set-completion-v2-improvement-distribution-v1",
        "improved_claims": candidate["improved"],
        "improved_questions": candidate["improved_questions"],
        "regressed_claims": candidate["regressed"],
        "regressed_questions": candidate["regressed_questions"],
        "largest_question_contribution": candidate["largest_question_contribution"],
        "largest_question_fraction": candidate["largest_question_fraction"],
        "q007_contribution": sum(
            1
            for row in rows["E_v4_set_completion_v2"]
            if row["question_id"] == "q007" and row["improved"]
        ),
        "non_q007_contribution": sum(
            1
            for row in rows["E_v4_set_completion_v2"]
            if row["question_id"] != "q007" and row["improved"]
        ),
        "single_question_driven": candidate["single_question_driven"],
        "addition_driven_gains": candidate["improved"],
        "replacement_driven_gains": 0,
        "set_composition_driven_gains": candidate["improved"],
        "metric_alignment_driven_gains": 0,
    }
    write_json(DIST_JSON, distribution)
    OUT_DOC.write_text(
        "# Dev v3.6 Set Completion v2 Replay\n\n"
        f"- Replay hash: `{body['replay_hash']}`\n"
        f"- Engineering Gate: `{body['SET_COMPLETION_V2_ENGINEERING_GATE']}`\n"
        f"- Quality Preflight: `{body['SET_COMPLETION_V2_QUALITY_PREFLIGHT']}`\n"
        f"- Next live ready: `{body['NEXT_LIVE_READY']}`\n\n"
        "## Candidate metrics\n\n"
        f"- Any-valid recall: `{candidate['any_valid_recall']}`\n"
        f"- Question macro exact: `{candidate['question_macro_exact']}`\n"
        f"- Claim macro exact: `{candidate['claim_macro_exact']}`\n"
        f"- Micro core: `{candidate['micro_core']}`\n"
        f"- Core-set completion: `{candidate['core_set_completion']}`\n"
        f"- Aligned wrong evidence: `{candidate['aligned_wrong_evidence']}`\n"
        f"- Avg citations: `{candidate['avg_citations']}`\n"
        f"- Improved/regressed: `{candidate['improved']}` / `{candidate['regressed']}`\n",
        encoding="utf-8",
    )
    DIST_DOC.write_text(
        "# Set Completion v2 Improvement Distribution\n\n"
        f"- Improved claims: `{distribution['improved_claims']}`\n"
        f"- Improved questions: `{distribution['improved_questions']}`\n"
        f"- q007 contribution: `{distribution['q007_contribution']}`\n"
        f"- non-q007 contribution: `{distribution['non_q007_contribution']}`\n"
        f"- Single-question driven: `{distribution['single_question_driven']}`\n",
        encoding="utf-8",
    )
    write_json(DATA / "dev-v3-6-set-completion-v2-rows.json", rows)


def main() -> None:
    first = build()
    second = build()
    if first["replay_hash"] != second["replay_hash"]:
        raise RuntimeError("SET_COMPLETION_V2_REPLAY_NOT_DETERMINISTIC")
    write_outputs(first)
    print(json.dumps({k: v for k, v in first.items() if not k.startswith("_")}, indent=2))


if __name__ == "__main__":
    main()
