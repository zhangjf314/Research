"""Attribute recall collapse in Evidence Selection v3.

This is an offline diagnostic. It reads frozen Stage 13.21/13.23 artifacts and
claim-level labels only for scoring and attribution; it does not call providers
or mutate historical results.
"""

from __future__ import annotations

import json
from collections import Counter

from paper_research.generation.citation_selection import CitationCandidate
from paper_research.generation.evidence_selection_v3 import select_evidence_v3

try:
    from scripts.stage13_23_common import (
        DATA,
        DOCS,
        RUN_ROOT,
        candidate_rows,
        canonical_hash,
        final_slot,
        load_gold,
        registry_maps,
        relation_sets,
        selected_runs,
        write_json,
    )
except ModuleNotFoundError:
    from stage13_23_common import (  # type: ignore[no-redef]
        DATA,
        DOCS,
        RUN_ROOT,
        candidate_rows,
        canonical_hash,
        final_slot,
        load_gold,
        registry_maps,
        relation_sets,
        selected_runs,
        write_json,
    )

OUT_JSONL = DATA / "evidence-selection-v3-recall-collapse-v1.jsonl"
OUT_JSON = DATA / "evidence-selection-v3-recall-collapse-v1.json"
OUT_DOC = DOCS / "evidence-selection-v3-recall-collapse-v1.md"


def to_candidate(row: dict) -> CitationCandidate:
    return CitationCandidate(
        citation_id=row["citation_id"],
        paper_id=row["paper_id"],
        page=row["page"],
        block_id=row["block_id"],
        text=row["text"],
        neighboring_context=row.get("neighboring_context", ""),
        evidence_role=tuple(row.get("evidence_role", [])),
        retrieval_origin=row.get("retrieval_origin", "original_selected"),
        original_selected=bool(row.get("original_selected", False)),
        adjacent_completion=bool(row.get("adjacent_completion", False)),
        currently_cited=bool(row.get("currently_cited", False)),
        retrieval_score=float(row.get("retrieval_score", 0.0)),
        lexical_alignment=float(row.get("lexical_alignment", 0.0)),
        numeric_coverage=float(row.get("numeric_coverage", 0.0)),
        comparison_side_coverage=float(row.get("comparison_side_coverage", 0.0)),
        claim_scope_coverage=float(row.get("claim_scope_coverage", 0.0)),
        redundancy_group=row.get("redundancy_group"),
        token_cost=int(row.get("token_cost", 0)),
    )


def keys_for(ids: set[str] | list[str], key_by_citation: dict[str, str]) -> set[str]:
    return {key_by_citation[cid] for cid in ids if cid in key_by_citation}


def classify(
    *,
    baseline_any: bool,
    candidate_any: bool,
    final_any: bool,
    baseline_ids: set[str],
    selected_ids: set[str],
    eligible_valid_ids: set[str],
    rejected_valid_ids: set[str],
    replaced_without_gain: bool,
) -> str:
    if final_any:
        return "no_failure"
    if baseline_any and not (baseline_ids & selected_ids):
        return (
            "baseline_removed_without_gain"
            if replaced_without_gain
            else "eligible_baseline_rejected"
        )
    if candidate_any and rejected_valid_ids:
        return "complementary_candidate_rejected"
    if candidate_any and not eligible_valid_ids:
        return "excessive_single_evidence_veto"
    if candidate_any:
        return "set_completeness_not_considered"
    if baseline_ids and not selected_ids:
        return "claim_fallback_regression"
    return "true_no_valid_candidate"


def build() -> tuple[list[dict], dict]:
    runs = selected_runs()
    gold = load_gold()
    rows: list[dict] = []
    veto_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    replacement_rows: list[dict] = []
    for required_claim_id, record in sorted(gold.items()):
        question_id = record["question_id"]
        run_dir = RUN_ROOT / runs[question_id]
        _registry, key_by_citation = registry_maps(run_dir)
        final = final_slot(run_dir, required_claim_id)
        claim_text = final.get("claim_text") or record["required_claim_text"]
        candidates_raw = candidate_rows(run_dir, required_claim_id)
        candidates = tuple(to_candidate(row) for row in candidates_raw)
        by_id = {candidate.citation_id: candidate for candidate in candidates}
        valid_sets = relation_sets(record)
        any_valid = valid_sets["core"] | valid_sets["supporting"] | valid_sets["equivalent"]
        exact_valid = valid_sets["core"] | valid_sets["supporting"]
        candidate_key_by_id = keys_for(by_id, key_by_citation)
        baseline_ids = set(final["citation_ids"])
        baseline_keys = keys_for(baseline_ids, key_by_citation)
        v3 = select_evidence_v3(claim_text, candidates, tuple(final["citation_ids"]))
        selected_ids = set(v3.primary_citation_ids + v3.supporting_citation_ids)
        selected_keys = keys_for(selected_ids, key_by_citation)
        candidate_valid_ids = {
            cid for cid, key in key_by_citation.items() if cid in by_id and key in any_valid
        }
        eligible_valid_ids = {
            cid
            for cid in candidate_valid_ids
            if cid in v3.eligibility_results and v3.eligibility_results[cid].eligible
        }
        rejected_valid_ids = candidate_valid_ids - selected_ids
        hard_vetoes: dict[str, list[str]] = {}
        for cid, result in v3.eligibility_results.items():
            for reason in result.hard_fail_reasons:
                veto_counts[reason] += 1
                hard_vetoes.setdefault(reason, []).append(cid)
        baseline_any = bool(baseline_keys & any_valid)
        candidate_any = bool(candidate_key_by_id & any_valid)
        final_any = bool(selected_keys & any_valid)
        replaced_without_gain = bool(v3.baseline_replaced and baseline_any and not final_any)
        category = classify(
            baseline_any=baseline_any,
            candidate_any=candidate_any,
            final_any=final_any,
            baseline_ids=baseline_ids,
            selected_ids=selected_ids,
            eligible_valid_ids=eligible_valid_ids,
            rejected_valid_ids=rejected_valid_ids,
            replaced_without_gain=replaced_without_gain,
        )
        category_counts[category] += 1
        if v3.baseline_replaced:
            replacement_rows.append(
                {
                    "question_id": question_id,
                    "required_claim_id": required_claim_id,
                    "baseline_ids": sorted(baseline_ids),
                    "selected_ids": sorted(selected_ids),
                    "baseline_any_valid": baseline_any,
                    "final_any_valid": final_any,
                    "replacement_reason": v3.replacement_reason,
                    "classification": category,
                }
            )
        rows.append(
            {
                "question_id": question_id,
                "required_claim_id": required_claim_id,
                "baseline_citation_set": sorted(baseline_ids),
                "candidate_evidence_set": sorted(by_id),
                "candidate_any_valid_upper_bound_status": candidate_any,
                "candidate_exact_upper_bound_status": bool(candidate_key_by_id & exact_valid),
                "selection_v3_eligible_candidates": sorted(
                    cid for cid, result in v3.eligibility_results.items() if result.eligible
                ),
                "candidates_rejected_by_hard_veto": hard_vetoes,
                "selected_v3_citation_set": sorted(selected_ids),
                "baseline_retained": bool(baseline_ids & selected_ids),
                "baseline_replaced": v3.baseline_replaced,
                "baseline_removed": bool(baseline_ids - selected_ids),
                "supporting_evidence_accepted": sorted(selected_ids - baseline_ids),
                "supporting_evidence_rejected": sorted(rejected_valid_ids),
                "final_any_valid_status": final_any,
                "recall_lost_relative_to_candidate_upper_bound": candidate_any and not final_any,
                "recall_lost_relative_to_baseline": baseline_any and not final_any,
                "exact_veto_or_replacement_rule": v3.replacement_reason
                or ("hard_veto" if rejected_valid_ids else "none"),
                "single_candidate_eligibility": bool(eligible_valid_ids),
                "set_level_achievable_eligibility": candidate_any,
                "complementary_rejected": bool(rejected_valid_ids),
                "baseline_remained_eligible": baseline_any and not v3.baseline_replaced,
                "baseline_replaced_without_proven_net_gain": replaced_without_gain,
                "citation_slot_remained_available": len(selected_ids) < 3,
                "primary_failure_category": category,
            }
        )
    summary = {
        "schema_version": "evidence-selection-v3-recall-collapse-v1",
        "required_claims": len(rows),
        "candidate_upper_bound_hit_claims": sum(
            row["candidate_any_valid_upper_bound_status"] for row in rows
        ),
        "v3_final_hit_claims": sum(row["final_any_valid_status"] for row in rows),
        "recall_lost_claims": sum(
            row["recall_lost_relative_to_candidate_upper_bound"] for row in rows
        ),
        "root_cause_distribution": dict(sorted(category_counts.items())),
        "unknown_reasons": category_counts.get("unknown", 0),
        "hard_veto_counts": dict(sorted(veto_counts.items())),
        "baseline_replacements": replacement_rows,
        "q002_rule_path": [
            row
            for row in rows
            if row["question_id"] == "q002"
            and row["recall_lost_relative_to_candidate_upper_bound"]
        ],
        "SELECTION_V3_RECALL_COLLAPSE_ATTRIBUTION": "COMPLETE"
        if category_counts.get("unknown", 0) == 0
        else "INCOMPLETE",
        "PRIMARY_SELECTION_V3_FAILURE": category_counts.most_common(1)[0][0]
        if category_counts
        else "none",
        "rows_hash": canonical_hash(rows),
    }
    return rows, summary


def write_outputs(rows: list[dict], summary: dict) -> None:
    OUT_JSONL.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    write_json(OUT_JSON, summary)
    OUT_DOC.write_text(
        "# Evidence Selection v3 Recall Collapse Audit\n\n"
        f"- Attribution: `{summary['SELECTION_V3_RECALL_COLLAPSE_ATTRIBUTION']}`\n"
        f"- Primary failure: `{summary['PRIMARY_SELECTION_V3_FAILURE']}`\n"
        f"- Candidate upper-bound hit claims: `{summary['candidate_upper_bound_hit_claims']}`\n"
        f"- V3 final hit claims: `{summary['v3_final_hit_claims']}`\n"
        f"- Recall lost claims: `{summary['recall_lost_claims']}`\n"
        f"- Unknown reasons: `{summary['unknown_reasons']}`\n\n"
        "## Root cause distribution\n\n"
        + "\n".join(
            f"- `{name}`: `{count}`"
            for name, count in summary["root_cause_distribution"].items()
        )
        + "\n\n## Baseline replacements\n\n"
        + "\n".join(
            f"- `{row['question_id']}` `{row['required_claim_id']}`: "
            f"{row['baseline_ids']} -> {row['selected_ids']}; "
            f"classification=`{row['classification']}`"
            for row in summary["baseline_replacements"]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    rows, summary = build()
    write_outputs(rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
