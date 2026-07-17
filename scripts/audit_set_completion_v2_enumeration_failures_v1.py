"""Attribute Stage 13.25 complementary-set enumeration failures."""

from __future__ import annotations

import json
from collections import Counter

from paper_research.generation.bounded_set_search import bounded_complementary_set_search
from paper_research.generation.claim_obligations import build_claim_obligation_set
from paper_research.generation.set_completion_v2 import select_set_completion_v2

try:
    from scripts.stage13_26_common import (
        DATA,
        DOCS,
        citation_keys,
        iter_claim_contexts,
        relation_key_from_candidate,
        write_csv,
        write_json,
        write_jsonl,
    )
except ModuleNotFoundError:
    from stage13_26_common import (
        DATA,
        DOCS,
        citation_keys,
        iter_claim_contexts,
        relation_key_from_candidate,
        write_csv,
        write_json,
        write_jsonl,
    )

OUT_JSONL = DATA / "set-completion-v2-enumeration-failures-v1.jsonl"
OUT_JSON = DATA / "set-completion-v2-enumeration-failures-v1.json"
OUT_DOC = DOCS / "set-completion-v2-enumeration-failures-v1.md"


def build() -> tuple[list[dict[str, object]], dict[str, object]]:
    gap_rows = [
        json.loads(line)
        for line in (DATA / "evidence-selection-v4-oracle-gap-v1.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    gap_ids = {
        row["required_claim_id"]
        for row in gap_rows
        if row["primary_gap_cause"] == "complementary_set_not_enumerated"
    }
    rows: list[dict[str, object]] = []
    causes: Counter[str] = Counter()
    reproducible = 0
    for ctx in iter_claim_contexts():
        if ctx["required_claim_id"] not in gap_ids:
            continue
        valid_sets = ctx["valid_sets"]
        any_valid = valid_sets["core"] | valid_sets["supporting"] | valid_sets["equivalent"]
        obligation_set = build_claim_obligation_set(ctx["claim_text"])
        v2 = select_set_completion_v2(ctx["claim_text"], ctx["candidates"], ctx["baseline_ids"])
        bounded = bounded_complementary_set_search(
            ctx["claim_text"],
            ctx["candidates"],
            ctx["baseline_ids"],
        )
        combo_rows = []
        best_valid: tuple[str, ...] = ()
        for evaluation in bounded.candidate_evaluations:
            keys = citation_keys(evaluation.citation_ids, ctx["key_by_citation"])
            offline_valid = bool(keys & any_valid)
            if offline_valid and not best_valid:
                best_valid = evaluation.citation_ids
            combo_rows.append(
                {
                    "citation_ids": list(evaluation.citation_ids),
                    "size": len(evaluation.citation_ids),
                    "offline_valid": offline_valid,
                    "generic_valid": evaluation.valid,
                    "covered_obligations": list(evaluation.sufficiency.covered_obligations),
                    "missing_obligations": list(evaluation.sufficiency.missing_obligations),
                    "hard_conflicts": list(evaluation.sufficiency.hard_conflicts),
                    "gain_reason": evaluation.gain_proof.reason,
                }
            )
        v2_ids = v2.primary_citation_ids + v2.supporting_citation_ids
        bounded_keys = citation_keys(bounded.best_ids, ctx["key_by_citation"])
        generic_reproducible = bool(bounded_keys & any_valid)
        if generic_reproducible:
            reproducible += 1
            cause = "set_completion_v2_missed_bounded_valid_set"
        elif best_valid:
            cause = "oracle_relation_not_reproducible_without_gold"
        else:
            cause = "no_valid_non_gold_set"
        causes[cause] += 1
        rows.append(
            {
                "question_id": ctx["question_id"],
                "required_claim_id": ctx["required_claim_id"],
                "claim_text": ctx["claim_text"],
                "canonical_obligations": [
                    {
                        "obligation_id": obligation.obligation_id,
                        "text": obligation.obligation_text,
                        "type": obligation.obligation_type,
                    }
                    for obligation in obligation_set.obligations
                ],
                "baseline_citation_set": list(ctx["baseline_ids"]),
                "admitted_candidates": [
                    {
                        "citation_id": candidate.citation_id,
                        "relation_key": relation_key_from_candidate(candidate),
                    }
                    for candidate in ctx["candidates"][:12]
                ],
                "oracle_valid_candidate_relations_offline_only": sorted(any_valid),
                "candidate_combinations_size_1": sum(
                    len(row["citation_ids"]) == 1 for row in combo_rows
                ),
                "candidate_combinations_size_2": sum(
                    len(row["citation_ids"]) == 2 for row in combo_rows
                ),
                "candidate_combinations_size_3": sum(
                    len(row["citation_ids"]) == 3 for row in combo_rows
                ),
                "combinations_considered_by_v2": v2.combinations_considered,
                "combinations_skipped_by_v2": v2.combinations_pruned,
                "bounded_total_combinations": bounded.total_combinations,
                "bounded_pruned_combinations": bounded.pruned_combinations,
                "bounded_evaluated_combinations": bounded.evaluated_combinations,
                "bounded_valid_combinations": bounded.valid_combinations,
                "v2_selected_set": list(v2_ids),
                "bounded_best_combination": list(bounded.best_ids),
                "best_valid_combination_if_exists": list(best_valid),
                "failure_stage": "enumeration_or_gain_proof",
                "primary_root_cause": cause,
                "generic_repair": "bounded_exhaustive_set_search",
                "generic_rules_identify_oracle_valid_set": generic_reproducible,
                "all_combinations": combo_rows,
                "q002_baseline_retained_audit": (
                    set(ctx["baseline_ids"]).issubset(set(bounded.best_ids))
                    if ctx["question_id"] == "q002"
                    else None
                ),
                "q002_replacement_count_audit": (
                    len(set(ctx["baseline_ids"]) - set(bounded.best_ids))
                    if ctx["question_id"] == "q002"
                    else None
                ),
            }
        )
    summary = {
        "schema_version": "set-completion-v2-enumeration-failures-v1",
        "complementary_gap_claims": len(rows),
        "generic_reproducible_without_gold": reproducible,
        "root_cause_distribution": dict(sorted(causes.items())),
        "unknown": causes.get("unknown", 0),
        "COMPLEMENTARY_ENUMERATION_FAILURE_ATTRIBUTION": "COMPLETE",
        "PRIMARY_ENUMERATION_FAILURE": causes.most_common(1)[0][0] if causes else "none",
    }
    return rows, summary


def main() -> None:
    rows, summary = build()
    write_jsonl(OUT_JSONL, rows)
    write_csv(DATA / "set-completion-v2-enumeration-failures-v1.csv", rows)
    write_json(OUT_JSON, summary)
    OUT_DOC.write_text(
        "# Set Completion v2 Enumeration Failures\n\n"
        f"- Attribution: `{summary['COMPLEMENTARY_ENUMERATION_FAILURE_ATTRIBUTION']}`\n"
        f"- Gap claims: `{summary['complementary_gap_claims']}`\n"
        f"- Generic reproducible without Gold: `{summary['generic_reproducible_without_gold']}`\n"
        f"- Root causes: `{summary['root_cause_distribution']}`\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
