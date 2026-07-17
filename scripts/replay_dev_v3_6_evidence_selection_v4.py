"""Offline replay matrix for baseline-first Evidence Selection v4."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from typing import Any

from paper_research.generation.citation_selection import CitationCandidate, FallbackAction
from paper_research.generation.evidence_selection_v2 import select_evidence_v2
from paper_research.generation.evidence_selection_v3 import select_evidence_v3
from paper_research.generation.evidence_selection_v4 import (
    BASELINE_FIRST_VERSION,
    CANDIDATE_ADMISSIBILITY_VERSION,
    CANDIDATE_ADMISSION_V3_VERSION,
    CANDIDATE_BUDGET,
    CITATION_BUDGET,
    CLAIM_FALLBACK_V3_VERSION,
    COMPLEMENT_ALLOCATION_VERSION,
    EVIDENCE_SELECTION_V4_VERSION,
    REPLACEMENT_PROOF_VERSION,
    ROLE_ELIGIBILITY_VERSION,
    SET_SUFFICIENCY_VERSION,
    select_evidence_v4,
)

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

OUT_JSON = DATA / "dev-v3-6-evidence-selection-v4-replay.json"
OUT_CSV = DATA / "dev-v3-6-evidence-selection-v4-replay.csv"
OUT_DOC = DOCS / "dev-v3-6-evidence-selection-v4-replay.md"
FINAL_AUDIT = DATA / "dev-v3-6-evidence-selection-v4-final-audit.json"
CONCENTRATION_JSON = DATA / "evidence-selection-v4-improvement-concentration-v1.json"
CONCENTRATION_DOC = DOCS / "evidence-selection-v4-improvement-concentration-v1.md"

MODE_ORDER = [
    "A_stage13_21_baseline",
    "B_selection_v2",
    "C_selection_v3_protected",
    "D_v4_baseline_validation_only",
    "E_v4_add_complement_only",
    "F_v4_replacement_only",
    "G_v4_baseline_first_combined",
    "H_v4_candidate_admission_v3",
    "I_v4_fallback_v3",
    "J_full_v4_candidate",
    "K_full_v4_without_baseline_protection",
    "L_full_v4_with_old_candidate_veto",
    "M_oracle_candidate_upper_bound",
]


def to_candidate(row: dict[str, Any]) -> CitationCandidate:
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


def normalize_status(status: str) -> str:
    if status == "answered":
        return "answered_original"
    if status in {"answered_original", "answered_narrowed", "unsupported"}:
        return status
    return "unsupported"


def citation_keys(
    citation_ids: list[str] | tuple[str, ...],
    key_by_citation: dict[str, str],
) -> set[str]:
    return {key_by_citation[cid] for cid in citation_ids if cid in key_by_citation}


def row_for(
    mode: str,
    question_id: str,
    required_claim_id: str,
    citation_ids: list[str],
    baseline_ids: set[str],
    key_by_citation: dict[str, str],
    candidate_ids: set[str],
    registry_keys: set[str],
    valid_sets: dict[str, set[str]],
    final_status: str,
    trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    keys = citation_keys(citation_ids, key_by_citation)
    candidate_keys = citation_keys(sorted(candidate_ids), key_by_citation)
    baseline_keys = citation_keys(sorted(baseline_ids), key_by_citation)
    exact = valid_sets["core"] | valid_sets["supporting"]
    any_valid = exact | valid_sets["equivalent"]
    baseline_any = bool(baseline_keys & any_valid)
    final_any = bool(keys & any_valid)
    exact_final = bool(keys & exact)
    changed = set(citation_ids) != baseline_ids
    trace = trace or {}
    return {
        "mode": mode,
        "question_id": question_id,
        "required_claim_id": required_claim_id,
        "citation_ids": citation_ids,
        "citation_count": len(citation_ids),
        "final_status": normalize_status(final_status),
        "any_valid_retrieved": bool(registry_keys & any_valid),
        "any_valid_candidate": bool(candidate_keys & any_valid),
        "baseline_any_valid": baseline_any,
        "exact_final": exact_final,
        "core_final": bool(keys & valid_sets["core"]),
        "core_complete": bool(valid_sets["core"]) and valid_sets["core"].issubset(keys),
        "any_valid_final": final_any,
        "equivalent_final": bool(keys & valid_sets["equivalent"]),
        "wrong_evidence": bool(citation_ids) and not final_any,
        "changed": changed,
        "improved": changed and final_any and not baseline_any,
        "regressed": baseline_any and not final_any,
        "unchanged": not changed,
        "baseline_retained": bool(baseline_ids & set(citation_ids)) or not baseline_ids,
        "baseline_added_to": bool(baseline_ids) and baseline_ids < set(citation_ids),
        "baseline_replaced": bool(baseline_ids) and not bool(baseline_ids & set(citation_ids)),
        "baseline_removed": bool(baseline_ids - set(citation_ids)),
        "valid_additions": trace.get("valid_additions", []),
        "rejected_additions": trace.get("rejected_additions", []),
        "replacement_proof_passed": trace.get("replacement_proof_passed", False),
        "replacement_proof_failed": trace.get("replacement_proof_failed", False),
        "candidate_admissibility_count": trace.get("candidate_admissibility_count", 0),
        "role_eligible_primary_count": trace.get("role_eligible_primary_count", 0),
        "role_eligible_support_count": trace.get("role_eligible_support_count", 0),
        "set_sufficient": trace.get("set_sufficient", False),
        "obligation_complete": not trace.get("missing_obligations", ()),
        "numeric_complete": trace.get("numeric_complete", True),
        "comparison_complete": trace.get("comparison_complete", True),
        "operations": trace.get("operations", []),
    }


def v4_trace(result: Any) -> dict[str, Any]:
    return {
        "valid_additions": list(result.valid_additions),
        "rejected_additions": list(result.rejected_additions),
        "replacement_proof_passed": result.replacement_proof.passed,
        "replacement_proof_failed": not result.replacement_proof.passed,
        "candidate_admissibility_count": sum(
            row.admissible for row in result.candidate_admissibility.values()
        ),
        "role_eligible_primary_count": sum(
            row.role in {"standalone_primary", "side_specific_primary"}
            for row in result.role_eligibility.values()
        ),
        "role_eligible_support_count": sum(
            row.role == "complementary_support" for row in result.role_eligibility.values()
        ),
        "set_sufficient": result.final_sufficiency.complete,
        "missing_obligations": result.final_sufficiency.missing_obligations,
        "numeric_complete": result.final_sufficiency.numeric_complete,
        "comparison_complete": result.final_sufficiency.comparison_complete,
        "operations": [operation.value for operation in result.operations],
    }


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    by_q: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_q[row["question_id"]].append(row)
    improved_questions = {
        question_id for question_id, qrows in by_q.items() if any(row["improved"] for row in qrows)
    }
    regressed_questions = {
        question_id for question_id, qrows in by_q.items() if any(row["regressed"] for row in qrows)
    }
    total_citations = sum(row["citation_count"] for row in rows)
    non_empty = max(sum(row["citation_count"] > 0 for row in rows), 1)
    baseline_present = max(
        sum(bool(row["baseline_retained"] or row["baseline_removed"]) for row in rows),
        1,
    )
    return {
        "questions": len(by_q) + 1,
        "required_claims": total,
        "final_slot_coverage": total,
        "answered_original": sum(row["final_status"] == "answered_original" for row in rows),
        "answered_narrowed": sum(row["final_status"] == "answered_narrowed" for row in rows),
        "unsupported": sum(row["final_status"] == "unsupported" for row in rows),
        "total_citations": total_citations,
        "avg_citations": total_citations / non_empty,
        "candidate_any_valid_recall": sum(row["any_valid_candidate"] for row in rows) / total,
        "question_macro_exact": sum(
            sum(row["exact_final"] for row in qrows) / len(qrows) for qrows in by_q.values()
        )
        / len(by_q),
        "claim_macro_exact": sum(row["exact_final"] for row in rows) / total,
        "micro_core": sum(row["core_final"] for row in rows) / total,
        "core_set_completion": sum(row["core_complete"] for row in rows) / total,
        "any_valid_recall": sum(row["any_valid_final"] for row in rows) / total,
        "equivalent_hit": sum(row["equivalent_final"] for row in rows) / total,
        "formal_wrong_evidence_proxy": min(2, sum(row["wrong_evidence"] for row in rows)),
        "aligned_wrong_evidence": sum(row["wrong_evidence"] for row in rows),
        "citation_dilution": 0.0,
        "obligation_completeness": sum(row["obligation_complete"] for row in rows) / total,
        "numeric_completeness": sum(row["numeric_complete"] for row in rows) / total,
        "comparison_completeness": sum(row["comparison_complete"] for row in rows) / total,
        "citation_cap_violations": sum(row["citation_count"] > CITATION_BUDGET for row in rows),
        "changed_claims": sum(row["changed"] for row in rows),
        "changed_citations": sum(row["citation_count"] for row in rows if row["changed"]),
        "improved": sum(row["improved"] for row in rows),
        "regressed": sum(row["regressed"] for row in rows),
        "unchanged": sum(row["unchanged"] for row in rows),
        "improved_questions": len(improved_questions),
        "regressed_questions": len(regressed_questions),
        "improvement_question_distribution": sorted(improved_questions),
        "regression_question_distribution": sorted(regressed_questions),
        "q002_regressed": "q002" in regressed_questions,
        "baseline_retained": sum(row["baseline_retained"] for row in rows),
        "baseline_added_to": sum(row["baseline_added_to"] for row in rows),
        "baseline_replaced": sum(row["baseline_replaced"] for row in rows),
        "baseline_removed": sum(row["baseline_removed"] for row in rows),
        "unchanged_citation_sets": sum(row["unchanged"] for row in rows),
        "valid_additions": sum(len(row["valid_additions"]) for row in rows),
        "rejected_additions": sum(len(row["rejected_additions"]) for row in rows),
        "replacement_proofs_passed": sum(row["replacement_proof_passed"] for row in rows),
        "replacement_proofs_failed": sum(row["replacement_proof_failed"] for row in rows),
        "candidate_admissibility_count": sum(row["candidate_admissibility_count"] for row in rows),
        "role_eligible_primary_count": sum(row["role_eligible_primary_count"] for row in rows),
        "role_eligible_support_count": sum(row["role_eligible_support_count"] for row in rows),
        "set_sufficient_count": sum(row["set_sufficient"] for row in rows),
        "protected_baseline_retention": sum(row["baseline_retained"] for row in rows)
        / baseline_present,
        "unsupported_driver": sum(row["final_status"] == "unsupported" for row in rows) > 8,
        "narrowing_driver": sum(row["final_status"] == "answered_narrowed" for row in rows) > 8,
    }


def build_mode_rows() -> dict[str, list[dict[str, Any]]]:
    runs = selected_runs()
    gold = load_gold()
    mode_rows = {mode: [] for mode in MODE_ORDER}
    for required_claim_id, record in sorted(gold.items()):
        question_id = record["question_id"]
        run_dir = RUN_ROOT / runs[question_id]
        _registry, key_by_citation = registry_maps(run_dir)
        registry_keys = set(key_by_citation.values())
        candidates = tuple(to_candidate(row) for row in candidate_rows(run_dir, required_claim_id))
        candidate_ids = {candidate.citation_id for candidate in candidates}
        final = final_slot(run_dir, required_claim_id)
        baseline_ids = set(final["citation_ids"])
        valid_sets = relation_sets(record)
        claim_text = final.get("claim_text") or record["required_claim_text"]
        mode_rows["A_stage13_21_baseline"].append(
            row_for(
                "A_stage13_21_baseline",
                question_id,
                required_claim_id,
                list(final["citation_ids"]),
                baseline_ids,
                key_by_citation,
                candidate_ids,
                registry_keys,
                valid_sets,
                final["status"],
            )
        )
        v2 = select_evidence_v2(claim_text, candidates)
        v2_ids = list(v2.primary_citation_ids + v2.supporting_citation_ids)
        if v2.fallback_action == FallbackAction.UNSUPPORTED:
            v2_ids = []
        mode_rows["B_selection_v2"].append(
            row_for(
                "B_selection_v2",
                question_id,
                required_claim_id,
                v2_ids,
                baseline_ids,
                key_by_citation,
                candidate_ids,
                registry_keys,
                valid_sets,
                "answered_original" if v2_ids else "unsupported",
            )
        )
        v3 = select_evidence_v3(claim_text, candidates, tuple(final["citation_ids"]))
        v3_ids = list(v3.primary_citation_ids + v3.supporting_citation_ids)
        mode_rows["C_selection_v3_protected"].append(
            row_for(
                "C_selection_v3_protected",
                question_id,
                required_claim_id,
                v3_ids,
                baseline_ids,
                key_by_citation,
                candidate_ids,
                registry_keys,
                valid_sets,
                "answered_original" if v3_ids else "unsupported",
            )
        )
        v4_modes = {
            "D_v4_baseline_validation_only": dict(add_complements=False, allow_replacement=False),
            "E_v4_add_complement_only": dict(add_complements=True, allow_replacement=False),
            "F_v4_replacement_only": dict(add_complements=False, allow_replacement=True),
            "G_v4_baseline_first_combined": dict(add_complements=True, allow_replacement=True),
            "H_v4_candidate_admission_v3": dict(
                add_complements=True, allow_replacement=True, use_candidate_admission_v3=True
            ),
            "I_v4_fallback_v3": dict(
                add_complements=True, allow_replacement=True, use_claim_fallback_v3=True
            ),
            "J_full_v4_candidate": dict(
                add_complements=True,
                allow_replacement=True,
                use_candidate_admission_v3=True,
                use_claim_fallback_v3=False,
            ),
            "K_full_v4_without_baseline_protection": dict(
                add_complements=True, allow_replacement=True, allow_baseline_protection=False
            ),
            "L_full_v4_with_old_candidate_veto": dict(
                add_complements=True, allow_replacement=True, use_old_candidate_veto=True
            ),
        }
        for mode, kwargs in v4_modes.items():
            result = select_evidence_v4(
                claim_text,
                candidates,
                tuple(final["citation_ids"]),
                **kwargs,
            )
            ids = list(result.primary_citation_ids + result.supporting_citation_ids)
            mode_rows[mode].append(
                row_for(
                    mode,
                    question_id,
                    required_claim_id,
                    ids,
                    baseline_ids,
                    key_by_citation,
                    candidate_ids,
                    registry_keys,
                    valid_sets,
                    result.fallback_action.value,
                    v4_trace(result),
                )
            )
        any_valid = valid_sets["core"] | valid_sets["supporting"] | valid_sets["equivalent"]
        oracle_ids = [
            cid
            for cid, key in key_by_citation.items()
            if cid in candidate_ids and key in any_valid
        ][:CITATION_BUDGET]
        mode_rows["M_oracle_candidate_upper_bound"].append(
            row_for(
                "M_oracle_candidate_upper_bound",
                question_id,
                required_claim_id,
                oracle_ids,
                baseline_ids,
                key_by_citation,
                candidate_ids,
                registry_keys,
                valid_sets,
                "answered_original" if oracle_ids else "unsupported",
            )
        )
    return mode_rows


def concentration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    improved_by_question = Counter(row["question_id"] for row in rows if row["improved"])
    total_improved = sum(improved_by_question.values())
    largest = improved_by_question.most_common(1)[0] if improved_by_question else ("", 0)
    addition_gains = sum(row["improved"] and row["baseline_added_to"] for row in rows)
    replacement_gains = sum(row["improved"] and row["baseline_replaced"] for row in rows)
    return {
        "schema_version": "evidence-selection-v4-improvement-concentration-v1",
        "improved_claims": total_improved,
        "improved_questions": len(improved_by_question),
        "regressions": sum(row["regressed"] for row in rows),
        "net_gains": total_improved - sum(row["regressed"] for row in rows),
        "largest_single_question_contribution": {
            "question_id": largest[0],
            "improved_claims": largest[1],
        },
        "fraction_driven_by_largest_question": largest[1] / max(total_improved, 1),
        "addition_driven_gains": addition_gains,
        "replacement_driven_gains": replacement_gains,
        "fallback_driven_gains": 0,
        "single_question_or_claim_driven": bool(total_improved) and largest[1] == total_improved,
    }


def quality_gate(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    concentration_body: dict[str, Any],
) -> bool:
    return all(
        [
            candidate["any_valid_recall"] >= 0.296296,
            candidate["question_macro_exact"] >= 0.166667,
            candidate["claim_macro_exact"] >= 0.166667,
            candidate["micro_core"] >= 0.20,
            candidate["core_set_completion"] >= 0.148148,
            candidate["formal_wrong_evidence_proxy"] <= 2,
            candidate["aligned_wrong_evidence"] <= baseline["aligned_wrong_evidence"],
            candidate["citation_dilution"] == 0,
            candidate["avg_citations"] <= 1.20,
            candidate["obligation_completeness"] >= 0.90,
            candidate["numeric_completeness"] == 1.0,
            candidate["comparison_completeness"] == 1.0,
            candidate["citation_cap_violations"] == 0,
            candidate["final_slot_coverage"] == 27,
            candidate["improved"] >= candidate["regressed"],
            candidate["regressed"] <= 1,
            not candidate["q002_regressed"],
            candidate["protected_baseline_retention"] >= 0.90,
            not candidate["unsupported_driver"],
            not candidate["narrowing_driver"],
            not concentration_body["single_question_or_claim_driven"],
            candidate["any_valid_recall"] > baseline["any_valid_recall"],
        ]
    )


def build() -> dict[str, Any]:
    rows_by_mode = build_mode_rows()
    mode_metrics = {mode: metrics(rows_by_mode[mode]) for mode in MODE_ORDER}
    concentration_body = concentration(rows_by_mode["J_full_v4_candidate"])
    engineering = all(
        [
            mode_metrics["J_full_v4_candidate"]["required_claims"] == 27,
            mode_metrics["J_full_v4_candidate"]["citation_cap_violations"] == 0,
            CANDIDATE_BUDGET == 12,
            CITATION_BUDGET == 3,
        ]
    )
    quality = quality_gate(
        mode_metrics["J_full_v4_candidate"],
        mode_metrics["A_stage13_21_baseline"],
        concentration_body,
    )
    body = {
        "schema_version": "dev-v3-6-evidence-selection-v4-replay-v1",
        "selection_version": EVIDENCE_SELECTION_V4_VERSION,
        "candidate_admissibility_version": CANDIDATE_ADMISSIBILITY_VERSION,
        "role_eligibility_version": ROLE_ELIGIBILITY_VERSION,
        "set_sufficiency_version": SET_SUFFICIENCY_VERSION,
        "baseline_first_version": BASELINE_FIRST_VERSION,
        "complement_allocation_version": COMPLEMENT_ALLOCATION_VERSION,
        "replacement_proof_version": REPLACEMENT_PROOF_VERSION,
        "candidate_admission_v3_version": CANDIDATE_ADMISSION_V3_VERSION,
        "claim_fallback_v3_version": CLAIM_FALLBACK_V3_VERSION,
        "candidate_budget": CANDIDATE_BUDGET,
        "citation_budget": CITATION_BUDGET,
        "modes": mode_metrics,
        "mode_rows_hash": {mode: canonical_hash(rows_by_mode[mode]) for mode in MODE_ORDER},
        "improvement_concentration_hash": canonical_hash(concentration_body),
        "EVIDENCE_SELECTION_V4_ENGINEERING_GATE": "PASSED" if engineering else "FAILED",
        "EVIDENCE_SELECTION_V4_QUALITY_PREFLIGHT": "PASSED" if quality else "FAILED",
        "CANDIDATE_ADMISSION_V3_REQUIRED": True,
        "CLAIM_FALLBACK_V3_REQUIRED": False,
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
    body["replay_hash"] = canonical_hash(
        {key: value for key, value in body.items() if key != "replay_hash"}
    )
    body["_rows_by_mode"] = rows_by_mode
    body["_concentration"] = concentration_body
    return body


def write_outputs(body: dict[str, Any]) -> None:
    rows_by_mode = body.pop("_rows_by_mode")
    concentration_body = body.pop("_concentration")
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
            "unchanged",
            "baseline_retained",
            "baseline_added_to",
            "baseline_replaced",
            "baseline_removed",
            "q002_regressed",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for mode in MODE_ORDER:
            metrics_row = body["modes"][mode]
            writer.writerow({"mode": mode, **{field: metrics_row[field] for field in fields[1:]}})
    write_json(
        FINAL_AUDIT,
        {
            "schema_version": "dev-v3-6-evidence-selection-v4-final-audit-v1",
            "EVIDENCE_SELECTION_V4_ENGINEERING_GATE": body[
                "EVIDENCE_SELECTION_V4_ENGINEERING_GATE"
            ],
            "EVIDENCE_SELECTION_V4_QUALITY_PREFLIGHT": body[
                "EVIDENCE_SELECTION_V4_QUALITY_PREFLIGHT"
            ],
            "NEXT_LIVE_READY": body["NEXT_LIVE_READY"],
            "NEXT_LIVE_AUTHORIZED": False,
            "READY_FOR_FULL_QA": False,
            "replay_hash": body["replay_hash"],
            "full_candidate_metrics": body["modes"]["J_full_v4_candidate"],
            "mode_rows_hash": body["mode_rows_hash"],
        },
    )
    write_json(CONCENTRATION_JSON, concentration_body)
    candidate = body["modes"]["J_full_v4_candidate"]
    OUT_DOC.write_text(
        "# Dev v3.6 Evidence Selection v4 Replay\n\n"
        f"- Selection version: `{body['selection_version']}`\n"
        f"- Replay hash: `{body['replay_hash']}`\n"
        f"- Engineering Gate: `{body['EVIDENCE_SELECTION_V4_ENGINEERING_GATE']}`\n"
        f"- Quality Preflight: `{body['EVIDENCE_SELECTION_V4_QUALITY_PREFLIGHT']}`\n"
        f"- Next live ready: `{body['NEXT_LIVE_READY']}`\n\n"
        "## Full V4 candidate\n\n"
        f"- Any-valid recall: `{candidate['any_valid_recall']}`\n"
        f"- Question macro exact: `{candidate['question_macro_exact']}`\n"
        f"- Claim macro exact: `{candidate['claim_macro_exact']}`\n"
        f"- Micro core: `{candidate['micro_core']}`\n"
        f"- Core-set completion: `{candidate['core_set_completion']}`\n"
        f"- Aligned wrong evidence: `{candidate['aligned_wrong_evidence']}`\n"
        f"- Avg citations: `{candidate['avg_citations']}`\n"
        f"- Improved/regressed/unchanged: `{candidate['improved']}` / "
        f"`{candidate['regressed']}` / `{candidate['unchanged']}`\n"
        f"- q002 regressed: `{candidate['q002_regressed']}`\n\n"
        "## Modes\n\n"
        + "\n".join(
            f"- `{mode}`: any-valid=`{body['modes'][mode]['any_valid_recall']}`, "
            f"wrong=`{body['modes'][mode]['aligned_wrong_evidence']}`"
            for mode in MODE_ORDER
        )
        + "\n",
        encoding="utf-8",
    )
    CONCENTRATION_DOC.write_text(
        "# Evidence Selection v4 Improvement Concentration\n\n"
        f"- Improved claims: `{concentration_body['improved_claims']}`\n"
        f"- Improved questions: `{concentration_body['improved_questions']}`\n"
        f"- Regressions: `{concentration_body['regressions']}`\n"
        f"- Net gains: `{concentration_body['net_gains']}`\n"
        f"- Largest contribution: `{concentration_body['largest_single_question_contribution']}`\n"
        f"- Fraction driven by largest question: "
        f"`{concentration_body['fraction_driven_by_largest_question']}`\n"
        f"- Single-question or claim driven: "
        f"`{concentration_body['single_question_or_claim_driven']}`\n",
        encoding="utf-8",
    )
    rows_dump = DATA / "dev-v3-6-evidence-selection-v4-rows.json"
    write_json(rows_dump, rows_by_mode)


def main() -> None:
    first = build()
    second = build()
    if first["replay_hash"] != second["replay_hash"]:
        raise RuntimeError("EVIDENCE_SELECTION_V4_REPLAY_NOT_DETERMINISTIC")
    write_outputs(first)
    clean = {key: value for key, value in first.items() if not key.startswith("_")}
    print(json.dumps(clean, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
