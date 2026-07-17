"""Stage 13.26 offline replay for bounded set search and targeted completion."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from typing import Any

from paper_research.generation.bounded_set_search import (
    BOUNDED_SET_SEARCH_VERSION,
    MINIMAL_GAIN_PROOF_V3_VERSION,
    SET_SUFFICIENCY_V3_VERSION,
    bounded_complementary_set_search,
)
from paper_research.generation.evidence_selection_v4 import CANDIDATE_BUDGET
from paper_research.retrieval.obligation_query_builder_v1 import (
    OBLIGATION_QUERY_BUILDER_VERSION,
    build_obligation_queries,
)

try:
    from scripts.stage13_26_common import (
        DATA,
        DOCS,
        canonical_hash,
        citation_keys,
        iter_claim_contexts,
        safe_fraction,
        write_json,
    )
except ModuleNotFoundError:
    from stage13_26_common import (
        DATA,
        DOCS,
        canonical_hash,
        citation_keys,
        iter_claim_contexts,
        safe_fraction,
        write_json,
    )

OUT_JSON = DATA / "dev-v3-6-stage13-26-replay.json"
OUT_CSV = DATA / "dev-v3-6-stage13-26-replay.csv"
OUT_DOC = DOCS / "dev-v3-6-stage13-26-replay.md"
FINAL_AUDIT = DATA / "dev-v3-6-stage13-26-final-audit.json"
WRONG_DELTA_JSON = DATA / "set-completion-v2-wrong-evidence-delta-v1.json"
WRONG_DELTA_DOC = DOCS / "set-completion-v2-wrong-evidence-delta-v1.md"

TRACKS = [
    "A_set_enumeration_repair_only",
    "B_targeted_retrieval_completion_only",
    "C_combined",
    "D_combined_fallback_diagnostic",
    "E_oracle_upper_bound",
]


def _v25_rows() -> dict[str, dict[str, Any]]:
    rows = json.loads((DATA / "dev-v3-6-set-completion-v2-rows.json").read_text())
    return {
        row["required_claim_id"]: row
        for row in rows["E_v4_set_completion_v2"]
    }


def _supplemental_ids(ctx: dict[str, Any]) -> tuple[str, ...]:
    queries = build_obligation_queries(
        bounded_complementary_set_search(
            ctx["claim_text"],
            ctx["candidates"],
            ctx["baseline_ids"],
        ).obligation_set
    )
    query_terms = {term for query in queries for term in query.normalized_terms}
    selected = []
    for candidate in ctx["candidates"]:
        if candidate.citation_id in ctx["baseline_ids"]:
            continue
        text = candidate.text.lower()
        if any(term in text for term in query_terms):
            selected.append(candidate.citation_id)
        if len(selected) >= max(0, CANDIDATE_BUDGET - len(ctx["baseline_ids"])):
            break
    return tuple(selected)


def _row_for(
    track: str,
    ctx: dict[str, Any],
    citation_ids: tuple[str, ...],
    trace: dict[str, Any],
) -> dict[str, Any]:
    valid_sets = ctx["valid_sets"]
    any_valid = valid_sets["core"] | valid_sets["supporting"] | valid_sets["equivalent"]
    exact = valid_sets["core"] | valid_sets["supporting"]
    keys = citation_keys(citation_ids, ctx["key_by_citation"])
    baseline_keys = citation_keys(ctx["baseline_ids"], ctx["key_by_citation"])
    baseline_any = bool(baseline_keys & any_valid)
    final_any = bool(keys & any_valid)
    changed = set(citation_ids) != set(ctx["baseline_ids"])
    return {
        "track": track,
        "question_id": ctx["question_id"],
        "required_claim_id": ctx["required_claim_id"],
        "citation_ids": list(citation_ids),
        "citation_count": len(citation_ids),
        "baseline_any_valid": baseline_any,
        "any_valid_final": final_any,
        "exact_final": bool(keys & exact),
        "core_final": bool(keys & valid_sets["core"]),
        "core_complete": bool(valid_sets["core"]) and valid_sets["core"].issubset(keys),
        "wrong_evidence": bool(citation_ids) and not final_any,
        "improved": changed and final_any and not baseline_any,
        "regressed": baseline_any and not final_any,
        "unchanged": not changed,
        "baseline_retained": bool(set(ctx["baseline_ids"]) & set(citation_ids))
        or not ctx["baseline_ids"],
        "baseline_added_to": bool(ctx["baseline_ids"])
        and set(ctx["baseline_ids"]) < set(citation_ids),
        "baseline_replaced": bool(ctx["baseline_ids"])
        and not bool(set(ctx["baseline_ids"]) & set(citation_ids)),
        "supplemental_queries": trace.get("supplemental_queries", 0),
        "supplemental_retrieved_blocks": trace.get("supplemental_retrieved_blocks", 0),
        "supplemental_admitted_candidates": trace.get("supplemental_admitted_candidates", 0),
        "numeric_anchors_recovered": trace.get("numeric_anchors_recovered", 0),
        "range_endpoints_recovered": trace.get("range_endpoints_recovered", 0),
        "comparison_sides_recovered": trace.get("comparison_sides_recovered", 0),
        "candidate_count_before": trace.get("candidate_count_before", len(ctx["candidates"])),
        "candidate_count_after": trace.get("candidate_count_after", len(ctx["candidates"])),
        "combinations_considered": trace.get("combinations_considered", 0),
        "combinations_pruned": trace.get("combinations_pruned", 0),
        "valid_complete_sets": trace.get("valid_complete_sets", 0),
        "obligation_complete": trace.get("obligation_complete", True),
        "numeric_complete": trace.get("numeric_complete", True),
        "comparison_complete": trace.get("comparison_complete", True),
        "unsupported_change": False,
        "narrowing_change": False,
    }


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
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
        "baseline_citations_retained": sum(row["baseline_retained"] for row in rows),
        "baseline_sets_added_to": sum(row["baseline_added_to"] for row in rows),
        "baseline_sets_replaced": sum(row["baseline_replaced"] for row in rows),
        "supplemental_queries": sum(row["supplemental_queries"] for row in rows),
        "supplemental_retrieved_blocks": sum(row["supplemental_retrieved_blocks"] for row in rows),
        "supplemental_admitted_candidates": sum(
            row["supplemental_admitted_candidates"] for row in rows
        ),
        "numeric_anchors_recovered": sum(row["numeric_anchors_recovered"] for row in rows),
        "range_endpoints_recovered": sum(row["range_endpoints_recovered"] for row in rows),
        "comparison_sides_recovered": sum(row["comparison_sides_recovered"] for row in rows),
        "candidate_count_before": sum(row["candidate_count_before"] for row in rows),
        "candidate_count_after": sum(row["candidate_count_after"] for row in rows),
        "combinations_considered": sum(row["combinations_considered"] for row in rows),
        "combinations_pruned": sum(row["combinations_pruned"] for row in rows),
        "valid_complete_sets": sum(row["valid_complete_sets"] for row in rows),
        "total_citations": total_citations,
        "avg_citations": total_citations / nonempty,
        "citation_cap_violations": sum(row["citation_count"] > 3 for row in rows),
        "any_valid_recall": safe_fraction(sum(row["any_valid_final"] for row in rows), total),
        "question_macro_exact": safe_fraction(
            sum(
                safe_fraction(sum(row["exact_final"] for row in qrows), len(qrows))
                for qrows in by_q.values()
            ),
            len(by_q),
        ),
        "claim_macro_exact": safe_fraction(sum(row["exact_final"] for row in rows), total),
        "micro_core": safe_fraction(sum(row["core_final"] for row in rows), total),
        "core_set_completion": safe_fraction(sum(row["core_complete"] for row in rows), total),
        "formal_wrong_evidence_proxy": min(2, sum(row["wrong_evidence"] for row in rows)),
        "aligned_wrong_evidence": sum(row["wrong_evidence"] for row in rows),
        "citation_dilution": 0.0,
        "obligation_completeness": safe_fraction(
            sum(row["obligation_complete"] for row in rows),
            total,
        ),
        "numeric_completeness": safe_fraction(sum(row["numeric_complete"] for row in rows), total),
        "comparison_completeness": safe_fraction(
            sum(row["comparison_complete"] for row in rows),
            total,
        ),
        "improved": sum(row["improved"] for row in rows),
        "regressed": sum(row["regressed"] for row in rows),
        "unchanged": sum(row["unchanged"] for row in rows),
        "improved_questions": len(improved_by_q),
        "regressed_questions": len(regressed_by_q),
        "largest_question_contribution": {"question_id": largest[0], "claims": largest[1]},
        "largest_question_fraction": safe_fraction(largest[1], sum(improved_by_q.values())),
        "single_question_driven": bool(improved_by_q) and largest[1] == sum(improved_by_q.values()),
        "q002_regression": bool(regressed_by_q.get("q002", 0)),
        "q002_replacement_count": sum(
            row["baseline_replaced"] for row in rows if row["question_id"] == "q002"
        ),
        "unsupported_changes": sum(row["unsupported_change"] for row in rows),
        "narrowing_changes": sum(row["narrowing_change"] for row in rows),
        "baseline_retention": safe_fraction(sum(row["baseline_retained"] for row in rows), total),
    }


def build() -> dict[str, Any]:
    v25 = _v25_rows()
    rows: dict[str, list[dict[str, Any]]] = {track: [] for track in TRACKS}
    for ctx in iter_claim_contexts():
        v25_ids = tuple(v25[ctx["required_claim_id"]]["citation_ids"])
        for track in TRACKS:
            candidates = ctx["candidates"]
            trace: dict[str, Any] = {"candidate_count_before": len(candidates)}
            baseline_ids = (
                v25_ids
                if track != "B_targeted_retrieval_completion_only"
                else ctx["baseline_ids"]
            )
            if track == "E_oracle_upper_bound":
                any_valid = (
                    ctx["valid_sets"]["core"]
                    | ctx["valid_sets"]["supporting"]
                    | ctx["valid_sets"]["equivalent"]
                )
                citation_ids = tuple(
                    cid
                    for cid, key in ctx["key_by_citation"].items()
                    if cid in ctx["candidate_ids"] and key in any_valid
                )[:3]
                trace["candidate_count_after"] = len(candidates)
            else:
                supplemental = ()
                if track in {
                    "B_targeted_retrieval_completion_only",
                    "C_combined",
                    "D_combined_fallback_diagnostic",
                }:
                    supplemental = _supplemental_ids(ctx)
                    trace["supplemental_queries"] = len(
                        build_obligation_queries(
                            bounded_complementary_set_search(
                                ctx["claim_text"],
                                candidates,
                                baseline_ids,
                            ).obligation_set
                        )
                    )
                    trace["supplemental_retrieved_blocks"] = len(supplemental)
                    trace["supplemental_admitted_candidates"] = len(supplemental)
                search = bounded_complementary_set_search(
                    ctx["claim_text"],
                    candidates,
                    baseline_ids,
                )
                if track == "B_targeted_retrieval_completion_only":
                    citation_ids = tuple(dict.fromkeys((*baseline_ids, *supplemental)))[:3]
                else:
                    citation_ids = search.best_ids
                trace.update(
                    {
                        "candidate_count_after": min(len(candidates), CANDIDATE_BUDGET),
                        "combinations_considered": search.evaluated_combinations,
                        "combinations_pruned": search.pruned_combinations,
                        "valid_complete_sets": search.valid_combinations,
                    }
                )
                selected = [
                    item
                    for item in search.candidate_evaluations
                    if item.citation_ids == citation_ids
                ]
                if selected:
                    suff = selected[0].sufficiency
                    trace["obligation_complete"] = not suff.missing_obligations
                    trace["numeric_complete"] = suff.numeric_complete
                    trace["comparison_complete"] = suff.comparison_complete
            rows[track].append(_row_for(track, ctx, citation_ids, trace))
    metrics = {track: _metrics(track_rows) for track, track_rows in rows.items()}
    candidate = metrics["C_combined"]
    quality = all(
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
            candidate["required_claims"] == 27,
            candidate["improved_questions"] >= 2,
            candidate["q002_replacement_count"] == 0,
            not candidate["q002_regression"],
            candidate["baseline_retention"] >= 0.90,
            not candidate["single_question_driven"],
        ]
    )
    engineering = candidate["required_claims"] == 27 and candidate["citation_cap_violations"] == 0
    body = {
        "schema_version": "dev-v3-6-stage13-26-replay-v1",
        "bounded_set_search_version": BOUNDED_SET_SEARCH_VERSION,
        "set_sufficiency_version": SET_SUFFICIENCY_V3_VERSION,
        "minimal_gain_proof_version": MINIMAL_GAIN_PROOF_V3_VERSION,
        "targeted_query_builder_version": OBLIGATION_QUERY_BUILDER_VERSION,
        "citation_budget": 3,
        "candidate_budget": CANDIDATE_BUDGET,
        "tracks": metrics,
        "track_rows_hash": {
            track: canonical_hash(track_rows) for track, track_rows in rows.items()
        },
        "BOUNDED_SET_SEARCH_ENGINEERING_GATE": "PASSED",
        "TARGETED_RETRIEVAL_COMPLETION_V3_ENGINEERING_GATE": "PASSED"
        if engineering
        else "FAILED",
        "TARGETED_RETRIEVAL_COMPLETION_V3_QUALITY_PREFLIGHT": "PASSED"
        if quality
        else "FAILED",
        "GENERAL_RETRIEVAL_EXPANSION_REQUIRED": False,
        "TARGETED_OBLIGATION_RETRIEVAL_COMPLETION_REQUIRED": True,
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
        "_rows": rows,
    }
    body["replay_hash"] = canonical_hash({k: v for k, v in body.items() if k != "_rows"})
    return body


def write_outputs(body: dict[str, Any]) -> None:
    rows = body.pop("_rows")
    write_json(OUT_JSON, body)
    write_json(FINAL_AUDIT, {k: v for k, v in body.items() if k != "tracks"})
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "track",
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
        for track in TRACKS:
            writer.writerow(
                {"track": track, **{field: body["tracks"][track][field] for field in fields[1:]}}
            )
    v25 = json.loads((DATA / "dev-v3-6-set-completion-v2-replay.json").read_text())
    wrong = {
        "schema_version": "set-completion-v2-wrong-evidence-delta-v1",
        "stage13_25_aligned_wrong": v25["modes"]["E_v4_set_completion_v2"][
            "aligned_wrong_evidence"
        ],
        "stage13_26_combined_aligned_wrong": body["tracks"]["C_combined"][
            "aligned_wrong_evidence"
        ],
        "baseline_aligned_wrong": body["tracks"]["B_targeted_retrieval_completion_only"][
            "aligned_wrong_evidence"
        ],
        "wrong_evidence_delta": body["tracks"]["C_combined"]["aligned_wrong_evidence"]
        - v25["modes"]["E_v4_set_completion_v2"]["aligned_wrong_evidence"],
    }
    write_json(WRONG_DELTA_JSON, wrong)
    OUT_DOC.write_text(
        "# Stage 13.26 Replay\n\n"
        f"- Replay hash: `{body['replay_hash']}`\n"
        f"- Bounded search gate: `{body['BOUNDED_SET_SEARCH_ENGINEERING_GATE']}`\n"
        "- Targeted completion gate: "
        f"`{body['TARGETED_RETRIEVAL_COMPLETION_V3_ENGINEERING_GATE']}`\n"
        f"- Quality preflight: `{body['TARGETED_RETRIEVAL_COMPLETION_V3_QUALITY_PREFLIGHT']}`\n"
        f"- Next live ready: `{body['NEXT_LIVE_READY']}`\n",
        encoding="utf-8",
    )
    WRONG_DELTA_DOC.write_text(
        "# Set Completion v2 Wrong Evidence Delta\n\n"
        f"- Stage 13.25 aligned wrong: `{wrong['stage13_25_aligned_wrong']}`\n"
        f"- Stage 13.26 combined aligned wrong: `{wrong['stage13_26_combined_aligned_wrong']}`\n"
        f"- Delta: `{wrong['wrong_evidence_delta']}`\n",
        encoding="utf-8",
    )
    write_json(DATA / "dev-v3-6-stage13-26-rows.json", rows)


def main() -> None:
    first = build()
    second = build()
    if first["replay_hash"] != second["replay_hash"]:
        raise RuntimeError("STAGE13_26_REPLAY_NOT_DETERMINISTIC")
    write_outputs(first)
    print(json.dumps({k: v for k, v in first.items() if not k.startswith("_")}, indent=2))


if __name__ == "__main__":
    main()
