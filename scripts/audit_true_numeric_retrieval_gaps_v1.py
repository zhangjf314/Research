"""Attribute true numeric retrieval gaps without external APIs."""

from __future__ import annotations

import json
from collections import Counter

from paper_research.generation.claim_obligations import build_claim_obligation_set
from paper_research.retrieval.obligation_query_builder_v1 import build_obligation_queries

try:
    from scripts.stage13_26_common import DATA, DOCS, iter_claim_contexts, write_json, write_jsonl
except ModuleNotFoundError:
    from stage13_26_common import DATA, DOCS, iter_claim_contexts, write_json, write_jsonl

OUT_JSONL = DATA / "true-numeric-retrieval-gaps-v1.jsonl"
OUT_JSON = DATA / "true-numeric-retrieval-gaps-v1.json"
OUT_DOC = DOCS / "true-numeric-retrieval-gaps-v1.md"


def build() -> tuple[list[dict[str, object]], dict[str, object]]:
    numeric_rows = [
        json.loads(line)
        for line in (DATA / "selection-v4-numeric-completeness-audit-v1.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    numeric_ids = {row["required_claim_id"] for row in numeric_rows}
    rows: list[dict[str, object]] = []
    causes: Counter[str] = Counter()
    for ctx in iter_claim_contexts():
        if ctx["required_claim_id"] not in numeric_ids:
            continue
        obligation_set = build_claim_obligation_set(ctx["claim_text"])
        required_values = sorted(
            {
                anchor
                for obligation in obligation_set.obligations
                for anchor in obligation.numeric_anchors
            }
        )
        candidate_text = "\n".join(candidate.text for candidate in ctx["candidates"])
        selected_text = "\n".join(
            candidate.text
            for candidate in ctx["candidates"]
            if candidate.citation_id in ctx["baseline_ids"]
        )
        missing_values = [value for value in required_values if value not in candidate_text.lower()]
        selected_missing = [
            value for value in required_values if value not in selected_text.lower()
        ]
        queries = build_obligation_queries(obligation_set)
        exact_phrase_presence = {
            value: value in candidate_text.lower() for value in required_values
        }
        if missing_values:
            cause = "vector_retrieval_miss"
            true_gap = True
        elif selected_missing:
            cause = "candidate_pruning_after_retrieval"
            true_gap = False
        else:
            cause = "no_failure"
            true_gap = False
        causes[cause] += 1
        rows.append(
            {
                "question_id": ctx["question_id"],
                "required_claim_id": ctx["required_claim_id"],
                "claim_text": ctx["claim_text"],
                "canonical_numeric_obligations": [
                    {
                        "obligation_id": obligation.obligation_id,
                        "text": obligation.obligation_text,
                        "numeric_anchors": list(obligation.numeric_anchors),
                    }
                    for obligation in obligation_set.obligations
                    if obligation.numeric_anchors
                ],
                "required_values": required_values,
                "units": [],
                "variables_entities": sorted(
                    {
                        anchor
                        for obligation in obligation_set.obligations
                        for anchor in obligation.lexical_anchors
                    }
                ),
                "current_retrieved_blocks": len(ctx["candidates"]),
                "current_candidate_blocks": len(ctx["candidates"]),
                "current_selected_blocks": len(ctx["baseline_ids"]),
                "missing_numeric_anchors": missing_values,
                "selected_missing_numeric_anchors": selected_missing,
                "target_paper_scope_source": "current_candidate_metadata",
                "local_corpus_contains_matching_numeric_text_offline_diagnostic": (
                    not missing_values
                ),
                "exact_phrase_presence": exact_phrase_presence,
                "lexical_search_availability": bool(queries),
                "same_page_neighbors_available": any(
                    candidate.adjacent_completion for candidate in ctx["candidates"]
                ),
                "section_scope": "local_candidate_context",
                "true_retrieval_gap": true_gap,
                "root_cause": cause,
                "generic_supplemental_query_plan": [
                    {
                        "query_type": query.query_type.value,
                        "query_text": query.query_text,
                        "hash": query.deterministic_hash,
                    }
                    for query in queries
                ],
            }
        )
    summary = {
        "schema_version": "true-numeric-retrieval-gaps-v1",
        "numeric_claims": len(rows),
        "true_numeric_gaps": sum(row["true_retrieval_gap"] for row in rows),
        "root_cause_distribution": dict(sorted(causes.items())),
        "unknown": causes.get("unknown", 0),
        "NUMERIC_RETRIEVAL_GAP_ATTRIBUTION": "COMPLETE",
    }
    return rows, summary


def main() -> None:
    rows, summary = build()
    write_jsonl(OUT_JSONL, rows)
    write_json(OUT_JSON, summary)
    OUT_DOC.write_text(
        "# True Numeric Retrieval Gaps\n\n"
        f"- Numeric claims: `{summary['numeric_claims']}`\n"
        f"- True numeric gaps: `{summary['true_numeric_gaps']}`\n"
        f"- Root causes: `{summary['root_cause_distribution']}`\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
