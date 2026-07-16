# ruff: noqa: E501
"""Summarize reviewed support, exact-vs-human behavior, and citation failures."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

try:
    from scripts.evidence_qa_dev_lib_v1 import DATA, DOCS, read_jsonl
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import DATA, DOCS, read_jsonl  # type: ignore[no-redef]

AUDIT = DATA / "evidence-qa-dev-v3-1-citation-audit-v1.jsonl"
SUMMARY_JSON = DATA / "evidence-qa-dev-v3-1-citation-audit-summary-v1.json"
SUMMARY_DOC = DOCS / "evidence-qa-dev-v3-1-citation-audit-summary-v1.md"
MATRIX_JSON = DATA / "dev-v3-1-exact-vs-human-support-v1.json"
MATRIX_DOC = DOCS / "dev-v3-1-exact-vs-human-support-v1.md"
TAXONOMY_JSONL = DATA / "dev-v3-1-citation-failure-taxonomy-v1.jsonl"
TAXONOMY_JSON = DATA / "dev-v3-1-citation-failure-taxonomy-v1.json"
TAXONOMY_DOC = DOCS / "dev-v3-1-citation-failure-taxonomy-v1.md"
READINESS_JSON = DATA / "stage13-9-phase-b-readiness-v1.json"
READINESS_DOC = DOCS / "stage13-9-phase-b-readiness-v1.md"
STRICT = {"fully_supported"}
LENIENT = {"fully_supported", "partially_supported"}


def grouped(rows: list[dict[str, Any]], key) -> dict[str, Any]:
    output = {}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(key(row))].append(row)
    for name, items in sorted(groups.items()):
        labels = Counter(row["human_label"] for row in items)
        output[name] = {
            "n": len(items),
            "labels": dict(sorted(labels.items())),
            "strict_support_rate": sum(row["human_label"] in STRICT for row in items) / len(items),
            "lenient_support_rate": sum(row["human_label"] in LENIENT for row in items) / len(items),
        }
    return output


def support_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(row["human_label"] for row in rows)
    exact = [row for row in rows if row["automated_labels"]["exact_gold"]]
    misses = [row for row in rows if not row["automated_labels"]["exact_gold"]]
    exact_miss_supported = [row for row in misses if row["human_label"] in LENIENT]
    exact_not_full = [row for row in exact if row["human_label"] != "fully_supported"]
    evidence = {
        (row["paper_id"], row["block_id"]): row
        for row in read_jsonl(DATA / "evidence-corpus-v1.jsonl")
    }
    claims = {
        row["claim_id"]: row for row in read_jsonl(DATA / "claim-units-v1.jsonl")
    }
    enriched = []
    for row in rows:
        unit = evidence[(row["paper_id"], row["block_id"])]
        claim = claims[row["required_claim_id"]]
        enriched.append({
            **row,
            "_claim_role": claim["claim_role"],
            "_evidence_roles": unit["evidence_roles"],
            "_block_type": unit["block_type"],
        })
    return {
        "schema_version": "evidence-qa-dev-v3-1-citation-audit-summary-v1",
        "audit_nature": "AI-assisted manual citation audit",
        "scope_limitations": [
            "Fixed 10-question Dev batch.",
            "33 claim-citation pairs.",
            "Not an independent human double-blind audit.",
            "Must not be extrapolated to Full-50 or production citation precision.",
        ],
        "total": 33,
        "reviewed": 33,
        **{label: labels.get(label, 0) for label in (
            "fully_supported", "partially_supported", "related_but_insufficient",
            "unsupported", "gold_annotation_too_narrow", "ambiguous_claim",
            "malformed_evidence",
        )},
        "strict_support_rate": labels["fully_supported"] / 33,
        "lenient_support_rate": (labels["fully_supported"] + labels["partially_supported"]) / 33,
        "strata": {
            "automated_signal": grouped(enriched, lambda row: row["automated_signal"]),
            "evidence_source": grouped(enriched, lambda row: row["evidence_source"]),
            "question_id": grouped(enriched, lambda row: row["question_id"]),
            "category": grouped(enriched, lambda row: row["category"]),
            "difficulty": grouped(enriched, lambda row: row["difficulty"]),
            "claim_role": grouped(enriched, lambda row: row["_claim_role"]),
            "evidence_role": grouped(
                [
                    {**row, "_single_role": role}
                    for row in enriched
                    for role in (row["_evidence_roles"] or ["none"])
                ],
                lambda row: row["_single_role"],
            ),
            "block_type": grouped(enriched, lambda row: row["_block_type"]),
            "required_claim_id": grouped(enriched, lambda row: row["required_claim_id"]),
        },
        "exact_miss_but_supported": {
            "exact_miss_total": len(misses),
            "fully_supported": sum(row["human_label"] == "fully_supported" for row in misses),
            "partially_supported": sum(row["human_label"] == "partially_supported" for row in misses),
            "strict_rate": sum(row["human_label"] == "fully_supported" for row in misses) / len(misses),
            "lenient_rate": len(exact_miss_supported) / len(misses),
            "count": len(exact_miss_supported),
            "external_narrative_claimed_count": 17,
            "reviewed_records_recalculated_count": len(exact_miss_supported),
            "external_narrative_discrepancy": len(exact_miss_supported) != 17,
        },
        "exact_hit_but_not_fully_supported": {
            "exact_hit_total": len(exact),
            "partially_supported": sum(row["human_label"] == "partially_supported" for row in exact),
            "related_but_insufficient": sum(row["human_label"] == "related_but_insufficient" for row in exact),
            "unsupported": sum(row["human_label"] == "unsupported" for row in exact),
            "count": len(exact_not_full),
            "rate": len(exact_not_full) / len(exact),
        },
        "special_findings": {
            "q001": "Compound claim behavior remains visible: one citation may support only part of a combined proposition.",
            "q015": "Two limitation-oriented claims are misaligned with their cited evidence.",
            "q019": "Citations discuss scaling variables but do not support every precise numeric range in the claims.",
            "q050": "BERT architecture evidence is strong, while cross-paper positioning/comparison is only partially supported.",
            "gold_annotation_too_narrow_present": labels["gold_annotation_too_narrow"] > 0,
            "malformed_evidence_present": labels["malformed_evidence"] > 0,
        },
        "separation_rule": "Human citation support and Exact Gold Recall are parallel metrics and never overwrite each other.",
    }


def exact_matrix(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matrix = Counter(
        ("exact_hit" if row["automated_labels"]["exact_gold"] else "exact_miss", row["human_label"])
        for row in rows
    )
    miss = sum(count for (status, _), count in matrix.items() if status == "exact_miss")
    hit = sum(count for (status, _), count in matrix.items() if status == "exact_hit")
    payload = {
        "schema_version": "dev-v3-1-exact-vs-human-support-v1",
        "matrix": {
            f"{status}+{label}": matrix.get((status, label), 0)
            for status in ("exact_hit", "exact_miss")
            for label in (
                "fully_supported", "partially_supported",
                "related_but_insufficient", "unsupported",
                "gold_annotation_too_narrow", "ambiguous_claim", "malformed_evidence",
            )
        },
        "exact_miss_but_strict_supported_rate": matrix[("exact_miss", "fully_supported")] / miss,
        "exact_miss_but_lenient_supported_rate": (
            matrix[("exact_miss", "fully_supported")]
            + matrix[("exact_miss", "partially_supported")]
        ) / miss,
        "exact_hit_but_not_fully_supported_rate": (
            hit - matrix[("exact_hit", "fully_supported")]
        ) / hit,
        "exact_hit_but_unsupported_rate": matrix[("exact_hit", "unsupported")] / hit,
        "human_support_changes_exact_recall": False,
        "exact_recall_changes_human_support": False,
    }
    return payload


def taxonomy(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_claim: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_claim[(row["question_id"], row["required_claim_id"])].append(row)
    output = []
    for (question_id, claim_id), items in sorted(by_claim.items()):
        labels = [row["human_label"] for row in items]
        signals = [row["automated_signal"] for row in items]
        exact = any(row["automated_labels"]["exact_gold"] for row in items)
        adjacent = any(row["evidence_source"] == "adjacent_completion" for row in items)
        if all(label == "fully_supported" for label in labels):
            category = "citation_fully_supported"
        elif not exact and any(label in LENIENT for label in labels):
            category = (
                "same_page_boundary_miss"
                if "same_page_non_exact" in signals
                else "equivalent_non_gold_cited"
            )
        elif adjacent and any(label != "fully_supported" for label in labels):
            category = "adjacent_partial_support"
        elif len(items) > 2:
            category = "excessive_citation_dilution"
        elif any(label == "unsupported" for label in labels):
            category = "citation_unsupported"
        else:
            category = "citation_partially_supported"
        fixes = {
            "citation_fully_supported": ("retain strict claim-local validation", False, False, False),
            "same_page_boundary_miss": ("score claim-local completeness and prefer the primary block before adjacent support", False, True, False),
            "equivalent_non_gold_cited": ("preserve semantic support separately from exact-Gold diagnostics", False, False, True),
            "adjacent_partial_support": ("treat adjacent completion as supporting rather than automatically primary evidence", False, True, False),
            "excessive_citation_dilution": ("apply a per-claim citation cap and one-primary-citation-first policy", False, True, True),
            "citation_unsupported": ("require claim-local lexical/numeric/comparative evidence checks", False, True, True),
            "citation_partially_supported": ("decompose compound claims and verify numeric/comparative completeness", False, True, True),
        }
        fix, retrieval_change, selection_change, prompt_change = fixes[category]
        output.append({
            "question_id": question_id,
            "required_claim_id": claim_id,
            "failure_type": category,
            "human_labels": labels,
            "exact_status": "hit" if exact else "miss",
            "retrieval_status": "not_adjudicated_without_gold_injection",
            "selected_status": "selected",
            "cited_status": "cited",
            "possible_generic_fix": fix,
            "requires_retrieval_change": retrieval_change,
            "requires_selection_change": selection_change,
            "requires_prompt_change": prompt_change,
            "metric_only_issue": category == "equivalent_non_gold_cited",
            "risk": "medium" if category != "citation_fully_supported" else "low",
        })
    summary = {
        "schema_version": "dev-v3-1-citation-failure-taxonomy-summary-v1",
        "total_required_claims": len(output),
        "by_type": {
            name: {
                "count": len(items),
                "questions": sorted({row["question_id"] for row in items}),
                "required_claims": sorted(row["required_claim_id"] for row in items),
                "human_labels": dict(sorted(Counter(label for row in items for label in row["human_labels"]).items())),
                "possible_generic_fix": items[0]["possible_generic_fix"],
            }
            for name, items in sorted(
                (
                    name,
                    [row for row in output if row["failure_type"] == name],
                )
                for name in {row["failure_type"] for row in output}
            )
        },
        "generic_fix_candidates": [
            {
                "hypothesis": "Per-claim one-primary-citation-first selection with a small citation cap reduces dilution.",
                "supporting_audit_evidence": "q007 produced nine citations; support quality varies across citations.",
                "affected_questions": ["q007"],
                "expected_gain": "higher support concentration",
                "precision_risk": "may omit necessary secondary evidence",
                "token_impact": "lower",
                "latency_impact": "neutral",
                "implementation_complexity": "low",
                "generic": True,
                "requires_live_evaluation": True,
            },
            {
                "hypothesis": "Numeric and comparative completeness checks prevent citations that support only part of a claim.",
                "supporting_audit_evidence": "q019 numeric ranges and q050 cross-paper comparisons are only partially supported.",
                "affected_questions": ["q019", "q050"],
                "expected_gain": "lower unsupported/partial claim rate",
                "precision_risk": "may increase unsupported slots",
                "token_impact": "small",
                "latency_impact": "small",
                "implementation_complexity": "medium",
                "generic": True,
                "requires_live_evaluation": True,
            },
            {
                "hypothesis": "Compound claim decomposition improves citation completeness.",
                "supporting_audit_evidence": "q001 compound propositions and q019 multi-part numeric statements.",
                "affected_questions": ["q001", "q019"],
                "expected_gain": "better atomic support binding",
                "precision_risk": "more output slots",
                "token_impact": "moderate",
                "latency_impact": "small",
                "implementation_complexity": "medium",
                "generic": True,
                "requires_live_evaluation": True,
            },
        ],
    }
    return output, summary


def main() -> None:
    rows = read_jsonl(AUDIT)
    if len(rows) != 33 or not all(row["human_review_status"] == "approved" for row in rows):
        raise RuntimeError("33/33 approved citation audit required")
    summary = support_summary(rows)
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    SUMMARY_DOC.write_text(
        "# Evidence QA Dev v3.1 Human Citation Audit Summary\n\n"
        f"- Audit nature: **{summary['audit_nature']}**\n"
        f"- Reviewed: {summary['reviewed']}/{summary['total']}\n"
        f"- Labels: fully={summary['fully_supported']}, partial={summary['partially_supported']}, related={summary['related_but_insufficient']}, unsupported={summary['unsupported']}\n"
        f"- Strict/lenient support: {summary['strict_support_rate']:.6f}/{summary['lenient_support_rate']:.6f}\n"
        f"- Exact miss but lenient-supported: {summary['exact_miss_but_supported']['count']}/{summary['exact_miss_but_supported']['exact_miss_total']} = {summary['exact_miss_but_supported']['lenient_rate']:.6f}\n"
        f"- Exact hit but not fully supported: {summary['exact_hit_but_not_fully_supported']['count']}/{summary['exact_hit_but_not_fully_supported']['exact_hit_total']} = {summary['exact_hit_but_not_fully_supported']['rate']:.6f}\n"
        "- The external narrative stated 17/24 exact misses were supported; deterministic recomputation from the reviewed labels gives 18/24. Labels were not changed.\n"
        "- Exact Gold hit does not guarantee full support; Exact Gold miss does not imply an invalid citation.\n"
        "- Human support and Exact Gold Recall remain separate metrics.\n"
        "- This fixed 10-question, 33-pair AI-assisted audit cannot be extrapolated to Full-50 or production.\n",
        encoding="utf-8",
    )
    matrix = exact_matrix(rows)
    MATRIX_JSON.write_text(json.dumps(matrix, ensure_ascii=False, indent=2), encoding="utf-8")
    MATRIX_DOC.write_text(
        "# Dev v3.1 Exact Gold vs Human Support\n\n"
        f"- Exact-miss strict/lenient support: {matrix['exact_miss_but_strict_supported_rate']:.6f}/{matrix['exact_miss_but_lenient_supported_rate']:.6f}\n"
        f"- Exact-hit not fully supported: {matrix['exact_hit_but_not_fully_supported_rate']:.6f}\n"
        f"- Exact-hit unsupported: {matrix['exact_hit_but_unsupported_rate']:.6f}\n"
        "- Neither metric overwrites the other.\n",
        encoding="utf-8",
    )
    taxonomy_rows, taxonomy_summary = taxonomy(rows)
    TAXONOMY_JSONL.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in taxonomy_rows),
        encoding="utf-8",
    )
    TAXONOMY_JSON.write_text(
        json.dumps(taxonomy_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    TAXONOMY_DOC.write_text(
        "# Dev v3.1 Citation Failure Taxonomy\n\n"
        + "\n".join(
            f"- `{name}`: {body['count']} claims; questions={body['questions']}; generic fix={body['possible_generic_fix']}"
            for name, body in taxonomy_summary["by_type"].items()
        )
        + "\n\nNo question/block special case, Gold injection, or human-label production selection is used.\n",
        encoding="utf-8",
    )
    comparison = json.loads(
        (DATA / "citation-recall-v2-comparison.json").read_text(encoding="utf-8")
    )
    blocking_ambiguity = comparison["gold_relation_ambiguity_count"] > 0
    readiness = {
        "schema_version": "stage13-9-phase-b-readiness-v1",
        "review_imported_33": True,
        "human_support_summary_complete": True,
        "citation_recall_v2_frozen": True,
        "gold_relation_audit_complete": True,
        "blocking_gold_relation_ambiguity": blocking_ambiguity,
        "historical_v2_recalculation_complete": True,
        "completed_only_denominator_removed": True,
        "stage13_8_gate_remains_failed": True,
        "automatic_exact_and_human_support_separated": True,
        "actionable_failure_type_identified": any(
            row["failure_type"] != "citation_fully_supported" for row in taxonomy_rows
        ),
        "generic_fix_candidate_exists": bool(taxonomy_summary["generic_fix_candidates"]),
        "reranker_enabled": False,
        "gold_or_human_label_online_injection": False,
        "ready_for_dev_v3_2": False if blocking_ambiguity else True,
        "readiness_decision": (
            "CITATION_RECALL_V2_BLOCKED_BY_GOLD_RELATION_AMBIGUITY"
            if blocking_ambiguity
            else "READY_FOR_DEV_V3_2"
        ),
        "dev_v3_2_authorized": False,
        "dev_v3_2_run": False,
        "full_qa_run": False,
        "deep_research_run": False,
        "production_ready": False,
        "v1_0_status": "not_satisfied",
        "current_release": "v0.9.0-rc3",
    }
    READINESS_JSON.write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    READINESS_DOC.write_text(
        "# Stage 13.9 Phase B Readiness\n\n"
        f"- Decision: **{readiness['readiness_decision']}**\n"
        f"- READY_FOR_DEV_V3_2: **{readiness['ready_for_dev_v3_2']}**\n"
        "- DEV_V3_2_AUTHORIZED: **False**\n"
        "- Stage 13.8 historical Gate: still FAILED\n"
        "- Dev v3.2 / Full QA / Deep Research: not run\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "strict": summary["strict_support_rate"],
        "lenient": summary["lenient_support_rate"],
        "exact_miss_supported": summary["exact_miss_but_supported"]["count"],
        "exact_hit_not_full": summary["exact_hit_but_not_fully_supported"]["count"],
        "taxonomy_claims": len(taxonomy_rows),
        "ready_for_dev_v3_2": readiness["ready_for_dev_v3_2"],
        "decision": readiness["readiness_decision"],
    }))


if __name__ == "__main__":
    main()
