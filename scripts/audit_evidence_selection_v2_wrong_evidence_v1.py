"""Attribute Selection v2 offline wrong-evidence diagnostics."""

from __future__ import annotations

import json
import re
from collections import Counter

try:
    from scripts.stage13_23_common import (
        DATA,
        DOCS,
        RUN_ROOT,
        candidate_rows,
        canonical_hash,
        final_slot,
        load_gold,
        read_json,
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
        read_json,
        registry_maps,
        relation_sets,
        selected_runs,
        write_json,
    )

OUT_JSONL = DATA / "evidence-selection-v2-wrong-evidence-audit-v1.jsonl"
OUT_JSON = DATA / "evidence-selection-v2-wrong-evidence-audit-v1.json"
OUT_DOC = DOCS / "evidence-selection-v2-wrong-evidence-audit-v1.md"


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", text.lower()))


def classify(claim: str, evidence: str, baseline_quality: str, replaced: bool) -> str:
    claim_tokens = _tokens(claim)
    evidence_tokens = _tokens(evidence)
    if replaced and baseline_quality in {"valid", "equivalent"}:
        return "replacement_of_better_baseline"
    if re.search(r"\d", claim) and not re.search(r"\d", evidence):
        return "numeric_anchor_missing"
    if re.search(r"\b(compare|better|versus|whereas|while|both)\b", claim, re.I):
        return "comparison_one_side_only"
    if re.search(r"\b(limit|limitation|fail|cannot|not)\b", claim, re.I) and re.search(
        r"\b(improve|achieve|outperform|advantage)\b", evidence, re.I
    ):
        return "limitation_vs_capability_confusion"
    if re.search(r"\b(survey|overview|review)\b", evidence, re.I) and not re.search(
        r"\b(survey|overview|review)\b", claim, re.I
    ):
        return "survey_vs_primary_source_confusion"
    if len(claim_tokens & evidence_tokens) <= 2:
        return "lexical_overlap_without_entailment"
    if re.search(r"\b(setup|train|optimizer|gpu|dataset)\b", evidence, re.I) and re.search(
        r"\b(result|score|accuracy|bleu|rouge)\b", claim, re.I
    ):
        return "result_vs_setup_confusion"
    return "obligation_keyword_false_positive"


def main() -> None:
    replay = read_json(DATA / "dev-v3-6-evidence-selection-v2-replay.json")
    runs = selected_runs()
    gold = load_gold()
    rows = []
    for detail in replay["details"]:
        qid = detail["question_id"]
        required_claim_id = detail["required_claim_id"]
        run_dir = RUN_ROOT / runs[qid]
        _registry, key_by_citation = registry_maps(run_dir)
        candidates = {row["citation_id"]: row for row in candidate_rows(run_dir, required_claim_id)}
        gold_sets = relation_sets(gold[required_claim_id])
        valid = gold_sets["core"] | gold_sets["supporting"] | gold_sets["equivalent"]
        final = final_slot(run_dir, required_claim_id)
        baseline_valid = any(
            key_by_citation.get(cid) in valid for cid in detail["baseline_citations"]
        )
        baseline_quality = "valid" if baseline_valid else "wrong_or_unverified"
        for cid in detail["selection_v2_citations"]:
            key = key_by_citation.get(cid)
            if key in valid:
                continue
            candidate = candidates.get(cid)
            if candidate is None:
                continue
            replaced = cid not in set(detail["baseline_citations"])
            category = classify(
                final.get("claim_text") or "",
                candidate["text"],
                baseline_quality,
                replaced,
            )
            rows.append(
                {
                    "question_id": qid,
                    "required_claim_id": required_claim_id,
                    "final_claim_text": final.get("claim_text"),
                    "candidate_evidence_text_hash": canonical_hash(candidate["text"]),
                    "citation_id": cid,
                    "selected_role": "primary",
                    "selection_score_components": {
                        "lexical_alignment": candidate.get("lexical_alignment", 0),
                        "retrieval_score": candidate.get("retrieval_score", 0),
                        "numeric_coverage": candidate.get("numeric_coverage", 0),
                        "comparison_side_coverage": candidate.get("comparison_side_coverage", 0),
                    },
                    "lexical_overlap": len(
                        _tokens(final.get("claim_text") or "") & _tokens(candidate["text"])
                    ),
                    "obligation_coverage": candidate.get("claim_scope_coverage", 0),
                    "numeric_compatibility": candidate.get("numeric_coverage", 0),
                    "comparison_compatibility": candidate.get("comparison_side_coverage", 0),
                    "polarity_compatibility": category != "limitation_vs_capability_confusion",
                    "entity_compatibility": category not in {"entity_mismatch", "method_mismatch"},
                    "method_object_compatibility": category != "method_mismatch",
                    "original_adjacent_role": candidate.get("retrieval_origin"),
                    "retrieval_score": candidate.get("retrieval_score", 0),
                    "redundancy_contribution": 0,
                    "selected_because_of_rule": "selection_v2_obligation_weighted_score",
                    "offline_core_or_equivalent_relation": False,
                    "offline_quality": "wrong",
                    "why_wrong_evidence_was_admitted": category,
                    "missing_veto_rule": category,
                    "baseline_citation": detail["baseline_citations"],
                    "baseline_citation_offline_quality": baseline_quality,
                    "v2_replaced_better_baseline_choice": replaced
                    and baseline_quality in {"valid", "equivalent"},
                    "v2_added_unsupported_supporting_evidence": False,
                    "generic_failure_category": category,
                }
            )
    counts = Counter(row["generic_failure_category"] for row in rows)
    body = {
        "schema_version": "evidence-selection-v2-wrong-evidence-audit-v1",
        "records": len(rows),
        "unknown": counts.get("unknown", 0),
        "root_cause_distribution": dict(sorted(counts.items())),
        "rows_hash": canonical_hash(rows),
        "diagnostic_only": True,
        "not_production_input": True,
    }
    OUT_JSONL.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    write_json(OUT_JSON, {**body, "rows": rows})
    OUT_DOC.write_text(
        "# Evidence Selection v2 Wrong Evidence Audit\n\n"
        f"- Records: `{body['records']}`\n"
        f"- Unknown: `{body['unknown']}`\n"
        f"- Rows hash: `{body['rows_hash']}`\n\n"
        + "\n".join(f"- `{key}`: {value}" for key, value in body["root_cause_distribution"].items())
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(body, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
