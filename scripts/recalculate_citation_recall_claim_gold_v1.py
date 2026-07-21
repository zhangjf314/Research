"""Recalculate historical citation metrics with frozen claim-level Gold."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from statistics import mean
from typing import Any

try:
    from scripts.evidence_qa_dev_lib_v1 import DATA, DOCS, overlap, read_jsonl
    from scripts.recalculate_citation_recall_v2 import (
        CLAIM_MATCH_THRESHOLD,
        generated_claims,
        load_experiments,
    )
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import (  # type: ignore[no-redef]
        DATA,
        DOCS,
        overlap,
        read_jsonl,
    )
    from recalculate_citation_recall_v2 import (  # type: ignore[no-redef]
        CLAIM_MATCH_THRESHOLD,
        generated_claims,
        load_experiments,
    )

ANSWERABLE_IDS = ["q001", "q002", "q004", "q007", "q008", "q013", "q015", "q019", "q050"]
CLAIM_GOLD = DATA / "claim-evidence-gold-dev-v1.jsonl"
COMPARISON_JSON = DATA / "claim-gold-citation-comparison-v1.json"
COMPARISON_CSV = DATA / "claim-gold-citation-comparison-v1.csv"
COMPARISON_DOC = DOCS / "claim-gold-citation-comparison-v1.md"
TAXONOMY_JSONL = DATA / "dev-v3-1-citation-failure-taxonomy-v2.jsonl"
TAXONOMY_JSON = DATA / "dev-v3-1-citation-failure-taxonomy-v2.json"
TAXONOMY_DOC = DOCS / "dev-v3-1-citation-failure-taxonomy-v2.md"
PLAN_JSON = DATA / "dev-v3-2-citation-improvement-plan-v1.json"
PLAN_DOC = DOCS / "dev-v3-2-citation-improvement-plan-v1.md"
READINESS_JSON = DATA / "stage13-10-phase-b-readiness-v1.json"
READINESS_DOC = DOCS / "stage13-10-phase-b-readiness-v1.md"
HISTORICAL_FILES = {
    "stage13_5": DATA / "stage13-5-schema-failure-freeze-v1.json",
    "stage13_6": DATA / "evidence-qa-dev-v3-readiness-v1.json",
    "stage13_7": DATA / "stage13-review-hash-migration-v1.json",
    "stage13_8": DATA / "evidence-qa-dev-v3-1.json",
    "stage13_9_metric_v2": DATA / "citation-recall-metric-v2.json",
    "gold_set_v1": DATA / "gold-set-v1.jsonl",
    "retrieval_gold_v2": DATA / "retrieval-gold-v2.jsonl",
}

AUDIT_FILES = {
    "stage11c_a": DATA / "citation-human-audit-sample-v1.jsonl",
    "stage13_2_b": DATA / "evidence-qa-dev-citation-audit-v1.jsonl",
    "stage13_3_dev_v2": DATA / "evidence-qa-dev-v2-citation-audit-v1.jsonl",
    "stage13_8_dev_v3_1": DATA / "evidence-qa-dev-v3-1-citation-audit-v1.jsonl",
}


def relation_sets(row: dict[str, Any]) -> dict[str, set[str]]:
    core: set[str] = set()
    for item in row["approved_core_relations"]:
        if isinstance(item, str):
            core.add(item)
        else:
            core.update(item["required_relations"])
    return {
        "core": core,
        "supporting": set(row["approved_supporting_relations"]),
        "equivalent": set(row["equivalent_non_gold_relations"]),
        "rejected": set(row["rejected_relations"]),
    }


def cited_triples_by_claim(
    experiment_row: dict[str, Any] | None,
    claims: list[dict[str, Any]],
) -> dict[str, set[tuple[str, int, str]]]:
    if not experiment_row or str(experiment_row.get("status", "")).lower() != "completed":
        return {claim["required_claim_id"]: set() for claim in claims}
    answer = experiment_row.get("answer", {})
    direct = {
        claim["required_claim_id"]: {
            (citation["paper_id"], int(citation["page"]), citation["block_id"])
            for citation in claim.get("citations", [])
        }
        for claim in answer.get("claims", [])
        if claim.get("required_claim_id")
    }
    legacy = generated_claims(answer)
    output: dict[str, set[tuple[str, int, str]]] = {}
    for claim in claims:
        claim_id = claim["required_claim_id"]
        if claim_id in direct:
            output[claim_id] = direct[claim_id]
            continue
        match = max(
            legacy,
            key=lambda item: overlap(claim["required_claim_text"], item["text"]),
            default=None,
        )
        output[claim_id] = (
            match["triples"]
            if match
            and overlap(claim["required_claim_text"], match["text"]) >= CLAIM_MATCH_THRESHOLD
            else set()
        )
    return output


def human_support(version: str) -> dict[str, Any]:
    path = AUDIT_FILES[version]
    if not path.exists():
        return {"available": False, "records": 0, "strict": None, "lenient": None}
    rows = [row for row in read_jsonl(path) if row.get("human_review_status") == "approved"]
    labels = Counter(row.get("human_label") for row in rows)
    return {
        "available": bool(rows),
        "records": len(rows),
        "strict": labels["fully_supported"] / len(rows) if rows else None,
        "lenient": (
            labels["fully_supported"] + labels["partially_supported"]
        ) / len(rows)
        if rows
        else None,
        "labels": dict(sorted(labels.items(), key=lambda item: str(item[0]))),
    }


def calculate_experiment(
    experiment: dict[str, Any],
    claims_by_question: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    per_question: dict[str, dict[str, Any]] = {}
    claim_exact_scores: list[float] = []
    claim_core_completions: list[float] = []
    claim_any_hits: list[float] = []
    core_total = core_hits = equivalent_total = equivalent_hits = 0
    supporting_only = incomplete_core = 0
    failed: list[str] = []
    detail: list[dict[str, Any]] = []
    for question_id in ANSWERABLE_IDS:
        run = experiment["rows"].get(question_id)
        completed = bool(run and str(run.get("status", "")).lower() == "completed")
        if not completed:
            failed.append(question_id)
        claims = claims_by_question[question_id]
        cited_by_claim = cited_triples_by_claim(run, claims)
        q_exact_num = q_exact_den = 0
        q_core_completion: list[float] = []
        q_any: list[float] = []
        for claim in claims:
            sets = relation_sets(claim)
            relations = {
                relation["relation_id"]: relation
                for relation in claim["candidate_evidence_relations"]
            }
            triple_to_id = {
                (relation["paper_id"], int(relation["page"]), relation["block_id"]): relation_id
                for relation_id, relation in relations.items()
            }
            cited_ids = {
                triple_to_id[triple]
                for triple in cited_by_claim[claim["required_claim_id"]]
                if triple in triple_to_id
            }
            exact = sets["core"] | sets["supporting"]
            exact_hits = exact & cited_ids
            core_claim_hits = sets["core"] & cited_ids
            equivalent_claim_hits = sets["equivalent"] & cited_ids
            exact_score = len(exact_hits) / len(exact) if exact else 0.0
            core_complete = float(bool(sets["core"]) and sets["core"] <= cited_ids)
            any_valid = float(bool((exact | sets["equivalent"]) & cited_ids))
            claim_exact_scores.append(exact_score)
            claim_core_completions.append(core_complete)
            claim_any_hits.append(any_valid)
            q_exact_num += len(exact_hits)
            q_exact_den += len(exact)
            q_core_completion.append(core_complete)
            q_any.append(any_valid)
            core_total += len(sets["core"])
            core_hits += len(core_claim_hits)
            equivalent_total += len(sets["equivalent"])
            equivalent_hits += len(equivalent_claim_hits)
            supporting_only += bool(
                (sets["supporting"] & cited_ids)
                and not ((sets["core"] | sets["equivalent"]) & cited_ids)
            )
            incomplete_core += bool(
                core_claim_hits and sets["core"] and not sets["core"] <= cited_ids
            )
            detail.append(
                {
                    "question_id": question_id,
                    "required_claim_id": claim["required_claim_id"],
                    "run_completed": completed,
                    "exact_relation_numerator": len(exact_hits),
                    "exact_relation_denominator": len(exact),
                    "exact_relation_recall": exact_score,
                    "core_relation_numerator": len(core_claim_hits),
                    "core_relation_denominator": len(sets["core"]),
                    "core_set_complete": bool(core_complete),
                    "any_valid_evidence_hit": bool(any_valid),
                    "equivalent_hit_count": len(equivalent_claim_hits),
                    "cited_relation_ids": sorted(cited_ids),
                    "cited_triples": sorted(cited_by_claim[claim["required_claim_id"]]),
                }
            )
        per_question[question_id] = {
            "exact_relation_recall": q_exact_num / q_exact_den if q_exact_den else 0.0,
            "core_set_completion": mean(q_core_completion),
            "any_valid_evidence_recall": mean(q_any),
            "run_completed": completed,
        }
    support = human_support(experiment["evaluation_version"])
    return {
        "evaluation_version": experiment["evaluation_version"],
        "metric_status": "claim_gold_recalculated_diagnostic",
        "available_questions": sorted(experiment["rows"]),
        "missing_questions": sorted(set(ANSWERABLE_IDS) - set(experiment["rows"])),
        "failed_questions": failed,
        "answerable_denominator": 9,
        "required_claim_denominator": 27,
        "core_relation_denominator": core_total,
        "answerable_question_macro_exact_relation_recall": mean(
            row["exact_relation_recall"] for row in per_question.values()
        ),
        "required_claim_macro_exact_relation_recall": mean(claim_exact_scores),
        "micro_core_relation_recall": core_hits / core_total if core_total else 0.0,
        "micro_core_relation_numerator": core_hits,
        "claim_core_set_completion": mean(claim_core_completions),
        "answerable_question_macro_core_set_completion": mean(
            row["core_set_completion"] for row in per_question.values()
        ),
        "claim_any_valid_evidence_recall": mean(claim_any_hits),
        "answerable_question_macro_any_valid_evidence_recall": mean(
            row["any_valid_evidence_recall"] for row in per_question.values()
        ),
        "equivalent_valid_evidence_hit_rate": (
            equivalent_hits / equivalent_total if equivalent_total else 0.0
        ),
        "equivalent_numerator": equivalent_hits,
        "equivalent_denominator": equivalent_total,
        "supporting_only_hit_rate": supporting_only / 27,
        "incomplete_core_set_rate": incomplete_core / 27,
        "human_strict_citation_support": support["strict"],
        "human_lenient_citation_support": support["lenient"],
        "human_support_audit": support,
        "historical_metric": experiment["historical_reported_recall"],
        "historical_formula": experiment["historical_formula"],
        "historical_gate_modified": False,
        "comparable_under_claim_gold": True,
        "per_question": per_question,
        "per_claim": detail,
        "limitations": [
            "Fixed 27-claim Dev diagnostic; not Full-50 or Production.",
            "Legacy generated claims use the frozen lexical claim matcher.",
            "Equivalent evidence does not modify historical exact Gold recall.",
            "Provider/schema/validation failures score zero under the fixed denominator.",
        ],
    }


def compare_dev(
    dev2: dict[str, Any], dev31: dict[str, Any]
) -> dict[str, Any]:
    outcomes = Counter()
    deltas: dict[str, dict[str, float | str]] = {}
    for question_id in ANSWERABLE_IDS:
        before = dev2["per_question"][question_id]["exact_relation_recall"]
        after = dev31["per_question"][question_id]["exact_relation_recall"]
        outcome = "improved" if after > before else "regressed" if after < before else "unchanged"
        outcomes[outcome] += 1
        deltas[question_id] = {
            "dev_v2": before,
            "dev_v3_1": after,
            "delta": after - before,
            "outcome": outcome,
        }
    nonzero = [abs(row["delta"]) for row in deltas.values() if row["delta"]]
    return {
        "outcomes": dict(outcomes),
        "per_question": deltas,
        "single_question_driven": bool(nonzero and max(nonzero) == sum(nonzero)),
        "equivalent_evidence_driver": (
            dev31["equivalent_valid_evidence_hit_rate"]
            > dev2["equivalent_valid_evidence_hit_rate"]
        ),
        "schema_recovery_contribution": (
            "q050 is a fixed-denominator zero in Dev v2 and completed in Dev v3.1; "
            "its delta includes engineering/schema recovery as well as citation behavior."
        ),
    }


def taxonomy_rows(
    dev31: dict[str, Any], claims: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    details = {
        row["required_claim_id"]: row for row in dev31["per_claim"]
    }
    output: list[dict[str, Any]] = []
    numeric_pattern = re.compile(r"\b\d+(?:\.\d+)?\b|range|shape|dimension|variable", re.I)
    for claim in claims:
        detail = details[claim["required_claim_id"]]
        sets = relation_sets(claim)
        relations = {
            relation["relation_id"]: relation
            for relation in claim["candidate_evidence_relations"]
        }
        cited = set(detail["cited_relation_ids"])
        failure_types: list[str] = []
        missing_core = sets["core"] - cited
        for relation_id in missing_core:
            relation = relations[relation_id]
            if not relation["retrieved_in_dev_v3_1"]:
                failure_types.append("core_gold_not_retrieved")
            elif not relation["selected_in_dev_v3_1"]:
                failure_types.append("core_gold_retrieved_not_selected")
            else:
                failure_types.append("core_gold_selected_not_cited")
        if sets["supporting"] & cited and not sets["core"] & cited:
            failure_types.append("supporting_only_cited")
        if sets["equivalent"] & cited:
            failure_types.append("equivalent_valid_evidence_cited")
        cited_rejected = sets["rejected"] & cited
        labels = {
            relations[relation_id]["adjudication_label"] for relation_id in cited_rejected
        }
        if "partially_relevant" in labels:
            failure_types.append("partial_evidence_cited")
        if "insufficient" in labels or "unrelated" in labels:
            failure_types.append("wrong_evidence_cited")
        if sets["core"] & cited and not sets["core"] <= cited:
            failure_types.append("incomplete_core_set")
        if numeric_pattern.search(claim["required_claim_text"]) and missing_core:
            failure_types.append("numeric_evidence_missing")
        comparison_language = re.search(
            r"\b(compar|versus|first paper|second paper|rather than)\b",
            claim["required_claim_text"],
            re.I,
        )
        if comparison_language and not detail["core_set_complete"]:
            failure_types.append("comparison_side_missing")
        if len(sets["core"]) > 1 and not detail["core_set_complete"]:
            failure_types.append("claim_too_broad")
        if claim["no_valid_gold_evidence"]:
            failure_types.append("no_valid_gold_evidence")
        if not sets["core"] and sets["equivalent"] & cited:
            failure_types.append("metric_only_legacy_gold_issue")
        human_labels = sorted(
            {
                label
                for relation_id in cited
                for label in relations[relation_id]["human_citation_support_labels"]
            }
        )
        if "fully_supported" in human_labels:
            failure_types.append("citation_fully_supported")
        if "partially_supported" in human_labels:
            failure_types.append("citation_partially_supported")
        if {"unsupported", "related_but_insufficient"} & set(human_labels):
            failure_types.append("citation_unsupported")
        failure_types = sorted(set(failure_types))
        fixes = []
        if any("not_retrieved" in item for item in failure_types):
            fixes.append("core-set-aware evidence allocation")
        if any("selected_not_cited" in item for item in failure_types):
            fixes.extend(["primary citation first", "per-claim citation cap"])
        if "numeric_evidence_missing" in failure_types:
            fixes.append("numeric evidence completeness validator")
        if "comparison_side_missing" in failure_types:
            fixes.append("comparison-side completeness validator")
        if "claim_too_broad" in failure_types:
            fixes.append("compound claim decomposition")
        if "wrong_evidence_cited" in failure_types:
            fixes.append("shrink claim or return unsupported")
        output.append(
            {
                "question_id": claim["question_id"],
                "required_claim_id": claim["required_claim_id"],
                "generated_claim": claim["required_claim_text"],
                "core_relations": sorted(sets["core"]),
                "cited_relations": sorted(cited),
                "retrieved_relations": sorted(
                    relation_id
                    for relation_id, relation in relations.items()
                    if relation["retrieved_in_dev_v3_1"]
                ),
                "selected_relations": sorted(
                    relation_id
                    for relation_id, relation in relations.items()
                    if relation["selected_in_dev_v3_1"]
                ),
                "equivalent_relations": sorted(sets["equivalent"]),
                "human_support_labels": human_labels,
                "failure_types": failure_types,
                "root_cause": (
                    "claim-level citation/evidence failure"
                    if any(
                        item
                        not in {
                            "citation_fully_supported",
                            "citation_partially_supported",
                            "equivalent_valid_evidence_cited",
                            "metric_only_legacy_gold_issue",
                        }
                        for item in failure_types
                    )
                    else "metric-only or supported citation behavior"
                ),
                "generic_fix_candidate": sorted(set(fixes)),
                "retrieval_change_needed": any("not_retrieved" in item for item in failure_types),
                "selection_change_needed": any(
                    item in failure_types
                    for item in ("core_gold_retrieved_not_selected", "wrong_evidence_cited")
                ),
                "prompt_change_needed": any(
                    item in failure_types
                    for item in (
                        "core_gold_selected_not_cited",
                        "excessive_citation_dilution",
                        "claim_too_broad",
                    )
                ),
                "claim_decomposition_needed": "claim_too_broad" in failure_types,
                "metric_only_issue": "metric_only_legacy_gold_issue" in failure_types,
                "severity": (
                    "high"
                    if not detail["any_valid_evidence_hit"]
                    else "medium"
                    if not detail["core_set_complete"] and sets["core"]
                    else "low"
                ),
            }
        )
    return output


def improvement_plan(taxonomy: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        ("primary citation first", "Prefer the strongest claim-local evidence before supplements."),
        ("per-claim citation cap", "Limit dilution from weak extra citations."),
        (
            "compound claim decomposition",
            "Split claims whose minimum evidence set spans subclaims.",
        ),
        (
            "numeric evidence completeness validator",
            "Require cited text to contain required numeric facts.",
        ),
        ("comparison-side completeness validator", "Require evidence for every comparison side."),
        (
            "core-set-aware evidence allocation",
            "Allocate all members of a minimum complete evidence set.",
        ),
        ("original evidence first", "Use adjacent completion only as supporting context."),
        (
            "shrink claim or return unsupported",
            "Avoid unsupported breadth when evidence is incomplete.",
        ),
    ]
    rows = []
    for name, hypothesis in candidates:
        affected = [
            row
            for row in taxonomy
            if name in row["generic_fix_candidate"]
            or (
                name == "original evidence first"
                and "wrong_evidence_cited" in row["failure_types"]
            )
        ]
        rows.append(
            {
                "candidate": name,
                "hypothesis": hypothesis,
                "audit_evidence": sorted(
                    {failure for row in affected for failure in row["failure_types"]}
                ),
                "affected_claims": sorted(row["required_claim_id"] for row in affected),
                "affected_questions": sorted({row["question_id"] for row in affected}),
                "expected_recall_impact": "positive if offline failure stage is addressed",
                "expected_support_impact": "non-negative; requires live validation",
                "precision_risk": "medium",
                "token_impact": "low to moderate",
                "latency_impact": "low; no new provider call",
                "implementation_complexity": "medium",
                "generic": True,
                "no_gold_online_dependency": True,
                "requires_live_validation": True,
            }
        )
    return {
        "schema_version": "dev-v3-2-citation-improvement-plan-v1",
        "mode": "offline_design_only",
        "candidates": rows,
        "live_code_implemented": False,
        "gold_online_dependency": False,
        "human_label_online_dependency": False,
    }


def write_csv(experiments: list[dict[str, Any]]) -> None:
    fields = [
        "evaluation_version",
        "answerable_question_macro_exact_relation_recall",
        "required_claim_macro_exact_relation_recall",
        "micro_core_relation_recall",
        "claim_core_set_completion",
        "answerable_question_macro_core_set_completion",
        "claim_any_valid_evidence_recall",
        "answerable_question_macro_any_valid_evidence_recall",
        "equivalent_valid_evidence_hit_rate",
        "supporting_only_hit_rate",
        "incomplete_core_set_rate",
        "human_strict_citation_support",
        "human_lenient_citation_support",
        "historical_metric",
    ]
    with COMPARISON_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in experiments:
            writer.writerow({field: row.get(field) for field in fields})


def main() -> None:
    claims = read_jsonl(CLAIM_GOLD)
    if len(claims) != 27 or any(row["adjudication_status"] != "approved" for row in claims):
        raise RuntimeError("frozen 27/27 approved claim Gold is required")
    claims_by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in claims:
        claims_by_question[row["question_id"]].append(row)
    experiments = [
        calculate_experiment(experiment, claims_by_question)
        for experiment in load_experiments()
    ]
    by_version = {row["evaluation_version"]: row for row in experiments}
    dev2 = by_version["stage13_3_dev_v2"]
    dev31 = by_version["stage13_8_dev_v3_1"]
    comparison = compare_dev(dev2, dev31)
    output = {
        "schema_version": "claim-gold-citation-comparison-v1",
        "metric_status": "claim_gold_recalculated_diagnostic",
        "gold_version": "claim-evidence-gold-dev-v1",
        "fixed_answerable_questions": ANSWERABLE_IDS,
        "experiments": experiments,
        "dev_v2_vs_dev_v3_1": comparison,
        "historical_gate_modified": False,
        "stage13_8_historical_gate": "FAILED",
        "stage13_8_historical_citation_recall": 0.295,
        "dev_v2_historical_citation_recall": 0.29583333333333334,
        "historical_protection_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in HISTORICAL_FILES.items()
        },
        "limitations": [
            "AI-assisted manual claim-level Gold over fixed Dev claims only.",
            "New metrics are diagnostic and do not replace historical formal Gates.",
            "No Full-50 or Production extrapolation.",
        ],
    }
    COMPARISON_JSON.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(experiments)
    focus = {
        question_id: comparison["per_question"][question_id]
        for question_id in ("q001", "q004", "q015", "q019", "q050")
    }
    table_rows = "\n".join(
        "| {version} | {question:.6f} | {claim:.6f} | {micro:.6f} | "
        "{core:.6f} | {valid:.6f} |".format(
            version=row["evaluation_version"],
            question=row["answerable_question_macro_exact_relation_recall"],
            claim=row["required_claim_macro_exact_relation_recall"],
            micro=row["micro_core_relation_recall"],
            core=row["claim_core_set_completion"],
            valid=row["claim_any_valid_evidence_recall"],
        )
        for row in experiments
    )
    COMPARISON_DOC.write_text(
        f"""# Claim Gold citation comparison v1

Status: `claim_gold_recalculated_diagnostic`

| Experiment | Question exact | Claim macro | Micro core | Core set | Any valid |
|---|---:|---:|---:|---:|---:|
{table_rows}

Dev v2 versus Dev v3.1 outcomes: `{comparison['outcomes']}`.

Focus questions:

```json
{json.dumps(focus, ensure_ascii=False, indent=2)}
```

## Focus interpretation

- **q001:** Dev v3.1 increases any-valid evidence through equivalent relations but misses the
  complete two-relation attention/dependency core set. The compound claim remains a decomposition
  candidate.
- **q004:** Both versions hit two of three claim slots, but the GPU/Adam/warmup multi-relation
  training-config set is incomplete. The taxonomy separates not-retrieved from selected-not-cited
  members.
- **q015:** Neither version hits the adjudicated survey-location, ROUGE-limitation, and
  coordinate-ascent-limitation relations. Dev v3.1 contains wrong/insufficient evidence citations.
- **q019:** Neither version hits the exact numeric-range and complete model-shape relations.
  Numeric completeness and retrieval are the dominant failures.
- **q050:** Dev v2 is a fixed-denominator failure. Dev v3.1 hits BERT-side equivalent evidence, so
  any-valid exceeds exact recall, but the cross-paper comparison remains incomplete.

These metrics use AI-assisted manual claim-level Gold for 27 fixed Dev claims. They are diagnostic,
do not replace the Stage 13.8 historical failed Gate or its 0.295 recall, and cannot be extrapolated
to Full-50 or Production.
""",
        encoding="utf-8",
    )

    taxonomy = taxonomy_rows(dev31, claims)
    counts = Counter(
        failure for row in taxonomy for failure in row["failure_types"]
    )
    taxonomy_summary = {
        "schema_version": "dev-v3-1-citation-failure-taxonomy-v2",
        "metric_status": "claim_gold_recalculated_diagnostic",
        "records": len(taxonomy),
        "failure_type_counts": dict(sorted(counts.items())),
        "severity_counts": dict(Counter(row["severity"] for row in taxonomy)),
        "blocking_gold_ambiguity": False,
        "historical_gate_modified": False,
    }
    TAXONOMY_JSONL.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in taxonomy),
        encoding="utf-8",
    )
    TAXONOMY_JSON.write_text(
        json.dumps(taxonomy_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    TAXONOMY_DOC.write_text(
        "# Dev v3.1 citation failure taxonomy v2\n\n"
        "Status: `claim_gold_recalculated_diagnostic`\n\n"
        f"- Claims: {len(taxonomy)}\n"
        f"- Failure counts: `{dict(sorted(counts.items()))}`\n"
        "- Blocking Gold ambiguity: false\n"
        "- Historical Stage 13.8 Gate modified: false\n",
        encoding="utf-8",
    )
    plan = improvement_plan(taxonomy)
    PLAN_JSON.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PLAN_DOC.write_text(
        "# Dev v3.2 citation improvement plan v1\n\n"
        "Offline design only. No live code or model call was executed.\n\n"
        + "\n".join(
            f"- **{row['candidate']}**: {row['hypothesis']} "
            f"(affected claims: {len(row['affected_claims'])})"
            for row in plan["candidates"]
        )
        + "\n\nAll candidates are generic, have no online Gold/human-label dependency, and require "
        "explicitly authorized live validation.\n",
        encoding="utf-8",
    )
    non_metric_failures = sum(
        any(
            failure
            not in {
                "metric_only_legacy_gold_issue",
                "equivalent_valid_evidence_cited",
                "citation_fully_supported",
                "citation_partially_supported",
            }
            for failure in row["failure_types"]
        )
        for row in taxonomy
    )
    generic_evidence = sum(bool(row["generic_fix_candidate"]) for row in taxonomy)
    ready = non_metric_failures > 0 and generic_evidence > 0
    readiness = {
        "schema_version": "stage13-10-phase-b-readiness-v1",
        "ready_for_dev_v3_2": ready,
        "dev_v3_2_authorized": False,
        "dev_v3_2_executed": False,
        "blocking_gold_ambiguity": False,
        "reviewed_claims": 27,
        "claim_gold_frozen": True,
        "unified_recalculation_complete": True,
        "non_metric_failure_claims": non_metric_failures,
        "generic_fix_evidence_claims": generic_evidence,
        "gold_online_dependency": False,
        "human_label_online_dependency": False,
        "question_or_block_special_case": False,
        "historical_gate_modified": False,
        "decision": "READY_FOR_DEV_V3_2=true" if ready else "READY_FOR_DEV_V3_2=false",
    }
    READINESS_JSON.write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    READINESS_DOC.write_text(
        f"""# Stage 13.10 Phase B readiness

`{readiness['decision']}`

`DEV_V3_2_AUTHORIZED=false`

- Blocking Gold ambiguity: false
- Non-metric failure claims: {non_metric_failures}
- Claims with generic offline fix evidence: {generic_evidence}
- Historical Stage 13.8 Gate modified: false
- Dev v3.2 executed: false

This readiness decision permits only a future explicitly authorized controlled evaluation.
""",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "experiments": {
                    row["evaluation_version"]: {
                        key: row[key]
                        for key in (
                            "answerable_question_macro_exact_relation_recall",
                            "required_claim_macro_exact_relation_recall",
                            "micro_core_relation_recall",
                            "claim_core_set_completion",
                            "claim_any_valid_evidence_recall",
                        )
                    }
                    for row in experiments
                },
                "dev_v2_vs_dev_v3_1": comparison,
                "failure_type_counts": dict(sorted(counts.items())),
                "readiness": readiness,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
