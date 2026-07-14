# ruff: noqa: E501
"""Validate and summarize the Stage 11C.7 AI-assisted citation audit."""

import csv
import json
import subprocess
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

INPUT = Path("data/evaluation/citation-human-audit-sample-v1.jsonl")
GOLD = Path("data/evaluation/gold-set-v1.jsonl")
DIAGNOSTICS = Path("data/evaluation/qa-context-diagnostics-v1.json")
OUTPUT_JSON = Path("data/evaluation/citation-human-audit-summary-v1.json")
OUTPUT_CSV = Path("data/evaluation/citation-human-audit-summary-v1.csv")
OUTPUT_MD = Path("docs/citation-human-audit-summary-v1.md")

HUMAN_LABELS = (
    "fully_supported",
    "partially_supported",
    "related_but_insufficient",
    "unsupported",
    "gold_annotation_too_narrow",
)
AUTOMATED_LABELS = (
    "exact_gold_block",
    "same_gold_page",
    "semantic_support_non_gold",
    "weakly_related",
    "unsupported",
)
EXPECTED = Counter(
    {
        "fully_supported": 5,
        "partially_supported": 2,
        "related_but_insufficient": 7,
        "unsupported": 16,
        "gold_annotation_too_narrow": 0,
    }
)
AUTO_POSITIVE = {"exact_gold_block", "same_gold_page", "semantic_support_non_gold"}
ALLOWED_CHANGES = {
    "sample_id",
    "reviewer",
    "reviewed_at",
    "human_review_status",
    "human_label",
    "review_notes",
    "human_reviewer",
    "human_reviewed_at",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pair_key(row: dict) -> tuple:
    return (
        row["question_id"],
        row["claim_text"],
        row["cited_paper_id"],
        row["cited_page"],
        row["cited_block_id"],
    )


def validate(rows: list[dict]) -> dict:
    base_text = subprocess.check_output(
        ["git", "show", f"HEAD:{INPUT.as_posix()}"], text=True, encoding="utf-8"
    )
    base = [json.loads(line) for line in base_text.splitlines() if line.strip()]
    base_by_pair = {pair_key(row): row for row in base}
    immutable_changes = []
    for row in rows:
        previous = base_by_pair.get(pair_key(row))
        if previous is None:
            immutable_changes.append({"sample_id": row.get("sample_id"), "field": "record"})
            continue
        for field in set(previous) | set(row):
            if field not in ALLOWED_CHANGES and previous.get(field) != row.get(field):
                immutable_changes.append({"sample_id": row["sample_id"], "field": field})
    checks = {
        "record_count": len(rows) == 30,
        "sample_ids_unique": len({row.get("sample_id") for row in rows}) == 30,
        "sample_id_sequence": [row.get("sample_id") for row in rows]
        == [f"citation-audit-v1-{index:03d}" for index in range(1, 31)],
        "all_approved": all(row.get("human_review_status") == "approved" for row in rows),
        "pending_zero": not any(row.get("human_review_status") == "pending" for row in rows),
        "labels_valid": all(row.get("human_label") in HUMAN_LABELS for row in rows),
        "reviewer_complete": all(row.get("reviewer") for row in rows),
        "reviewed_at_complete": all(row.get("reviewed_at") for row in rows),
        "review_notes_complete": all(row.get("review_notes") for row in rows),
        "distribution_matches": Counter(row["human_label"] for row in rows) == +EXPECTED,
        "claim_citation_pairs_unique": len({pair_key(row) for row in rows}) == 30,
        "immutable_fields_unchanged": not immutable_changes,
        "review_aliases_preserved": all(
            row.get("reviewer") == row.get("human_reviewer")
            and row.get("reviewed_at") == row.get("human_reviewed_at")
            for row in rows
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"citation audit validation failed: {failed}; {immutable_changes}")
    return {"checks": checks, "immutable_changes": immutable_changes}


def enrich(rows: list[dict]) -> list[dict]:
    gold = {row["question_id"]: row for row in load_jsonl(GOLD)}
    diagnostics = json.loads(DIAGNOSTICS.read_text(encoding="utf-8"))
    retrieved = {
        row["question_id"]: row
        for row in diagnostics["runs"]
        if row["context_mode"] == "retrieved"
    }
    enriched = []
    for row in rows:
        item = dict(row)
        gold_row = gold[row["question_id"]]
        item["category"] = gold_row["category"]
        item["difficulty"] = gold_row["difficulty"]
        context = retrieved[row["question_id"]]["context"]
        item["context_rank"] = next(
            (
                rank
                for rank, context_item in enumerate(context, start=1)
                if context_item["paper_id"] == row["cited_paper_id"]
                and row["cited_block_id"] in context_item["block_ids"]
            ),
            None,
        )
        enriched.append(item)
    return enriched


def support_flags(label: str) -> tuple[bool, bool]:
    return label == "fully_supported", label in {"fully_supported", "partially_supported"}


def summarize_group(rows: list[dict]) -> dict:
    distribution = Counter(row["human_label"] for row in rows)
    strict = distribution["fully_supported"]
    lenient = strict + distribution["partially_supported"]
    return {
        "sample_count": len(rows),
        "human_label_distribution": {label: distribution[label] for label in HUMAN_LABELS},
        "strict_support_count": strict,
        "strict_support_rate": round(strict / len(rows), 6) if rows else None,
        "lenient_support_count": lenient,
        "lenient_support_rate": round(lenient / len(rows), 6) if rows else None,
        "fully_unsupported_count": distribution["unsupported"],
        "fully_unsupported_rate": round(distribution["unsupported"] / len(rows), 6)
        if rows
        else None,
        "related_but_insufficient_count": distribution["related_but_insufficient"],
        "related_but_insufficient_rate": round(
            distribution["related_but_insufficient"] / len(rows), 6
        )
        if rows
        else None,
        "gold_annotation_too_narrow_count": distribution["gold_annotation_too_narrow"],
        "gold_annotation_too_narrow_rate": round(
            distribution["gold_annotation_too_narrow"] / len(rows), 6
        )
        if rows
        else None,
    }


def grouped(rows: list[dict], field: str) -> dict:
    values: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        values[str(row.get(field))].append(row)
    return {key: summarize_group(items) for key, items in sorted(values.items())}


def automated_metrics(rows: list[dict]) -> dict:
    result = {}
    for automated_label in AUTOMATED_LABELS:
        items = [
            row
            for row in rows
            if row["automated_labels"]["classification"] == automated_label
        ]
        summary = summarize_group(items)
        predicts_support = automated_label in AUTO_POSITIVE
        strict_positive = sum(support_flags(row["human_label"])[0] for row in items)
        lenient_positive = sum(support_flags(row["human_label"])[1] for row in items)
        summary.update(
            {
                "automated_predicts_support": predicts_support,
                "strict_precision": round(strict_positive / len(items), 6) if items else None,
                "lenient_precision": round(lenient_positive / len(items), 6) if items else None,
                "strict_false_positive_count": len(items) - strict_positive
                if predicts_support
                else 0,
                "lenient_false_positive_count": len(items) - lenient_positive
                if predicts_support
                else 0,
                "strict_false_negative_count": strict_positive if not predicts_support else 0,
                "lenient_false_negative_count": lenient_positive if not predicts_support else 0,
                "unsupported_precision": round(
                    sum(row["human_label"] == "unsupported" for row in items) / len(items), 6
                )
                if items
                else None,
            }
        )
        false_examples = [
            {
                "sample_id": row["sample_id"],
                "question_id": row["question_id"],
                "human_label": row["human_label"],
                "review_notes": row["review_notes"],
            }
            for row in items
            if (predicts_support and not support_flags(row["human_label"])[1])
            or (not predicts_support and support_flags(row["human_label"])[1])
        ]
        summary["typical_misclassifications"] = false_examples[:3]
        result[automated_label] = summary
    return result


def confusion(rows: list[dict], *, lenient: bool) -> dict:
    counts = Counter()
    for row in rows:
        predicted = row["automated_labels"]["classification"] in AUTO_POSITIVE
        strict, broad = support_flags(row["human_label"])
        actual = broad if lenient else strict
        counts[(predicted, actual)] += 1
    return {
        "true_positive": counts[(True, True)],
        "false_positive": counts[(True, False)],
        "true_negative": counts[(False, False)],
        "false_negative": counts[(False, True)],
    }


def build_payload(rows: list[dict], validation: dict) -> dict:
    auto = automated_metrics(rows)
    semantic = auto["semantic_support_non_gold"]
    same_page = auto["same_gold_page"]
    weak = auto["weakly_related"]
    unsupported = auto["unsupported"]
    return {
        "status": "COMPLETED",
        "generated_at": datetime.now(UTC).isoformat(),
        "audit_type": "AI-assisted manual citation audit",
        "scope": "30-sample stratified review; not full-dataset human citation precision",
        "validation": validation,
        "overall": summarize_group(rows),
        "automated_label_metrics": auto,
        "confusion_matrix": {
            "strict": confusion(rows, lenient=False),
            "lenient": confusion(rows, lenient=True),
            "automated_label_by_human_label": {
                label: auto[label]["human_label_distribution"] for label in AUTOMATED_LABELS
            },
        },
        "grouped_metrics": {
            "automated_label": grouped(
                [
                    {**row, "automated_label": row["automated_labels"]["classification"]}
                    for row in rows
                ],
                "automated_label",
            ),
            "category": grouped(rows, "category"),
            "difficulty": grouped(rows, "difficulty"),
            "question_id": grouped(rows, "question_id"),
            "cited_paper": grouped(rows, "cited_paper_id"),
            "context_rank": grouped(rows, "context_rank"),
        },
        "key_findings": {
            "semantic_strict_precision": semantic["strict_precision"],
            "semantic_lenient_precision": semantic["lenient_precision"],
            "semantic_false_positive_lenient": semantic["lenient_false_positive_count"],
            "same_page_human_unsupported": same_page["fully_unsupported_count"],
            "weak_human_unsupported": weak["fully_unsupported_count"],
            "automated_unsupported_precision": unsupported["unsupported_precision"],
            "gold_annotation_too_narrow": summarize_group(rows)[
                "gold_annotation_too_narrow_count"
            ],
            "semantic_support_overestimates_human_support": (
                semantic["lenient_precision"] is not None
                and semantic["lenient_precision"] < 0.5
            ),
        },
        "records": rows,
    }


def write_outputs(payload: dict) -> None:
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for label in AUTOMATED_LABELS:
        metric = payload["automated_label_metrics"][label]
        rows.append(
            {
                "automated_label": label,
                "sample_count": metric["sample_count"],
                "strict_precision": metric["strict_precision"],
                "lenient_precision": metric["lenient_precision"],
                "strict_false_positive_count": metric["strict_false_positive_count"],
                "lenient_false_positive_count": metric["lenient_false_positive_count"],
                "strict_false_negative_count": metric["strict_false_negative_count"],
                "lenient_false_negative_count": metric["lenient_false_negative_count"],
                "human_unsupported_count": metric["fully_unsupported_count"],
                "unsupported_precision": metric["unsupported_precision"],
            }
        )
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    overall = payload["overall"]
    findings = payload["key_findings"]
    lines = [
        "# Citation Human Audit Summary v1",
        "",
        "> AI-assisted manual citation audit, 30-sample stratified review. This is not an independent blind review and does not confirm full-dataset human citation precision.",
        "",
        "## Validation",
        "",
        "- Records: 30/30; approved: 30; pending: 0; unique sample IDs and claim-citation pairs: 30.",
        "- Reviewer, reviewed_at, and review_notes are complete. Claim, citation, Gold evidence, and automated labels are unchanged.",
        "",
        "## Human support rates",
        "",
        f"- Strict support (`fully_supported` only): {overall['strict_support_count']}/30 = {overall['strict_support_rate']:.1%}.",
        f"- Lenient support (`fully_supported` + `partially_supported`): {overall['lenient_support_count']}/30 = {overall['lenient_support_rate']:.1%}.",
        f"- Related but insufficient: {overall['related_but_insufficient_count']}/30 = {overall['related_but_insufficient_rate']:.1%}.",
        f"- Fully unsupported: {overall['fully_unsupported_count']}/30 = {overall['fully_unsupported_rate']:.1%}.",
        f"- Gold annotation too narrow: {overall['gold_annotation_too_narrow_count']}/30 = {overall['gold_annotation_too_narrow_rate']:.1%}.",
        "",
        "## Automated label calibration",
        "",
        "| Automated label | N | Strict precision | Lenient precision | Human unsupported |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in AUTOMATED_LABELS:
        metric = payload["automated_label_metrics"][label]
        strict = "N/A" if metric["strict_precision"] is None else f"{metric['strict_precision']:.1%}"
        lenient = "N/A" if metric["lenient_precision"] is None else f"{metric['lenient_precision']:.1%}"
        lines.append(
            f"| `{label}` | {metric['sample_count']} | {strict} | {lenient} | {metric['fully_unsupported_count']} |"
        )
    lines += [
        "",
        "## Confusion matrices",
        "",
        f"- Strict: `{json.dumps(payload['confusion_matrix']['strict'], sort_keys=True)}`",
        f"- Lenient: `{json.dumps(payload['confusion_matrix']['lenient'], sort_keys=True)}`",
        "",
        "## Findings",
        "",
        f"- Token-set semantic support strict/lenient precision: {findings['semantic_strict_precision']:.1%}/{findings['semantic_lenient_precision']:.1%}; lenient false positives: {findings['semantic_false_positive_lenient']}.",
        f"- Same-page samples judged fully unsupported: {findings['same_page_human_unsupported']}.",
        f"- Weakly-related samples judged fully unsupported: {findings['weak_human_unsupported']}.",
        f"- Automated unsupported negative precision: {findings['automated_unsupported_precision']:.1%}.",
        f"- Gold annotation too narrow: {findings['gold_annotation_too_narrow']}.",
        "",
        "The prior 81.6% semantic-support value is a token-overlap signal, not citation correctness. In this stratified sample, only 5 citations are fully supported and 7 are fully or partially supported; 23 are related-but-insufficient or unsupported. The automated semantic label materially overestimates human-confirmed support.",
    ]
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = load_jsonl(INPUT)
    validation = validate(rows)
    enriched = enrich(rows)
    payload = build_payload(enriched, validation)
    write_outputs(payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "overall": payload["overall"],
                "key_findings": payload["key_findings"],
            }
        )
    )


if __name__ == "__main__":
    main()
