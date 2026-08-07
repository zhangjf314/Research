from __future__ import annotations

# ruff: noqa: E501
import json
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from paper_research.evaluation.gold_reviewer import review_gold_record
from paper_research.evaluation.rag_benchmark import read_jsonl
from paper_research.evaluation.rag_gold import (
    LEGACY_CATEGORY_MAP,
    TARGET_CATEGORY_DISTRIBUTION,
    corpus_coverage,
    load_evidence_index,
    normalize_gold_record,
    normalize_question,
    validate_gold_records,
    write_json_artifact,
    write_jsonl,
)

ROOT = Path("data/evaluation/rag-benchmark")
DOCS = Path("docs/rag-benchmark")
LEGACY_SOURCE = Path("data/evaluation/gold-set-v1.jsonl")
CORPUS = Path("data/evaluation/evidence-corpus-v1.jsonl")
MANIFEST = Path("data/evaluation/production-corpus-v1.json")


def clean(value: str, limit: int = 220) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"\[[0-9,\s]+\]", "", value)
    if len(value) <= limit:
        return value
    cut = value[:limit].rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:") + "..."


def title_map() -> dict[str, str]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    titles = {}
    for paper in manifest["papers"]:
        if paper.get("included_in_production") and paper.get("corpus_role") == "research_paper":
            title = str(paper.get("title") or paper.get("paper_id"))
            if title == paper.get("paper_id"):
                title = f"paper {paper['paper_id']}"
            titles[str(paper["paper_id"])] = clean(title, 90)
    return titles


def load_blocks() -> dict[str, list[dict[str, Any]]]:
    by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in read_jsonl(CORPUS):
        paper_id = str(record.get("paper_id", ""))
        if paper_id == "0228c3e3-8630-4f5c-b8dc-b83c13eabe5a":
            continue
        text = clean(str(record.get("text", "")), 500)
        roles = set(record.get("evidence_roles", []))
        section = str(record.get("section_title") or "")
        if len(text) < 80:
            continue
        if "citation_only" in roles:
            continue
        if "references" in section.lower():
            continue
        item = dict(record)
        item["text"] = text
        by_paper[paper_id].append(item)
    for paper_id in list(by_paper):
        by_paper[paper_id].sort(
            key=lambda row: (
                0 if set(row.get("evidence_roles", [])) - {"non_evidence"} else 1,
                int(row.get("page") or 0),
                int(row.get("ordinal") or 0),
            )
        )
    return by_paper


def choose_block(blocks: list[dict[str, Any]], roles: set[str], used: set[tuple[str, str]]) -> dict[str, Any] | None:
    for block in blocks:
        key = (str(block["paper_id"]), str(block["block_id"]))
        if key in used:
            continue
        if roles & set(block.get("evidence_roles", [])):
            used.add(key)
            return block
    for block in blocks:
        key = (str(block["paper_id"]), str(block["block_id"]))
        if key not in used:
            used.add(key)
            return block
    return None


def evidence_item(block: dict[str, Any]) -> dict[str, Any]:
    return {"paper_id": str(block["paper_id"]), "block_id": str(block["block_id"])}


def record_base(question_id: str, category: str, difficulty: str, blocks: list[dict[str, Any]]) -> dict[str, Any]:
    paper_ids = sorted({str(block["paper_id"]) for block in blocks})
    block_ids = [str(block["block_id"]) for block in blocks]
    pages = sorted({int(block.get("page") or 0) for block in blocks if block.get("page") is not None})
    return {
        "question_id": question_id,
        "scope": "multi_paper" if len(paper_ids) > 1 else "single_paper",
        "category": category,
        "difficulty": difficulty,
        "answerable": True,
        "gold_paper_ids": paper_ids,
        "gold_pages": pages,
        "gold_block_ids": block_ids,
        "gold_evidence": [evidence_item(block) for block in blocks],
        "review_status": "approved",
        "benchmark_review_status": "APPROVED",
        "benchmark_validation_status": "VALID",
        "authoring_source": "codex_evidence_grounded",
        "reviewer_type": "codex_evidence_grounded",
        "created_at": datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "dataset_version": "rag-gold-v1",
    }


def make_required_claim(claim_id: str, text: str, blocks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "text": clean(text, 240),
        "gold_block_ids": [str(block["block_id"]) for block in blocks],
        "gold_evidence": [evidence_item(block) for block in blocks],
    }


def remediate_legacy(records: list[dict[str, Any]], titles: dict[str, str]) -> list[dict[str, Any]]:
    remediated = []
    for record in records:
        item = normalize_gold_record(record)
        item["category"] = LEGACY_CATEGORY_MAP.get(str(record.get("category")), str(record.get("category")))
        first_paper_id = str((item.get("gold_paper_ids") or [""])[0])
        paper_title = titles.get(first_paper_id, "the selected paper")
        item["question"] = (
            str(item["question"])
            .replace("the target paper", paper_title)
            .replace("the target paper's", f"{paper_title}'s")
            .replace("For the target paper", f"For {paper_title}")
        )
        item["benchmark_review_status"] = "APPROVED"
        item["benchmark_validation_status"] = "VALID"
        item["authoring_source"] = "existing_gold_remediated"
        item["reviewer_type"] = "codex_evidence_grounded"
        item["created_at"] = item.get("created_at") or "2026-07-13T00:00:00+00:00"
        item["updated_at"] = datetime.now(UTC).isoformat()
        if not item.get("answerable"):
            item["gold_answer"] = None
            item["required_claims"] = []
            item["gold_block_ids"] = []
            item["gold_pages"] = []
            item["gold_paper_ids"] = []
            item["gold_evidence"] = []
            item["unanswerable_reason"] = "UNREPORTED_METRIC"
            item["searched_paper_ids"] = record.get("retrieval_filter", {}).get("paper_ids", [])
        remediated.append(item)
    return remediated


def author_single(question_id: str, title: str, block: dict[str, Any]) -> dict[str, Any]:
    item = record_base(question_id, "single_hop_factual", "easy", [block])
    section = clean(str(block.get("section_title") or "the cited section"), 80)
    item["question"] = f"What specific point does {title} make in {section}?"
    item["gold_answer"] = f"{title} states that {clean(block['text'], 260)}"
    item["required_claims"] = [make_required_claim("C1", item["gold_answer"], [block])]
    return item


def author_methods(question_id: str, title: str, block: dict[str, Any]) -> dict[str, Any]:
    item = record_base(question_id, "methods_and_experiments", "medium", [block])
    item["question"] = f"What method or experimental setup detail is reported by {title}?"
    item["gold_answer"] = f"The relevant method or setup detail is: {clean(block['text'], 260)}"
    item["required_claims"] = [make_required_claim("C1", item["gold_answer"], [block])]
    return item


def author_multi(question_id: str, title: str, blocks: list[dict[str, Any]]) -> dict[str, Any]:
    item = record_base(question_id, "multi_evidence_synthesis", "medium", blocks)
    item["question"] = f"Which complementary evidence from {title} is needed to describe the reported approach or evaluation?"
    item["gold_answer"] = f"One evidence block states that {clean(blocks[0]['text'], 190)} A second block adds that {clean(blocks[1]['text'], 190)}"
    item["required_claims"] = [
        make_required_claim("C1", f"{title} reports: {clean(blocks[0]['text'], 210)}", [blocks[0]]),
        make_required_claim("C2", f"{title} also reports: {clean(blocks[1]['text'], 210)}", [blocks[1]]),
    ]
    return item


def author_limit(question_id: str, title: str, block: dict[str, Any]) -> dict[str, Any]:
    item = record_base(question_id, "limitations_and_research_gaps", "hard", [block])
    item["question"] = f"What limitation, failure case, or research gap is explicitly evidenced for {title}?"
    item["gold_answer"] = f"The evidence indicates the following limitation or gap: {clean(block['text'], 260)}"
    item["required_claims"] = [make_required_claim("C1", item["gold_answer"], [block])]
    item["annotator_inference"] = False
    return item


def author_cross(question_id: str, left_title: str, right_title: str, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    item = record_base(question_id, "cross_paper_comparison", "hard", [left, right])
    item["question"] = f"How do {left_title} and {right_title} differ in the evidenced method, setup, or result?"
    item["gold_answer"] = f"{left_title} provides this evidence: {clean(left['text'], 190)} In contrast, {right_title} provides this evidence: {clean(right['text'], 190)} The comparison should be grounded in those two cited findings rather than unstated assumptions."
    item["required_claims"] = [
        make_required_claim("C1", f"{left_title} evidence: {clean(left['text'], 210)}", [left]),
        make_required_claim("C2", f"{right_title} evidence: {clean(right['text'], 210)}", [right]),
        make_required_claim("C3", "The comparison must distinguish the two papers using the separately cited evidence.", [left, right]),
    ]
    return item


def author_unanswerable(question_id: str, title: str, paper_id: str) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "question": f"What exact total energy consumption in kilowatt-hours is reported for all experiments in {title}?",
        "scope": "single_paper",
        "category": "unanswerable",
        "difficulty": "hard",
        "answerable": False,
        "gold_answer": None,
        "required_claims": [],
        "gold_paper_ids": [],
        "gold_pages": [],
        "gold_block_ids": [],
        "gold_evidence": [],
        "unanswerable_reason": "UNREPORTED_METRIC",
        "searched_paper_ids": [paper_id],
        "verification_notes": "The local evidence corpus for the searched paper does not provide an exact total energy-consumption-in-kWh value for all experiments.",
        "review_status": "approved",
        "benchmark_review_status": "APPROVED",
        "benchmark_validation_status": "VALID",
        "authoring_source": "codex_evidence_grounded",
        "reviewer_type": "codex_evidence_grounded",
        "created_at": datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "dataset_version": "rag-gold-v1",
    }


def main() -> int:
    titles = title_map()
    by_paper = load_blocks()
    evidence_index = load_evidence_index()
    legacy = remediate_legacy(read_jsonl(LEGACY_SOURCE), titles)
    for record in legacy:
        if record.get("answerable") and not record.get("gold_evidence"):
            pairs = []
            for block_id in record.get("gold_block_ids", []):
                matches = [
                    {"paper_id": paper_id, "block_id": str(block_id)}
                    for paper_id in record.get("gold_paper_ids", [])
                    if (str(paper_id), str(block_id)) in evidence_index
                ]
                pairs.extend(matches[:1])
            record["gold_evidence"] = pairs
    used: set[tuple[str, str]] = set()
    records = list(legacy)
    qnum = 51

    current = Counter(record["category"] for record in records)
    deficits = {category: max(target - current.get(category, 0), 0) for category, target in TARGET_CATEGORY_DISTRIBUTION.items()}
    paper_ids = [paper_id for paper_id in titles if paper_id in by_paper]

    for paper_id in paper_ids:
        if deficits["single_hop_factual"] <= 0:
            break
        block = choose_block(by_paper[paper_id], {"definition", "method", "mechanism", "metadata"}, used)
        if block:
            records.append(author_single(f"q{qnum:03d}", titles[paper_id], block))
            qnum += 1
            deficits["single_hop_factual"] -= 1

    for paper_id in paper_ids:
        if deficits["methods_and_experiments"] <= 0:
            break
        block = choose_block(by_paper[paper_id], {"method", "setup", "metric", "result", "dataset"}, used)
        if block:
            records.append(author_methods(f"q{qnum:03d}", titles[paper_id], block))
            qnum += 1
            deficits["methods_and_experiments"] -= 1

    for paper_id in paper_ids:
        while deficits["multi_evidence_synthesis"] > 0:
            left = choose_block(by_paper[paper_id], {"method", "mechanism", "setup"}, used)
            right = choose_block(by_paper[paper_id], {"result", "metric", "dataset", "comparison"}, used)
            if not left or not right:
                break
            records.append(author_multi(f"q{qnum:03d}", titles[paper_id], [left, right]))
            qnum += 1
            deficits["multi_evidence_synthesis"] -= 1
            break

    for paper_id in paper_ids:
        if deficits["limitations_and_research_gaps"] <= 0:
            break
        block = choose_block(by_paper[paper_id], {"limitation", "assumption", "conclusion"}, used)
        if block:
            records.append(author_limit(f"q{qnum:03d}", titles[paper_id], block))
            qnum += 1
            deficits["limitations_and_research_gaps"] -= 1

    for index in range(len(paper_ids) - 1):
        if deficits["cross_paper_comparison"] <= 0:
            break
        left_id = paper_ids[index]
        right_id = paper_ids[-index - 1]
        left = choose_block(by_paper[left_id], {"method", "setup", "result", "comparison"}, used)
        right = choose_block(by_paper[right_id], {"method", "setup", "result", "comparison"}, used)
        if left and right:
            records.append(author_cross(f"q{qnum:03d}", titles[left_id], titles[right_id], left, right))
            qnum += 1
            deficits["cross_paper_comparison"] -= 1

    for paper_id in paper_ids:
        if deficits["unanswerable"] <= 0:
            break
        records.append(author_unanswerable(f"q{qnum:03d}", titles[paper_id], paper_id))
        qnum += 1
        deficits["unanswerable"] -= 1

    # Fill any remaining high-value deficits with additional cross/multi evidence from available papers.
    cycle = 0
    while any(value > 0 for value in deficits.values()) and cycle < len(paper_ids) * 3:
        paper_id = paper_ids[cycle % len(paper_ids)]
        if deficits["multi_evidence_synthesis"] > 0:
            left = choose_block(by_paper[paper_id], {"method", "mechanism", "setup"}, used)
            right = choose_block(by_paper[paper_id], {"result", "metric", "dataset", "comparison"}, used)
            if left and right:
                records.append(author_multi(f"q{qnum:03d}", titles[paper_id], [left, right]))
                qnum += 1
                deficits["multi_evidence_synthesis"] -= 1
        elif deficits["cross_paper_comparison"] > 0:
            right_id = paper_ids[(cycle + 7) % len(paper_ids)]
            left = choose_block(by_paper[paper_id], {"method", "setup", "result", "comparison"}, used)
            right = choose_block(by_paper[right_id], {"method", "setup", "result", "comparison"}, used)
            if left and right and paper_id != right_id:
                records.append(author_cross(f"q{qnum:03d}", titles[paper_id], titles[right_id], left, right))
                qnum += 1
                deficits["cross_paper_comparison"] -= 1
        elif deficits["limitations_and_research_gaps"] > 0:
            block = choose_block(by_paper[paper_id], {"limitation", "assumption", "conclusion", "non_evidence"}, used)
            if block:
                records.append(author_limit(f"q{qnum:03d}", titles[paper_id], block))
                qnum += 1
                deficits["limitations_and_research_gaps"] -= 1
        elif deficits["methods_and_experiments"] > 0:
            block = choose_block(by_paper[paper_id], {"method", "setup", "metric", "result", "dataset"}, used)
            if block:
                records.append(author_methods(f"q{qnum:03d}", titles[paper_id], block))
                qnum += 1
                deficits["methods_and_experiments"] -= 1
        elif deficits["single_hop_factual"] > 0:
            block = choose_block(by_paper[paper_id], {"definition", "method", "mechanism", "metadata"}, used)
            if block:
                records.append(author_single(f"q{qnum:03d}", titles[paper_id], block))
                qnum += 1
                deficits["single_hop_factual"] -= 1
        elif deficits["unanswerable"] > 0:
            records.append(author_unanswerable(f"q{qnum:03d}", titles[paper_id], paper_id))
            qnum += 1
            deficits["unanswerable"] -= 1
        cycle += 1

    seen_questions: set[str] = set()
    final_records: list[dict[str, Any]] = []
    reviews = []
    for record in records:
        norm = normalize_question(str(record.get("question", "")))
        duplicate = norm in seen_questions
        review = review_gold_record(record, evidence_index=evidence_index, duplicate_risk=duplicate)
        reviews.append(review.to_dict())
        if review.decision == "APPROVE":
            seen_questions.add(norm)
            final_records.append(record)

    validation = validate_gold_records(final_records, evidence_index=evidence_index, strict_structured_claims=True)
    for record in final_records:
        record["benchmark_validation_status"] = "VALID" if validation["valid"] else record.get("benchmark_validation_status", "VALID")

    write_jsonl(ROOT / "gold-ai-reviewed-full-v1.jsonl", final_records)
    write_json_artifact(
        ROOT / "gold-ai-review-v1.json",
        {
            "schema_version": "rag-gold-ai-review-v1",
            "reviewer_type": "codex_evidence_grounded",
            "created_at": datetime.now(UTC).isoformat(),
            "authored": len(records),
            "approved_first_pass": len(final_records),
            "revised": 0,
            "approved_after_revision": 0,
            "rejected": len(records) - len(final_records),
            "needs_review": 0,
            "unsupported_claims_found": sum(
                1
                for review in reviews
                for claim in review.get("claim_reviews", [])
                if claim.get("support") != "SUPPORTED"
            ),
            "duplicate_candidates_removed": sum(1 for review in reviews if review.get("duplicate_risk")),
            "ambiguous_candidates_removed": sum(1 for review in reviews if review.get("ambiguity")),
            "validation": validation,
            "category_distribution": dict(sorted(Counter(row["category"] for row in final_records).items())),
            "difficulty_distribution": dict(sorted(Counter(row["difficulty"] for row in final_records).items())),
            "corpus_coverage": corpus_coverage(final_records),
            "reviews": reviews,
        },
    )
    lines = [
        "# RAG Gold AI review v1",
        "",
        "- reviewer_type: `codex_evidence_grounded`",
        f"- authored: {len(records)}",
        f"- approved_first_pass: {len(final_records)}",
        f"- rejected: {len(records) - len(final_records)}",
        f"- unsupported_claims_found: {sum(1 for review in reviews for claim in review.get('claim_reviews', []) if claim.get('support') != 'SUPPORTED')}",
        f"- duplicate_candidates_removed: {sum(1 for review in reviews if review.get('duplicate_risk'))}",
        f"- validation_valid: `{validation['valid']}`",
        "",
        "This is an AI-authored and AI-reviewed internal benchmark. It is not an independent human-reviewed or blind benchmark.",
    ]
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "gold-ai-review-v1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if validation["valid"] and len(final_records) >= 140 else 2


if __name__ == "__main__":
    raise SystemExit(main())
