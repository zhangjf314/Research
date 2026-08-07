from __future__ import annotations

# ruff: noqa: E501
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from paper_research.evaluation.rag_benchmark import canonical_json_hash, read_jsonl, write_json

VALID_EXPANSION_CATEGORIES = {
    "single_hop_factual",
    "multi_evidence_synthesis",
    "cross_paper_comparison",
    "methods_and_experiments",
    "limitations_and_research_gaps",
    "unanswerable",
}

VALID_LEGACY_CATEGORIES = {
    "algorithm_steps",
    "experiment_results",
    "experiment_setup",
    "limitations",
    "method",
    "multi_paper_comparison",
    "paper_contributions",
    "research_background",
    "unanswerable",
}

VALID_DIFFICULTIES = {"easy", "medium", "hard"}

VALID_UNANSWERABLE_REASONS = {
    "MISSING_EVIDENCE",
    "UNSUPPORTED_COMPARISON",
    "UNREPORTED_METRIC",
    "OUTSIDE_CORPUS",
    "FALSE_PREMISE",
}

TARGET_CATEGORY_DISTRIBUTION = {
    "single_hop_factual": 30,
    "multi_evidence_synthesis": 30,
    "cross_paper_comparison": 30,
    "methods_and_experiments": 25,
    "limitations_and_research_gaps": 20,
    "unanswerable": 15,
}

LEGACY_CATEGORY_MAP = {
    "algorithm_steps": "methods_and_experiments",
    "experiment_results": "methods_and_experiments",
    "experiment_setup": "methods_and_experiments",
    "limitations": "limitations_and_research_gaps",
    "method": "single_hop_factual",
    "multi_paper_comparison": "cross_paper_comparison",
    "paper_contributions": "single_hop_factual",
    "research_background": "single_hop_factual",
    "unanswerable": "unanswerable",
}


def normalize_question(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def token_set(text: str) -> set[str]:
    return set(normalize_question(text).split())


def overlap_ratio(left: str, right: str) -> float:
    left_tokens = token_set(left)
    right_tokens = token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens))


def load_evidence_index(root: Path = Path("data/reports/parsing-audit")) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    if not root.exists():
        return index
    for blocks_path in root.glob("*/paper_blocks.jsonl"):
        paper_id = blocks_path.parent.name
        for record in read_jsonl(blocks_path):
            block_id = str(record.get("block_id", ""))
            if block_id:
                index[(paper_id, block_id)] = record
    return index


def normalize_required_claims(record: dict[str, Any]) -> list[dict[str, Any]]:
    claims = record.get("required_claims") or []
    normalized: list[dict[str, Any]] = []
    question_blocks = [str(value) for value in record.get("gold_block_ids", [])]
    for index, claim in enumerate(claims, start=1):
        if isinstance(claim, dict):
            claim_id = str(claim.get("claim_id") or f"C{index}")
            text = str(claim.get("text") or "")
            claim_blocks = [str(value) for value in claim.get("gold_block_ids", [])]
        else:
            claim_id = f"C{index}"
            text = str(claim)
            claim_blocks = question_blocks
        normalized.append(
            {
                "claim_id": claim_id,
                "text": text,
                "gold_block_ids": claim_blocks,
            }
        )
    return normalized


def normalize_gold_record(record: dict[str, Any], *, dataset_version: str = "rag-gold-v1") -> dict[str, Any]:
    normalized = dict(record)
    normalized["dataset_version"] = dataset_version
    normalized["authoring_source"] = normalized.get("authoring_source") or "existing_gold"
    normalized["required_claims"] = normalize_required_claims(record)
    return normalized


def validate_gold_records(
    records: list[dict[str, Any]],
    *,
    evidence_index: dict[tuple[str, str], dict[str, Any]] | None = None,
    strict_structured_claims: bool = True,
    near_duplicate_threshold: float = 0.82,
) -> dict[str, Any]:
    evidence_index = evidence_index or load_evidence_index()
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    question_ids: Counter[str] = Counter(str(row.get("question_id", "")) for row in records)
    normalized_questions: Counter[str] = Counter(normalize_question(str(row.get("question", ""))) for row in records)

    for question_id, count in question_ids.items():
        if not question_id:
            errors.append({"type": "missing_question_id", "question_id": question_id})
        elif count > 1:
            errors.append({"type": "duplicate_question_id", "question_id": question_id, "count": count})

    for normalized, count in normalized_questions.items():
        if normalized and count > 1:
            errors.append({"type": "duplicate_question", "normalized_question": normalized, "count": count})

    near_duplicates: list[dict[str, Any]] = []
    for left_index, left in enumerate(records):
        for right in records[left_index + 1 :]:
            score = overlap_ratio(str(left.get("question", "")), str(right.get("question", "")))
            if score >= near_duplicate_threshold:
                near_duplicates.append(
                    {
                        "left_question_id": left.get("question_id"),
                        "right_question_id": right.get("question_id"),
                        "overlap": round(score, 6),
                    }
                )

    for record in records:
        qid = str(record.get("question_id", ""))
        question = str(record.get("question", "")).strip()
        category = str(record.get("category", ""))
        difficulty = str(record.get("difficulty", ""))
        answerable = bool(record.get("answerable"))
        review_status = str(record.get("review_status", ""))
        if not question:
            errors.append({"type": "empty_question", "question_id": qid})
        if category not in VALID_LEGACY_CATEGORIES and category not in VALID_EXPANSION_CATEGORIES:
            errors.append({"type": "invalid_category", "question_id": qid, "category": category})
        if difficulty not in VALID_DIFFICULTIES:
            errors.append({"type": "invalid_difficulty", "question_id": qid, "difficulty": difficulty})
        if review_status != "approved":
            warnings.append({"type": "not_approved", "question_id": qid, "review_status": review_status})

        paper_ids = [str(value) for value in record.get("gold_paper_ids", [])]
        block_ids = [str(value) for value in record.get("gold_block_ids", [])]
        pages = record.get("gold_pages", [])
        claims = record.get("required_claims") or []
        structured_claims = normalize_required_claims(record)

        if answerable:
            if not str(record.get("gold_answer", "")).strip():
                errors.append({"type": "answerable_missing_gold_answer", "question_id": qid})
            if not claims:
                errors.append({"type": "answerable_missing_required_claims", "question_id": qid})
            if strict_structured_claims and any(not isinstance(claim, dict) for claim in claims):
                errors.append({"type": "required_claims_not_structured", "question_id": qid})
            if not paper_ids:
                errors.append({"type": "answerable_missing_gold_paper_ids", "question_id": qid})
            if not block_ids:
                errors.append({"type": "answerable_missing_gold_block_ids", "question_id": qid})
            if not pages:
                errors.append({"type": "answerable_missing_gold_pages", "question_id": qid})
            for claim in structured_claims:
                if not claim["text"].strip():
                    errors.append({"type": "required_claim_empty_text", "question_id": qid, "claim_id": claim["claim_id"]})
                if not claim["gold_block_ids"]:
                    errors.append({"type": "required_claim_missing_evidence", "question_id": qid, "claim_id": claim["claim_id"]})
        else:
            if block_ids:
                errors.append({"type": "unanswerable_has_gold_block_ids", "question_id": qid})
            if record.get("gold_answer"):
                errors.append({"type": "unanswerable_has_gold_answer", "question_id": qid})
            reason = str(record.get("unanswerable_reason", ""))
            if reason not in VALID_UNANSWERABLE_REASONS:
                errors.append({"type": "missing_or_invalid_unanswerable_reason", "question_id": qid, "reason": reason})

        for paper_id in paper_ids:
            for block_id in block_ids:
                if (paper_id, block_id) not in evidence_index:
                    errors.append(
                        {
                            "type": "gold_block_not_found",
                            "question_id": qid,
                            "paper_id": paper_id,
                            "block_id": block_id,
                        }
                    )

    return {
        "record_count": len(records),
        "approved_count": sum(1 for row in records if row.get("review_status") == "approved"),
        "answerable_count": sum(1 for row in records if row.get("answerable")),
        "unanswerable_count": sum(1 for row in records if not row.get("answerable")),
        "duplicate_question_count": sum(1 for item in normalized_questions.values() if item > 1),
        "near_duplicate_question_count": len(near_duplicates),
        "near_duplicates": near_duplicates,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "valid": not errors,
    }


def expansion_plan(records: list[dict[str, Any]]) -> dict[str, Any]:
    current: Counter[str] = Counter()
    for record in records:
        mapped = LEGACY_CATEGORY_MAP.get(str(record.get("category", "")), str(record.get("category", "")))
        current[mapped] += 1
    rows = []
    for category, target in TARGET_CATEGORY_DISTRIBUTION.items():
        current_count = current.get(category, 0)
        rows.append(
            {
                "category": category,
                "current": current_count,
                "target": target,
                "deficit": max(target - current_count, 0),
            }
        )
    difficulty = Counter(str(row.get("difficulty", "")) for row in records)
    return {
        "schema_version": "rag-gold-expansion-plan-v1",
        "current_total": len(records),
        "target_total": sum(TARGET_CATEGORY_DISTRIBUTION.values()),
        "questions_to_add": max(sum(TARGET_CATEGORY_DISTRIBUTION.values()) - len(records), 0),
        "category_plan": rows,
        "difficulty_current": dict(sorted(difficulty.items())),
        "difficulty_target_guidance": {
            "easy": "25-30%",
            "medium": "40-50%",
            "hard": "25-30%",
        },
        "unanswerable_reason_types": sorted(VALID_UNANSWERABLE_REASONS),
        "approval_policy": {
            "candidate_default_review_status": "pending",
            "approved_requires_human_review": True,
            "llm_drafts_must_not_be_auto_approved": True,
        },
    }


def corpus_coverage(records: list[dict[str, Any]], corpus_manifest_path: Path = Path("data/evaluation/production-corpus-v1.json")) -> dict[str, Any]:
    manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8")) if corpus_manifest_path.exists() else {}
    papers = manifest.get("papers", [])
    included = [row for row in papers if row.get("included_in_production") and row.get("corpus_role") == "research_paper"]
    by_paper: dict[str, dict[str, Any]] = {}
    for paper in included:
        by_paper[str(paper.get("paper_id"))] = {
            "paper_id": paper.get("paper_id"),
            "title": paper.get("title"),
            "question_count": 0,
            "evidence_block_count": 0,
            "category_distribution": {},
        }
    for record in records:
        category = str(record.get("category", ""))
        for paper_id in record.get("gold_paper_ids", []):
            item = by_paper.setdefault(
                str(paper_id),
                {
                    "paper_id": str(paper_id),
                    "title": str(paper_id),
                    "question_count": 0,
                    "evidence_block_count": 0,
                    "category_distribution": {},
                },
            )
            item["question_count"] += 1
            item["evidence_block_count"] += len(record.get("gold_block_ids", []))
            counter = Counter(item["category_distribution"])
            counter[category] += 1
            item["category_distribution"] = dict(sorted(counter.items()))
    covered = [item for item in by_paper.values() if item["question_count"] > 0]
    return {
        "corpus_paper_count": len(included),
        "papers_covered": len(covered),
        "papers": sorted(by_paper.values(), key=lambda row: str(row["paper_id"])),
    }


def stratified_split(
    records: list[dict[str, Any]],
    *,
    seed: int = 20260807,
    test_ratio: float = 1 / 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import random

    groups: dict[tuple[str, str, bool], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[
            (
                str(record.get("category", "")),
                str(record.get("difficulty", "")),
                bool(record.get("answerable")),
            )
        ].append(record)
    rng = random.Random(seed)
    dev: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for group_records in groups.values():
        shuffled = list(group_records)
        rng.shuffle(shuffled)
        test_count = max(1, round(len(shuffled) * test_ratio)) if len(shuffled) > 1 else 0
        test.extend(shuffled[:test_count])
        dev.extend(shuffled[test_count:])
    return sorted(dev, key=lambda row: str(row.get("question_id"))), sorted(
        test, key=lambda row: str(row.get("question_id"))
    )


def dataset_hash(records: list[dict[str, Any]]) -> str:
    return canonical_json_hash(records)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def write_markdown_table(path: Path, title: str, rows: list[dict[str, Any]]) -> None:
    columns = list(rows[0]) if rows else []
    lines = [f"# {title}", ""]
    if columns:
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join("---" for _ in columns) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json_artifact(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)
