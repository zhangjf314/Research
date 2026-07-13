# ruff: noqa: E501
"""Normalize the 50-item silver set into a human-review queue.

The output filename follows the release plan, but no item is promoted to gold until a
human reviewer changes ``review_status`` to ``approved`` after checking the cited PDF.
"""

import json
from collections import Counter
from pathlib import Path

SOURCE = Path("data/evaluation/research_qa_50.jsonl")
OUTPUT = Path("data/evaluation/gold-set-v1.jsonl")


def _difficulty(item: dict) -> str:
    claims = len(item.get("expected_answer_points") or [])
    blocks = len(item.get("relevant_block_ids") or [])
    if claims > 2 or blocks > 3:
        return "hard"
    if claims > 1 or blocks > 1:
        return "medium"
    return "easy"


def _category(item: dict, paper_index: int) -> str:
    kind = item["question_type"]
    if kind == "research_problem":
        return "research_background"
    if kind == "main_contributions":
        return "paper_contributions"
    if kind == "method_summary":
        return "algorithm_steps" if paper_index % 2 else "method"
    if kind == "experiment_summary":
        return "experiment_results" if paper_index % 2 else "experiment_setup"
    if kind == "limitations":
        return "unanswerable" if paper_index in {0, 5} else "limitations"
    raise ValueError(kind)


def _normalize(item: dict, index: int) -> dict:
    paper_index = index // 5
    category = _category(item, paper_index)
    answerable = category != "unanswerable"
    question = item["question"]
    gold_answer = "\n".join(item.get("expected_answer_points") or []) if answerable else None
    if not answerable:
        paper_id = item["relevant_paper_ids"][0]
        question = f"What exact total energy consumption is reported for all experiments? Paper: {paper_id}"
    return {
        "question_id": f"q{index + 1:03d}",
        "question": question,
        "scope": "single_paper",
        "category": category,
        "difficulty": _difficulty(item),
        "answerable": answerable,
        "gold_paper_ids": item["relevant_paper_ids"] if answerable else [],
        "gold_block_ids": item.get("relevant_block_ids", []) if answerable else [],
        "gold_pages": item.get("relevant_pages", []) if answerable else [],
        "gold_answer": gold_answer,
        "required_claims": item.get("expected_answer_points", []) if answerable else [],
        "citation_notes": (
            "Silver evidence candidate copied from the evidence-bound analyzer; verify every "
            "block and page against the source PDF."
            if answerable
            else "Confirm that the requested exact value is absent before approving refusal."
        ),
        "review_status": "pending",
        "reviewer": None,
        "reviewed_at": None,
        "review_notes": None,
        "dataset_version": "gold-set-v1-pending-review",
    }


def _comparison(first: dict, second: dict, question_id: str) -> dict:
    first_id = first["relevant_paper_ids"][0]
    second_id = second["relevant_paper_ids"][0]
    claims = [*(first.get("expected_answer_points") or []), *(second.get("expected_answer_points") or [])]
    return {
        "question_id": question_id,
        "question": f"Compare the main methods or contributions of papers {first_id} and {second_id}.",
        "scope": "multi_paper",
        "category": "multi_paper_comparison",
        "difficulty": "hard",
        "answerable": True,
        "gold_paper_ids": [first_id, second_id],
        "gold_block_ids": [
            *(first.get("relevant_block_ids") or []),
            *(second.get("relevant_block_ids") or []),
        ],
        "gold_pages": sorted(
            set((first.get("relevant_pages") or []) + (second.get("relevant_pages") or []))
        ),
        "gold_answer": "\n".join(claims),
        "required_claims": claims,
        "citation_notes": "Pending human comparison; each paper must have at least one valid citation.",
        "review_status": "pending",
        "reviewer": None,
        "reviewed_at": None,
        "review_notes": None,
        "dataset_version": "gold-set-v1-pending-review",
    }


def main() -> None:
    source = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines()]
    if len(source) != 50:
        raise RuntimeError(f"expected 50 source items, got {len(source)}")
    output = [_normalize(item, index) for index, item in enumerate(source)]
    # Replace two redundant pending items with cross-paper comparisons; nothing is auto-approved.
    output[-2] = _comparison(source[1], source[6], output[-2]["question_id"])
    output[-1] = _comparison(source[2], source[7], output[-1]["question_id"])
    required = {
        "research_background",
        "method",
        "algorithm_steps",
        "experiment_setup",
        "experiment_results",
        "paper_contributions",
        "limitations",
        "multi_paper_comparison",
        "unanswerable",
    }
    counts = Counter(item["category"] for item in output)
    missing = required - counts.keys()
    if missing:
        raise RuntimeError(f"missing categories: {sorted(missing)}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as stream:
        for item in output:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps({"items": len(output), "review_status": {"pending": 50}, "categories": counts}, default=dict))


if __name__ == "__main__":
    main()
