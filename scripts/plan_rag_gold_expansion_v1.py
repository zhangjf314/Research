from __future__ import annotations

# ruff: noqa: E501
import argparse
from datetime import UTC, datetime
from pathlib import Path

from paper_research.evaluation.rag_benchmark import read_jsonl
from paper_research.evaluation.rag_gold import corpus_coverage, expansion_plan, write_json_artifact

ROOT = Path("data/evaluation/rag-benchmark")
DOCS = Path("docs/rag-benchmark")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/evaluation/gold-set-v1.jsonl"))
    args = parser.parse_args()
    records = read_jsonl(args.input)
    plan = expansion_plan(records)
    plan["created_at"] = datetime.now(UTC).isoformat()
    plan["source_gold"] = str(args.input)
    plan["corpus_coverage"] = corpus_coverage(records)
    output_json = ROOT / "gold-expansion-plan-v1.json"
    write_json_artifact(output_json, plan)

    lines = [
        "# RAG Gold expansion plan v1",
        "",
        "This plan expands the current human-reviewed internal Gold set toward approximately 150 questions before any Stage 2 RAG optimization.",
        "",
        f"- current_total: {plan['current_total']}",
        f"- target_total: {plan['target_total']}",
        f"- questions_to_add: {plan['questions_to_add']}",
        "- candidate review_status default: `pending`",
        "- approved requires human review: `true`",
        "",
        "## Category deficits",
        "",
        "| category | current | target | deficit |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in plan["category_plan"]:
        lines.append(f"| {row['category']} | {row['current']} | {row['target']} | {row['deficit']} |")
    lines.extend(
        [
            "",
            "## Difficulty guidance",
            "",
            "- easy: 25-30%",
            "- medium: 40-50%",
            "- hard: 25-30%",
            "",
            "New questions must not all be simple factual questions. Medium and hard questions should require multiple evidence blocks, multiple sections, cross-paper comparison, conflicting evidence, methods/results comparison, or limitations synthesis.",
            "",
            "## Corpus coverage",
            "",
            f"- corpus_paper_count: {plan['corpus_coverage']['corpus_paper_count']}",
            f"- papers_covered_by_current_gold: {plan['corpus_coverage']['papers_covered']}",
            "",
            "## Review policy",
            "",
            "LLM or rule-assisted authoring may only create draft candidates. Final Gold questions, required claims, answers, pages, and block evidence must be human reviewed before `review_status=approved`.",
        ]
    )
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "gold-expansion-plan-v1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
