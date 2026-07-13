import json
from pathlib import Path

from paper_research.analysis.types import PaperAnalysis
from paper_research.evaluation.dataset import EvaluationItem

QUESTIONS = {
    "research_problem": "What research problem does this paper address?",
    "main_contributions": "What are the paper's main contributions?",
    "method_summary": "What method or technical approach does the paper propose?",
    "experiment_summary": "How are the experiments designed and evaluated?",
    "limitations": "What limitations or unresolved issues are reported?",
}


def main() -> None:
    root = Path("data/reports/parsing-audit")
    items: list[EvaluationItem] = []
    for analysis_path in sorted(root.glob("*/paper_analysis.json")):
        analysis = PaperAnalysis.model_validate_json(analysis_path.read_text(encoding="utf-8"))
        for field_name, question in QUESTIONS.items():
            field = getattr(analysis, field_name)
            items.append(
                EvaluationItem(
                    id=f"{analysis.paper_id}-{field_name}",
                    question=f"{question} Paper: {analysis.title or analysis.paper_id}",
                    question_type=field_name,
                    relevant_paper_ids=[analysis.paper_id],
                    relevant_block_ids=[item.block_id for item in field.evidence],
                    relevant_pages=sorted({item.page_start for item in field.evidence}),
                    expected_answer_points=(
                        [field.value] if isinstance(field.value, str) else field.value or []
                    ),
                    annotation_status="silver",
                    notes="Generated from evidence-bound analysis; requires human PDF review.",
                )
            )
    if len(items) != 50:
        raise RuntimeError(f"expected 50 evaluation items, got {len(items)}")
    output = Path("data/evaluation/research_qa_50.jsonl")
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for item in items:
            stream.write(json.dumps(item.model_dump(), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
