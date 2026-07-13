import json
from pathlib import Path

from paper_research.analysis.service import AnalysisService


def main() -> None:
    root = Path("data/reports/parsing-audit")
    rows = []
    field_names = [
        "research_problem",
        "main_contributions",
        "method_summary",
        "experiment_summary",
        "main_results",
        "limitations",
    ]
    for paper_dir in sorted(path.parent for path in root.glob("*/paper_blocks.jsonl")):
        analysis = AnalysisService().analyze_artifacts(paper_dir.name, paper_dir)
        populated = 0
        evidence_bound = 0
        for name in field_names:
            field = getattr(analysis, name)
            if field.value:
                populated += 1
                evidence_bound += bool(field.evidence)
        rows.append(
            {
                "paper": paper_dir.name,
                "populated": populated,
                "evidence_bound": evidence_bound,
                "total": len(field_names),
            }
        )
    total_populated = sum(row["populated"] for row in rows)
    total_bound = sum(row["evidence_bound"] for row in rows)
    report = [
        "# Paper Analysis Audit",
        "",
        f"- Papers: {len(rows)}",
        f"- Populated core fields: {total_populated}/{len(rows) * len(field_names)}",
        f"- Evidence-bound populated fields: {total_bound}/{total_populated}",
        "",
        "| Paper | Populated | Evidence bound | Total fields |",
        "|---|---:|---:|---:|",
    ]
    report.extend(
        f"| {row['paper']} | {row['populated']} | {row['evidence_bound']} | {row['total']} |"
        for row in rows
    )
    Path("data/reports/paper-analysis-audit.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    Path("data/reports/paper-analysis-audit.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
