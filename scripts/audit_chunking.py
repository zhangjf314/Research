import argparse
import json
import statistics
from pathlib import Path

from paper_research.chunking.fixed_chunker import FixedTokenChunker
from paper_research.chunking.structural_chunker import StructuralChunker
from paper_research.parsing.types import PaperBlock


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare structural and fixed chunking.")
    parser.add_argument("--input", type=Path, default=Path("data/reports/parsing-audit"))
    parser.add_argument("--output", type=Path, default=Path("data/reports/chunking-audit.md"))
    args = parser.parse_args()
    rows: list[dict[str, int | str | float]] = []
    for blocks_path in sorted(args.input.glob("*/paper_blocks.jsonl")):
        blocks = [
            PaperBlock.model_validate(json.loads(line))
            for line in blocks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        paper_id = blocks_path.parent.name
        structural = StructuralChunker().chunk(paper_id, blocks)
        fixed = FixedTokenChunker().chunk(paper_id, blocks)
        rows.append(
            {
                "paper": paper_id,
                "structural": len(structural),
                "fixed": len(fixed),
                "with_section_pct": round(
                    100 * sum(bool(chunk.section_path) for chunk in structural) / len(structural), 1
                )
                if structural
                else 0,
                "with_page_pct": round(
                    100 * sum(chunk.page_start > 0 for chunk in structural) / len(structural), 1
                )
                if structural
                else 0,
            }
        )
    median_section = statistics.median(float(row["with_section_pct"]) for row in rows)
    report = [
        "# Chunking Audit",
        "",
        f"- Papers: {len(rows)}",
        f"- Structural chunks: {sum(int(row['structural']) for row in rows)}",
        f"- Fixed chunks: {sum(int(row['fixed']) for row in rows)}",
        f"- Median section coverage: {median_section:.1f}%",
        "",
        "| Paper | Structural | Fixed | Section metadata | Page metadata |",
        "|---|---:|---:|---:|---:|",
    ]
    report.extend(
        f"| {row['paper']} | {row['structural']} | {row['fixed']} | "
        f"{row['with_section_pct']}% | {row['with_page_pct']}% |"
        for row in rows
    )
    args.output.write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
