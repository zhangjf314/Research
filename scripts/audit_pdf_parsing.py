import argparse
import csv
import json
import statistics
import time
from pathlib import Path

from paper_research.ingestion.artifacts import write_parse_artifacts
from paper_research.parsing.pymupdf_parser import PyMuPDFParser


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit PDF parsing across a paper corpus.")
    parser.add_argument("--input", type=Path, default=Path("data/raw/audit"))
    parser.add_argument("--output", type=Path, default=Path("data/reports/parsing-audit"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int | float]] = []
    paper_parser = PyMuPDFParser()
    for pdf_path in sorted(args.input.glob("*.pdf")):
        started = time.perf_counter()
        try:
            parsed = paper_parser.parse(pdf_path)
            output_dir = args.output / pdf_path.stem
            write_parse_artifacts(parsed, output_dir)
            headings = sum(block.block_type == "heading" for block in parsed.blocks)
            located = sum(
                block.page_start > 0 and block.bbox.x1 > block.bbox.x0 for block in parsed.blocks
            )
            rows.append(
                {
                    "paper": pdf_path.name,
                    "status": "PASS" if parsed.blocks and located == len(parsed.blocks) else "WARN",
                    "pages": parsed.metadata.page_count,
                    "blocks": len(parsed.blocks),
                    "headings": headings,
                    "located_blocks": located,
                    "warnings": len(parsed.warnings),
                    "seconds": round(time.perf_counter() - started, 3),
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "paper": pdf_path.name,
                    "status": "FAIL",
                    "pages": 0,
                    "blocks": 0,
                    "headings": 0,
                    "located_blocks": 0,
                    "warnings": 0,
                    "seconds": round(time.perf_counter() - started, 3),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    write_reports(rows, args.output)


def write_reports(rows: list[dict[str, str | int | float]], output_dir: Path) -> None:
    csv_path = output_dir / "audit_results.csv"
    fieldnames = [
        "paper", "status", "pages", "blocks", "headings", "located_blocks",
        "warnings", "seconds", "error",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    passed = sum(row["status"] == "PASS" for row in rows)
    durations = [float(row["seconds"]) for row in rows]
    report = [
        "# Real-paper PDF Parsing Audit",
        "",
        f"- Papers: {len(rows)}",
        f"- Passed: {passed}",
        f"- Warned: {sum(row['status'] == 'WARN' for row in rows)}",
        f"- Failed: {sum(row['status'] == 'FAIL' for row in rows)}",
        f"- Median parse time: {statistics.median(durations) if durations else 0:.3f}s",
        "",
        "| Paper | Status | Pages | Blocks | Headings | Warnings | Seconds |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    report.extend(
        f"| {row['paper']} | {row['status']} | {row['pages']} | {row['blocks']} | "
        f"{row['headings']} | {row['warnings']} | {row['seconds']} |"
        for row in rows
    )
    (output_dir / "audit_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (output_dir / "audit_results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
