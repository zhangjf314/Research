import json
from pathlib import Path

from paper_research.parsing.types import ParsedPaper


def write_parse_artifacts(parsed: ParsedPaper, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "paper_metadata.json"
    blocks_path = output_dir / "paper_blocks.jsonl"
    report_path = output_dir / "parse_report.md"
    manifest_path = output_dir / "parse_manifest.json"

    metadata_path.write_text(
        json.dumps(parsed.metadata.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest_path.write_text(
        json.dumps(
            {
                "parser_name": parsed.parser_name or parsed.parser,
                "is_ocr": parsed.is_ocr,
                "ocr_confidence": parsed.ocr_confidence,
                "source_pages": sorted(
                    {block.source_page or block.page_start for block in parsed.blocks}
                ),
                "parse_warnings": [warning.model_dump() for warning in parsed.warnings],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with blocks_path.open("w", encoding="utf-8", newline="\n") as stream:
        for block in parsed.blocks:
            stream.write(json.dumps(block.model_dump(), ensure_ascii=False) + "\n")

    warning_lines = [
        f"- `{warning.code}` page {warning.page or 'n/a'}: {warning.message}"
        for warning in parsed.warnings
    ] or ["- None"]
    report = "\n".join(
        [
            "# PDF Parse Report",
            "",
            f"- Parser: `{parsed.parser}`",
            f"- Parser name: `{parsed.parser_name or parsed.parser}`",
            f"- OCR: `{parsed.is_ocr}`",
            f"- OCR confidence: `{parsed.ocr_confidence}`",
            f"- Pages: {parsed.metadata.page_count}",
            f"- Blocks: {len(parsed.blocks)}",
            f"- Warnings: {len(parsed.warnings)}",
            "",
            "## Warnings",
            "",
            *warning_lines,
            "",
        ]
    )
    report_path.write_text(report, encoding="utf-8")
    return {
        "metadata": metadata_path,
        "blocks": blocks_path,
        "manifest": manifest_path,
        "report": report_path,
    }
