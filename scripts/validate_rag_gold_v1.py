from __future__ import annotations

# ruff: noqa: E501
import argparse
from datetime import UTC, datetime
from pathlib import Path

from paper_research.evaluation.rag_benchmark import read_jsonl
from paper_research.evaluation.rag_gold import (
    corpus_coverage,
    load_evidence_index,
    validate_gold_records,
    write_json_artifact,
)

ROOT = Path("data/evaluation/rag-benchmark")
DOCS = Path("docs/rag-benchmark")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/evaluation/gold-set-v1.jsonl"))
    parser.add_argument("--output-json", type=Path, default=ROOT / "gold-validation-v1.json")
    parser.add_argument("--strict-structured-claims", action="store_true")
    parser.add_argument("--fail-on-warning", action="store_true")
    parser.add_argument("--no-fail", action="store_true", help="Write audit artifacts even when validation fails.")
    args = parser.parse_args()

    records = read_jsonl(args.input)
    report = validate_gold_records(
        records,
        evidence_index=load_evidence_index(),
        strict_structured_claims=args.strict_structured_claims,
    )
    report["schema_version"] = "rag-gold-validation-v1"
    report["validated_at"] = datetime.now(UTC).isoformat()
    report["input"] = str(args.input)
    report["strict_structured_claims"] = args.strict_structured_claims
    report["corpus_coverage"] = corpus_coverage(records)
    write_json_artifact(args.output_json, report)

    lines = [
        "# RAG Gold validation v1",
        "",
        f"- input: `{args.input}`",
        f"- valid: `{report['valid']}`",
        f"- records: {report['record_count']}",
        f"- approved: {report['approved_count']}",
        f"- answerable: {report['answerable_count']}",
        f"- unanswerable: {report['unanswerable_count']}",
        f"- errors: {report['error_count']}",
        f"- warnings: {report['warning_count']}",
        f"- duplicate questions: {report['duplicate_question_count']}",
        f"- near-duplicate questions: {report['near_duplicate_question_count']}",
        "",
        "Strict structured claim mode is required for the final frozen `rag-gold-v1` dataset. Existing legacy Gold may be audited without this flag while the derived benchmark copy is normalized.",
    ]
    if report["errors"]:
        lines.extend(["", "## Errors", ""])
        for error in report["errors"][:50]:
            lines.append(f"- `{error['type']}`: {error}")
    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        for warning in report["warnings"][:50]:
            lines.append(f"- `{warning['type']}`: {warning}")
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "gold-validation-v1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if args.no_fail:
        return 0
    if report["error_count"] or (args.fail_on_warning and report["warning_count"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
