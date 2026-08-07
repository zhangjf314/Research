from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime

from paper_research.evaluation.rag_stage2a import (
    OPT_DOCS,
    OPT_ROOT,
    generation_metric_audit,
    load_baseline_generation_rows,
    write_json,
)


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def main() -> int:
    rows = load_baseline_generation_rows()
    audit = generation_metric_audit(rows)
    payload = {
        "schema_version": "generation-metric-audit-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "benchmark_harness_commit": git_head(),
        **audit,
    }
    write_json(OPT_ROOT / "generation-metric-audit-v1.json", payload)
    precision = payload["citation_precision"]
    recall = payload["citation_recall"]
    lines = [
        "# Generation metric audit v1",
        "",
        f"- status: `{payload['status']}`",
        f"- answerable_questions: `{payload['answerable_questions']}`",
        f"- citation_precision: `{precision['value']}` "
        f"({precision['numerator']}/{precision['denominator']})",
        f"- citation_recall: `{recall['value']}` "
        f"({recall['numerator']}/{recall['denominator']})",
        "",
        "## Representation",
        "",
    ]
    for key, value in payload["representation"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Explanation", "", payload["explanation"], "", "## Sample buckets", ""])
    for key, value in payload["sample_bucket_counts"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Sampled examples", ""])
    for item in payload["sampled_examples"]:
        lines.append(
            f"- `{item['question_id']}`: {item['bucket']}, "
            f"correct={item['correct_citation_count']}/{item['citation_count']}"
        )
    OPT_DOCS.mkdir(parents=True, exist_ok=True)
    (OPT_DOCS / "generation-metric-audit-v1.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": payload["status"], "precision": precision, "recall": recall}))
    return 0 if payload["status"] == "METRICS_VALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
