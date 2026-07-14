"""Build explicit top-level Stage 11D summaries from isolated run directories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from paper_research.agents.smoke_artifacts import DEFAULT_RUN_ROOT, load_runs, select_runs

DEFAULT_JSON = Path("data/evaluation/deep-research-smoke-v1.json")
DEFAULT_CSV = Path("data/evaluation/deep-research-smoke-v1.csv")
DEFAULT_TRACE = Path("data/evaluation/deep-research-smoke-traces-v1.jsonl")
DEFAULT_REPORT = Path("docs/deep-research-smoke-v1.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection-policy",
        choices=("latest-successful", "latest-attempt", "explicit-run-id"),
        default="latest-successful",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    return parser.parse_args()


def summarize(root: Path, policy: str, run_id: str | None = None) -> dict:
    runs = load_runs(root)
    selected = select_runs(runs, policy, run_id)
    selected_ids = {run["metadata"]["run_id"] for run in selected}
    attempts = [
        {
            "run_id": run["metadata"]["run_id"],
            "question_id": run["metadata"]["question_id"],
            "status": run["result"]["graph_status"],
            "attempt_number": run["metadata"]["attempt_number"],
            "parent_run_id": run["metadata"]["parent_run_id"],
            "ended_at": run["metadata"]["ended_at"],
            "selected": run["metadata"]["run_id"] in selected_ids,
            "run_directory": str(run["run_dir"]),
        }
        for run in sorted(runs, key=lambda item: item["metadata"]["ended_at"])
    ]
    return {
        "status": "ENGINEERING_ONLY_EXPLICIT_SUMMARY",
        "quality_gate": "NOT_EVALUATED",
        "selection_policy": policy,
        "explicit_run_id": run_id,
        "selected_run_ids": sorted(selected_ids),
        "results": [run["result"] for run in selected],
        "attempts": attempts,
        "failed_attempts": [
            row for row in attempts if row["status"] not in {"completed", "refused"}
        ],
    }


def write_summary(summary: dict, root: Path) -> None:
    DEFAULT_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    results = summary["results"]
    scalar_fields = sorted(
        {
            key
            for row in results
            for key, value in row.items()
            if not isinstance(value, (list, dict))
        }
    )
    with DEFAULT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_fields)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key) for key in scalar_fields})
    selected_ids = set(summary["selected_run_ids"])
    trace_lines = []
    for run in load_runs(root):
        for line in (run["run_dir"] / "trace.jsonl").read_text(encoding="utf-8").splitlines():
            if line:
                event = json.loads(line)
                event["selected"] = run["metadata"]["run_id"] in selected_ids
                trace_lines.append(json.dumps(event, ensure_ascii=False))
    DEFAULT_TRACE.write_text(
        "\n".join(trace_lines) + ("\n" if trace_lines else ""), encoding="utf-8"
    )
    lines = [
        "# Stage 11D Deep Research Smoke Summary",
        "",
        "> Engineering-only summary; not QA quality, production, or v1.0 evidence.",
        "",
        f"- Selection policy: `{summary['selection_policy']}`",
        f"- Selected runs: `{', '.join(summary['selected_run_ids'])}`",
        f"- Total attempts: {len(summary['attempts'])}",
        f"- Failed attempts retained: {len(summary['failed_attempts'])}",
        "",
        "## All attempts",
        "",
    ]
    lines.extend(
        f"- `{row['run_id']}` / `{row['question_id']}`: `{row['status']}`; "
        f"attempt={row['attempt_number']}; selected={str(row['selected']).lower()}"
        for row in summary["attempts"]
    )
    DEFAULT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.selection_policy == "explicit-run-id" and not args.run_id:
        raise ValueError("explicit-run-id requires --run-id")
    if args.selection_policy != "explicit-run-id" and args.run_id:
        raise ValueError("--run-id is only valid with explicit-run-id")
    summary = summarize(args.run_root, args.selection_policy, args.run_id)
    write_summary(summary, args.run_root)
    print(
        json.dumps(
            {
                "selection_policy": args.selection_policy,
                "selected": summary["selected_run_ids"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
