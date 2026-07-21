"""Backfill terminal sentinels for failed Dev v3.2 runs without changing outcomes."""

from __future__ import annotations

import json

try:
    from scripts.evidence_qa_dev_v3_2_lib import RUN_ROOT
except ModuleNotFoundError:
    from evidence_qa_dev_v3_2_lib import RUN_ROOT  # type: ignore[no-redef]

TERMINAL_FILES = (
    "parsed-v3-2-output.json",
    "citation-selection-trace.json",
    "obligation-analysis.json",
    "numeric-validation.json",
    "comparison-validation.json",
    "claim-fallback-trace.json",
)


def main() -> None:
    updated = []
    for result_path in sorted(RUN_ROOT.glob("live-dev-v3-2-*/final-result.json")):
        run_dir = result_path.parent
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["status"] == "completed":
            continue
        sentinel = {
            "status": "not_run_due_to_raw_or_business_validation_failure",
            "run_id": result["run_id"],
            "question_id": result["question_id"],
            "failure_type": result["failure_type"],
            "failure_reason": result["failure_reason"],
            "raw_response_modified": False,
            "outcome_modified": False,
        }
        for name in TERMINAL_FILES:
            path = run_dir / name
            if not path.exists():
                path.write_text(
                    json.dumps(sentinel, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        updated.append(result["run_id"])
    print(json.dumps({"terminal_sentinels_completed": updated}, ensure_ascii=False))


if __name__ == "__main__":
    main()
