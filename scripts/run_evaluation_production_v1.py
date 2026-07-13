"""Production evaluation gate: formal metrics require human-approved records and providers."""

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

from paper_research.config import Settings

DATASET = Path("data/evaluation/gold-set-v1.jsonl")
JSON_OUTPUT = Path("data/evaluation/results-production-v1.json")
CSV_OUTPUT = Path("data/evaluation/results-production-v1.csv")
REPORT_OUTPUT = Path("docs/evaluation-report-production-v1.md")


def main() -> int:
    items = [
        json.loads(line)
        for line in DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    approved = [item for item in items if item.get("review_status") == "approved"]
    settings = Settings()
    blockers = []
    if len(approved) != 50:
        blockers.append(f"human gold approval incomplete: {len(approved)}/50 approved")
    if settings.app_profile != "production":
        blockers.append(f"APP_PROFILE is {settings.app_profile}, expected production")
    blockers.extend(settings.production_configuration_issues)

    payload = {
        "status": "BLOCKED" if blockers else "READY",
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": str(DATASET),
        "dataset_version": settings.dataset_version,
        "total_items": len(items),
        "approved_items": len(approved),
        "pending_items_excluded": len(items) - len(approved),
        "model_configuration": settings.provider_metadata,
        "random_seed": 42,
        "blockers": blockers,
        "results": [],
        "token_usage": None,
        "cost": None,
    }
    JSON_OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with CSV_OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["status", "approved_items", "blocker"])
        writer.writeheader()
        for blocker in blockers or [""]:
            writer.writerow(
                {"status": payload["status"], "approved_items": len(approved), "blocker": blocker}
            )
    lines = [
        "# Production Evaluation v1",
        "",
        f"- Status: **{payload['status']}**",
        f"- Human-approved items: **{len(approved)}/50**",
        f"- Pending items excluded: **{len(items) - len(approved)}**",
        f"- Profile: `{settings.app_profile}`",
        "",
        "## Blocking conditions",
        "",
        *[f"- {blocker}" for blocker in blockers],
        "",
        "No production quality, Token, cost, or model latency values are emitted until all "
        "records are human-approved and production providers are configured.",
        "",
    ]
    REPORT_OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 2 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
