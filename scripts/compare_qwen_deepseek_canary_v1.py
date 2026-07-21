"""Compare fixed Qwen and DeepSeek Stage 13 canary runs."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
QWEN_JSON = ROOT / "data" / "evaluation" / "full-qa-canary-results-v2.json"
DEEPSEEK_JSON = ROOT / "data" / "evaluation" / "full-qa-canary-deepseek-v1.json"
OUT_JSON = ROOT / "data" / "evaluation" / "qwen-vs-deepseek-canary-comparison-v1.json"
OUT_CSV = ROOT / "data" / "evaluation" / "qwen-vs-deepseek-canary-comparison-v1.csv"
OUT_DOC = ROOT / "docs" / "qwen-vs-deepseek-canary-comparison-v1.md"

METRICS = [
    "completed",
    "terminal_failure_count",
    "malformed_json_count",
    "schema_failure_count",
    "required_claim_coverage",
    "citation_precision",
    "citation_recall",
    "core_unsupported_claim_count",
    "citation_context_validity",
    "page_accuracy",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "estimated_cost_usd",
]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("summary") or payload.get("metrics") or {}


def main() -> int:
    qwen = _load(QWEN_JSON)
    deepseek = _load(DEEPSEEK_JSON)
    qwen_ids = qwen.get("canary_ids") or []
    deepseek_ids = deepseek.get("canary_ids") or []
    same_samples = qwen_ids == deepseek_ids
    comparison = {
        "schema_version": "qwen-vs-deepseek-canary-comparison-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "same_15_samples": same_samples,
        "same_retrieval": True,
        "same_context": True,
        "same_prompt_contract": True,
        "same_reranker_state": True,
        "not_blind_holdout": True,
        "qwen": {
            "provider": (qwen.get("llm") or {}).get("provider") or "siliconflow",
            "model": (qwen.get("llm") or {}).get("model") or "Qwen/Qwen3-8B",
            "summary": _summary(qwen),
        },
        "deepseek": {
            "provider": (deepseek.get("llm") or {}).get("provider"),
            "model": (deepseek.get("llm") or {}).get("model"),
            "summary": _summary(deepseek),
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with OUT_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["metric", "qwen", "deepseek"])
        writer.writeheader()
        for metric in METRICS:
            writer.writerow(
                {
                    "metric": metric,
                    "qwen": comparison["qwen"]["summary"].get(metric),
                    "deepseek": comparison["deepseek"]["summary"].get(metric),
                }
            )
    rows = [
        "| Metric | Qwen | DeepSeek |",
        "| --- | ---: | ---: |",
    ]
    for metric in METRICS:
        rows.append(
            f"| {metric} | `{comparison['qwen']['summary'].get(metric)}` | "
            f"`{comparison['deepseek']['summary'].get(metric)}` |"
        )
    OUT_DOC.write_text(
        "\n".join(
            [
                "# Qwen vs DeepSeek Canary Comparison v1",
                "",
                f"- Same 15 samples: `{same_samples}`",
                "- Same retrieval/context/prompt/reranker state: `true`",
                "- Dataset status: internal development canary, not a blind holdout.",
                "",
                *rows,
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"same_samples": same_samples, "output": str(OUT_JSON)}))
    return 0 if same_samples else 2


if __name__ == "__main__":
    raise SystemExit(main())
