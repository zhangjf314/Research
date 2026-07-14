"""Stage 13 QA runner with a mandatory offline retrieval gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL = ROOT / "data/evaluation/evidence-retrieval-v1.json"
OUTPUT = ROOT / "data/evaluation/evidence-qa-v1.json"
CSV_OUTPUT = ROOT / "data/evaluation/evidence-qa-v1.csv"
REPORT = ROOT / "docs/evidence-qa-v1.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("dev", "full"), default="dev")
    parser.add_argument("--confirm-live", action="store_true")
    return parser.parse_args()


def _blocked_payload(retrieval: dict, phase: str) -> dict:
    return {
        "status": "BLOCKED_BY_OFFLINE_RETRIEVAL_GATE",
        "phase_requested": phase,
        "dev_run": False,
        "full_run": False,
        "llm_called": False,
        "deep_research_called": False,
        "rerank_enabled": False,
        "provider": "siliconflow",
        "model": "Qwen/Qwen3-8B",
        "baseline_prompt_version": "qa-production-v1",
        "experimental_prompt_version": "qa-evidence-centric-v1",
        "collection": retrieval["collection"],
        "blocking_gates": [
            name for name, passed in retrieval["dev_qa_candidate_gates"].items() if not passed
        ],
        "metrics": {
            "answerable_accuracy": None,
            "refusal_accuracy": None,
            "required_claim_coverage": None,
            "claim_omitted_due_to_missing_evidence": None,
            "unsupported_before_generation": None,
            "unsupported_after_generation": None,
            "exact_citation_precision": None,
            "citation_recall": None,
            "claim_citation_binding": None,
            "citation_allocated_to_correct_claim_rate": None,
            "invalid_citation_rate": None,
            "extra_claim_count": None,
            "json_schema_success": None,
            "retries": 0,
            "latency": None,
            "tokens": 0,
            "monetary_cost_usd": "0",
        },
        "reason": (
            "Dev QA is forbidden because the offline exact-block candidate gate did not pass."
        ),
    }


def main() -> int:
    args = parse_args()
    retrieval = json.loads(RETRIEVAL.read_text(encoding="utf-8"))
    if not retrieval.get("allow_dev_qa"):
        payload = _blocked_payload(retrieval, args.phase)
        OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        with CSV_OUTPUT.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["status", "phase_requested", "llm_called", "tokens", "cost"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "status": payload["status"],
                    "phase_requested": args.phase,
                    "llm_called": False,
                    "tokens": 0,
                    "cost": "0",
                }
            )
        REPORT.write_text(
            """# Evidence QA v1

Status: **BLOCKED_BY_OFFLINE_RETRIEVAL_GATE**.

The corrected, non-Oracle routed candidate reached exact block availability 0.645833, below the
Stage 13 Dev threshold of 0.65. Gold page availability and metadata gates passed, but every gate is
mandatory. No SiliconFlow request, Embedding request, Reranker, Full QA, or Deep Research run was
made. Tokens and monetary cost are zero.

Stage 11C baseline artifacts remain unchanged. Required-claim coverage, citation precision/recall,
unsupported-claim rate, latency and human support cannot be reported for Stage 13 QA because no
Stage 13 QA output exists.
""",
            encoding="utf-8",
        )
        print(
            json.dumps({"status": payload["status"], "blocking_gates": payload["blocking_gates"]})
        )
        return 2
    if args.phase == "full":
        raise SystemExit("Full QA requires a completed and accepted Dev artifact")
    if not args.confirm_live:
        raise SystemExit("live Dev QA requires --confirm-live after separate human authorization")
    raise SystemExit("live Dev QA is not authorized in this Stage 13 run")


if __name__ == "__main__":
    raise SystemExit(main())
