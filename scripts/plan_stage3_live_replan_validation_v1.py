"""Freeze Stage 3 live replan validation tasks without LLM or Agent runs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path("data/evaluation/research-agent")
DOC_ROOT = Path("docs/research-agent")
CORPUS = Path("data/evaluation/production-corpus-v1.json")
PLAN_JSON = ROOT / "stage3-live-replan-validation-plan-v1.json"
PLAN_MD = DOC_ROOT / "stage3-live-replan-validation-plan-v1.md"
EXPECTED_STAGE2_HASH = "995a144385180b2931ec2c6366f7f7306301a42d77ad7c85f4be9e6d9e5091d9"


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    DOC_ROOT.mkdir(parents=True, exist_ok=True)
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    papers = [
        {
            "paper_id": paper["paper_id"],
            "title": paper["title"],
            "corpus_role": paper["corpus_role"],
            "included_in_production": paper["included_in_production"],
            "topic": topic_for(paper["title"]),
            "section_availability": "unknown_from_manifest",
        }
        for paper in corpus["papers"]
        if paper.get("included_in_production") is True
        and paper.get("corpus_role") == "research_paper"
    ]
    if len(papers) != 33:
        raise SystemExit(f"expected 33 research papers, found {len(papers)}")
    tasks = build_tasks()
    validation_set_hash = stable_hash({"tasks": tasks})
    payload = {
        "schema_version": "stage3-live-replan-validation-plan-v1",
        "created_at": now(),
        "git_commit": git_head(),
        "agent_runs": 0,
        "provider_requests": 0,
        "corpus_manifest": str(CORPUS),
        "formal_research_paper_count": len(papers),
        "excluded_roles": ["ocr_fixture", "release_acceptance_text_fixture"],
        "stage2_rag_backend_hash": EXPECTED_STAGE2_HASH,
        "validation_task_count": len(tasks),
        "validation_set_frozen": True,
        "stage3_live_replan_validation_set_hash": validation_set_hash,
        "corpus_candidate_map": papers,
        "validation_tasks": [
            {**task, "task_hash": stable_hash(task)} for task in tasks
        ],
        "stage4_task_exclusions": [
            {
                "task_id": task["task_id"],
                "research_question": task["research_question"],
                "exclude_exact_question": True,
                "exclude_obvious_paraphrase": True,
                "exclude_same_observation_dependent_comparison": True,
            }
            for task in tasks
        ],
        "gold_leakage_guard": {
            "uses_gold_answers": False,
            "uses_gold_block_ids": False,
            "uses_gold_pages": False,
            "uses_required_claims": False,
            "uses_stage4_benchmark_tasks": False,
            "agent_input_fields": ["research_question"],
        },
    }
    write_json(PLAN_JSON, payload)
    write_doc(payload)
    print(
        json.dumps(
            {
                "validation_task_count": len(tasks),
                "validation_set_hash": validation_set_hash,
                "provider_requests": 0,
                "agent_runs": 0,
            },
            ensure_ascii=False,
        )
    )
    return 0


def build_tasks() -> list[dict[str, Any]]:
    return [
        {
            "task_id": "stage3-replan-v1-task-1-dataset-bridge",
            "task_pattern": "OBSERVATION_DERIVED_DATASET_BRIDGE",
            "paper_ids_named_in_question": ["2403.10081", "2510.22344"],
            "research_question": (
                "Compare DRAGIN: Dynamic Retrieval Augmented Generation based on "
                "the Information Needs of Large Language Models with FAIR-RAG: "
                "Faithful Adaptive Iterative Refinement for Retrieval-Augmented "
                "Generation. Determine which evaluation datasets or benchmarks "
                "DRAGIN actually uses, then assess whether FAIR-RAG reports directly "
                "comparable results on any of those datasets; if direct dataset overlap "
                "is not supported by evidence, identify the closest evidence-supported "
                "comparison dimension."
            ),
            "authoring_notes": (
                "Dependency-shaped task: the second comparison target depends on "
                "datasets discovered from DRAGIN evidence. Notes are not sent to Agent."
            ),
        },
        {
            "task_id": "stage3-replan-v1-task-2-limitation-bridge",
            "task_pattern": "OBSERVATION_DERIVED_LIMITATION_BRIDGE",
            "paper_ids_named_in_question": ["2507.06956", "2602.07525"],
            "research_question": (
                "Compare Investigating the Robustness of Retrieval-Augmented Generation "
                "at the Query Level with IGMiRAG: Intuition-Guided Retrieval-Augmented "
                "Generation with Adaptive Mining of In-Depth Memory. Identify a major "
                "limitation or failure mode that the robustness paper explicitly reports, "
                "then determine whether IGMiRAG provides a directly targeted mechanism "
                "or experimental evidence addressing that limitation; if direct evidence "
                "is absent, state the closest evidence-supported mitigation and its boundary."
            ),
            "authoring_notes": (
                "Dependency-shaped task: the second search target should depend on the "
                "limitation/failure mode observed in the first paper."
            ),
        },
        {
            "task_id": "stage3-replan-v1-task-3-metric-comparability",
            "task_pattern": "OBSERVATION_DERIVED_METRIC_COMPARABILITY",
            "paper_ids_named_in_question": ["2309.15217", "2409.03759"],
            "research_question": (
                "Assess whether Ragas: Automated Evaluation of Retrieval Augmented "
                "Generation and VERA: Validation and Evaluation of Retrieval-Augmented "
                "Systems can be compared on a shared quality or efficiency metric. "
                "First identify the metrics each paper actually reports, then restrict "
                "the comparison to metrics that both papers report with evidence; clearly "
                "separate any initially expected comparison dimension that the evidence "
                "does not support."
            ),
            "authoring_notes": (
                "Dependency-shaped task: the valid comparison dimension should be chosen "
                "after observing metrics from both papers."
            ),
        },
    ]


def topic_for(title: str) -> str:
    lower = title.lower()
    if "retrieval" in lower or "rag" in lower:
        return "retrieval_augmented_generation"
    if "bert" in lower or "transformer" in lower:
        return "transformer_language_model"
    if "scaling" in lower:
        return "language_model_scaling"
    return "other"


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def write_doc(payload: dict[str, Any]) -> None:
    lines = [
        "# Stage 3 Live Replan Validation Plan v1",
        "",
        f"- validation_task_count: `{payload['validation_task_count']}`",
        f"- validation_set_frozen: `{payload['validation_set_frozen']}`",
        "- provider_requests: `0`",
        "- agent_runs: `0`",
        f"- validation_set_hash: `{payload['stage3_live_replan_validation_set_hash']}`",
        "",
        (
            "These tasks are development branch-coverage validation tasks, "
            "not Stage 4 benchmark tasks."
        ),
    ]
    for task in payload["validation_tasks"]:
        lines.extend(
            [
                "",
                f"## {task['task_id']}",
                "",
                f"- pattern: `{task['task_pattern']}`",
                (
                    "- paper_ids_named_in_question: "
                    f"`{', '.join(task['paper_ids_named_in_question'])}`"
                ),
                f"- task_hash: `{task['task_hash']}`",
                "",
                task["research_question"],
            ]
        )
    PLAN_MD.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def now() -> str:
    return datetime.now(UTC).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
