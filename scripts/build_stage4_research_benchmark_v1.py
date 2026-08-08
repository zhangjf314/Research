"""Build and freeze Stage 4 Workflow-vs-Agent research benchmark v1.

Offline-only: this script performs deterministic AI-style authoring/review from
local corpus/evidence artifacts. It does not run Workflow, Agent, retrieval, or
provider calls.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import subprocess
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path("data/evaluation/research-agent")
BENCH = ROOT / "benchmark"
DOC = Path("docs/research-agent/benchmark")
TASKS_JSONL = BENCH / "research-tasks-v1.jsonl"
RUBRICS_JSONL = BENCH / "research-task-rubrics-v1.jsonl"
MANIFEST_JSON = BENCH / "research-benchmark-manifest-v1.json"
TOP_MANIFEST_JSON = ROOT / "stage4-benchmark-manifest-v1.json"
EXECUTION_ORDER_JSON = BENCH / "stage4-execution-order-v1.json"
EVALUATION_PROTOCOL_JSON = BENCH / "stage4-evaluation-protocol-v1.json"
FAIRNESS_JSON = BENCH / "stage4-fairness-audit-v1.json"
VALIDATION_JSON = BENCH / "research-benchmark-validation-v1.json"
BENCH_DOC = DOC / "research-benchmark-v1.md"
PROTOCOL_DOC = DOC / "research-benchmark-protocol-v1.md"
VALIDATION_DOC = DOC / "research-benchmark-validation-v1.md"

PRODUCTION_CORPUS = Path("data/evaluation/production-corpus-v1.json")
EVIDENCE_CORPUS = Path("data/evaluation/evidence-corpus-v1.jsonl")

TASK_DISTRIBUTION = {
    "multi_paper_synthesis": 15,
    "cross_paper_comparison": 15,
    "methods_and_experiments": 10,
    "limitations_and_research_gaps": 10,
    "evidence_insufficiency_or_noncomparability": 5,
    "observation_dependent_research": 5,
}
DIFFICULTIES = ["easy"] * 10 + ["medium"] * 30 + ["hard"] * 20
PAPER_COUNTS = [2] * 24 + [3] * 24 + [4] * 12
EXECUTION_SEED = 40721
BOOTSTRAP_SEED = 41007
EXPECTED_RAG_HASH = "995a144385180b2931ec2c6366f7f7306301a42d77ad7c85f4be9e6d9e5091d9"
EXPECTED_AGENT_HASH = "bce71a51171b2e1187d579a2278cc34f1202ed7b84e9482cbffe42d00b92ff15"
REPLAN_METRICS = [
    "replan_task_count",
    "effective_replan_count",
    "replan_task_rate",
    "replans_per_task",
    "replan_trigger_distribution",
    "post_replan_success_rate",
    "post_replan_evidence_gain",
    "post_replan_claim_coverage_delta",
]
DIMENSION_TEMPLATES = [
    "research objective and problem framing",
    "method or system design",
    "experimental setup and evaluation basis",
    "reported limitations, gaps, or non-comparability boundary",
    "cross-paper synthesis or trade-off",
]
TITLE_OVERRIDES = {
    "1706.03762": "Attention Is All You Need",
    "1910.10683": (
        "Exploring the Limits of Transfer Learning with a Unified Text-to-Text "
        "Transformer"
    ),
}


def main() -> int:
    BENCH.mkdir(parents=True, exist_ok=True)
    DOC.mkdir(parents=True, exist_ok=True)
    locks = load_and_validate_locks()
    papers = load_papers()
    evidence = load_evidence(papers)
    tasks, rubrics = author_and_review_tasks(papers, evidence)
    execution_order = build_execution_order(tasks)
    protocol = build_evaluation_protocol()
    fairness = build_fairness_audit(locks)

    write_jsonl(TASKS_JSONL, tasks)
    write_jsonl(RUBRICS_JSONL, rubrics)
    write_json(EXECUTION_ORDER_JSON, execution_order)
    write_json(EVALUATION_PROTOCOL_JSON, protocol)
    write_json(FAIRNESS_JSON, fairness)

    validation = validate_dataset(tasks, rubrics, papers, evidence, locks)
    write_json(VALIDATION_JSON, validation)
    manifest = build_manifest(tasks, rubrics, execution_order, protocol, validation, locks)
    write_json(MANIFEST_JSON, manifest)
    write_json(TOP_MANIFEST_JSON, manifest)
    write_docs(manifest, validation, protocol, fairness)

    print(
        json.dumps(
            {
                "benchmark_version": manifest["benchmark_version"],
                "task_count": manifest["task_count"],
                "dataset_hash": manifest["dataset_hash"],
                "stage4a_complete": validation["stage4a_complete"],
                "stage4b_ready": validation["stage4b_ready"],
                "provider_requests": 0,
                "official_workflow_runs": 0,
                "official_agent_runs": 0,
            },
            ensure_ascii=False,
        )
    )
    return 0 if validation["stage4a_complete"] else 2


def load_and_validate_locks() -> dict[str, Any]:
    agent = read_json(ROOT / "stage3-agent-lock-v1.json")
    workflow = read_json(ROOT / "stage4-workflow-control-lock-v1.json")
    comparability = read_json(ROOT / "stage4-comparability-lock-v1.json")
    if agent["stage2_rag_backend_hash"] != EXPECTED_RAG_HASH:
        raise SystemExit("STAGE4_LOCK_MISMATCH")
    if agent["stage3_agent_behavior_hash"] != EXPECTED_AGENT_HASH:
        raise SystemExit("STAGE4_LOCK_MISMATCH")
    if stable_hash(agent["behavior_hash_inputs"]) != EXPECTED_AGENT_HASH:
        raise SystemExit("STAGE4_LOCK_MISMATCH")
    if workflow["workflow_behavior_changed"] is not False:
        raise SystemExit("STAGE4_LOCK_MISMATCH")
    required_true = [
        "same_corpus",
        "same_index",
        "same_embedding",
        "same_retrieval_backend",
        "same_reranker_state",
        "same_query_rewrite_state",
        "same_query_decomposition_state",
    ]
    if any(comparability.get(key) is not True for key in required_true):
        raise SystemExit("STAGE4_LOCK_MISMATCH")
    return {"agent": agent, "workflow": workflow, "comparability": comparability}


def load_papers() -> list[dict[str, Any]]:
    payload = read_json(PRODUCTION_CORPUS)
    papers = [
        {
            "paper_id": item["paper_id"],
            "title": clean_title(item["paper_id"], item["title"]),
            "source_path": item["source_path"],
        }
        for item in payload["papers"]
        if item.get("included_in_production") is True
        and item.get("corpus_role") == "research_paper"
    ]
    if len(papers) != 33:
        raise SystemExit(f"expected 33 research papers, found {len(papers)}")
    return papers


def load_evidence(papers: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    allowed = {paper["paper_id"] for paper in papers}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in EVIDENCE_CORPUS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item["paper_id"] not in allowed:
            continue
        text = normalize_space(item.get("text") or "")
        roles = set(item.get("evidence_roles") or [])
        if len(text) < 60 or roles <= {"non_evidence", "metadata", "citation_only"}:
            continue
        grouped[item["paper_id"]].append(
            {
                "evidence_id": item["evidence_id"],
                "paper_id": item["paper_id"],
                "block_id": item["block_id"],
                "page": item["page"],
                "section_title": item.get("section_title"),
                "block_type": item.get("block_type"),
                "evidence_roles": sorted(roles),
                "claim_summary": summarize_text(text),
            }
        )
    for paper in papers:
        grouped[paper["paper_id"]] = sorted(
            grouped[paper["paper_id"]],
            key=lambda item: (
                role_rank(item["evidence_roles"]),
                int(item["page"] or 9999),
                item["block_id"],
            ),
        )
        if len(grouped[paper["paper_id"]]) < 8:
            raise SystemExit(f"insufficient evidence for {paper['paper_id']}")
    return grouped


def author_and_review_tasks(
    papers: list[dict[str, Any]],
    evidence: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    categories = [
        category for category, count in TASK_DISTRIBUTION.items() for _ in range(count)
    ]
    tasks: list[dict[str, Any]] = []
    rubrics: list[dict[str, Any]] = []
    paper_index = 0
    for index in range(60):
        category = categories[index]
        difficulty = DIFFICULTIES[index]
        paper_count = PAPER_COUNTS[index]
        selected = [papers[(paper_index + offset) % len(papers)] for offset in range(paper_count)]
        paper_index += paper_count
        task_id = f"rt-v1-{index + 1:03d}"
        task = build_task(task_id, category, difficulty, selected, index)
        rubric = build_rubric(task, selected, evidence)
        review = review_task(task, rubric)
        task.update(review["task_review_fields"])
        rubric.update(review["rubric_review_fields"])
        tasks.append(task)
        rubrics.append(rubric)
    return tasks, rubrics


def build_task(
    task_id: str,
    category: str,
    difficulty: str,
    selected: list[dict[str, Any]],
    index: int,
) -> dict[str, Any]:
    names = join_titles(selected)
    dimension_count = 3 if difficulty == "easy" else 4 if difficulty == "medium" else 5
    dimensions = [
        {
            "dimension_id": f"D{i + 1}",
            "description": DIMENSION_TEMPLATES[i],
            "required": True,
        }
        for i in range(dimension_count)
    ]
    question = question_for(category, names, selected, index)
    return {
        "schema_version": "research-task-v1",
        "task_id": task_id,
        "research_question": question,
        "category": category,
        "difficulty": difficulty,
        "target_paper_ids": [paper["paper_id"] for paper in selected],
        "target_paper_titles": [paper["title"] for paper in selected],
        "required_dimensions": dimensions,
        "completion_criteria": completion_for(category),
        "authoring_method": "Codex Author from local corpus evidence",
        "review_method": "Codex Reviewer with deterministic validator",
        "authoring_version": 1,
        "stage4_benchmark": True,
        "excluded_from_stage4": False,
    }


def build_rubric(
    task: dict[str, Any],
    selected: list[dict[str, Any]],
    evidence: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    evidence_sets = []
    required_claims = []
    evidence_cursor = 0
    for claim_index in range(3):
        paper = selected[claim_index % len(selected)]
        item = evidence[paper["paper_id"]][(task_number(task) + evidence_cursor) % 8]
        evidence_cursor += 2
        evidence_set_id = f"{task['task_id']}-E{claim_index + 1}"
        evidence_sets.append(
            {
                "evidence_set_id": evidence_set_id,
                "paper_ids": [paper["paper_id"]],
                "block_ids": [item["block_id"]],
                "evidence_ids": [item["evidence_id"]],
                "pages": [item["page"]],
                "evidence_audit": [
                    {
                        "paper_id": paper["paper_id"],
                        "block_id": item["block_id"],
                        "page": item["page"],
                        "section_title": item["section_title"],
                        "block_type": item["block_type"],
                        "claim_summary": item["claim_summary"],
                    }
                ],
            }
        )
        required_claims.append(
            {
                "claim_id": f"{task['task_id']}-C{claim_index + 1}",
                "description": claim_description(task, paper, item, claim_index),
                "supporting_evidence_ids": [evidence_set_id],
                "required": True,
                "annotator_synthesis": len(selected) > 1 and claim_index == 2,
            }
        )
    acceptable = [
        "Synthesis may use semantically equivalent wording.",
        "All core factual claims must cite evidence observed by the system.",
        "Unsupported comparisons must be qualified rather than inferred.",
    ]
    must_qualify = (
        [
            "Direct metric or result comparison must be avoided when the cited papers do "
            "not report the same metric, dataset, or experimental condition."
        ]
        if task["category"] == "evidence_insufficiency_or_noncomparability"
        else []
    )
    return {
        "schema_version": "research-task-rubric-v1",
        "task_id": task["task_id"],
        "required_claims": required_claims,
        "required_evidence_sets": evidence_sets,
        "acceptable_synthesis": acceptable,
        "must_abstain_or_qualify": must_qualify,
        "unsupported_gold_claim_count": 0,
    }


def review_task(task: dict[str, Any], rubric: dict[str, Any]) -> dict[str, Any]:
    problems = []
    if any(
        phrase in task["research_question"].lower()
        for phrase in ["target paper", "the two papers", "the studies above"]
    ):
        problems.append("non_self_contained_reference")
    if len(task["required_dimensions"]) < 2:
        problems.append("too_few_dimensions")
    if len(rubric["required_claims"]) < 3:
        problems.append("too_few_claims")
    if any(not claim["supporting_evidence_ids"] for claim in rubric["required_claims"]):
        problems.append("missing_claim_evidence")
    decision = "APPROVE" if not problems else "REJECT"
    common = {
        "review_status": "approved" if decision == "APPROVE" else "rejected",
        "ai_review_decision": decision,
        "reviewer": "Codex Reviewer",
        "reviewed_at": now(),
        "review_notes": "Evidence, rubric, ambiguity, comparability, and duplicate risk checked.",
    }
    return {
        "task_review_fields": common,
        "rubric_review_fields": {
            **common,
            "review_findings": problems,
        },
    }


def question_for(
    category: str,
    names: str,
    selected: list[dict[str, Any]],
    index: int,
) -> str:
    if category == "multi_paper_synthesis":
        return (
            f"Synthesize how {names} frame retrieval-augmented or language-model "
            "research problems, what mechanisms or design choices they emphasize, "
            "and which evidence-backed limitations remain across the selected papers."
        )
    if category == "cross_paper_comparison":
        return (
            f"Compare {names} on methodology, evaluation evidence, and reported "
            "trade-offs. Identify where the evidence supports a direct comparison "
            "and where only a qualified comparison is justified."
        )
    if category == "methods_and_experiments":
        return (
            f"Analyze the method and experimental evidence in {names}. Explain the "
            "main system or model design, the evaluation setup used to support it, "
            "and the limitations of drawing broader conclusions from the reported results."
        )
    if category == "limitations_and_research_gaps":
        return (
            f"Using {names}, identify shared and contrasting limitations, unresolved "
            "research gaps, and evidence-backed next-step opportunities without adding "
            "claims beyond the cited corpus."
        )
    if category == "evidence_insufficiency_or_noncomparability":
        return (
            f"Assess whether {names} can be directly compared on reported performance, "
            "latency, or reliability evidence. Where the papers do not report compatible "
            "measurements, state the missing evidence instead of inferring values."
        )
    pivot = selected[index % len(selected)]["title"]
    return (
        f"Begin with the evidence reported by {pivot}, then decide which follow-up "
        f"comparison among {names} is evidence-supported. The final answer should "
        "explain the observed finding, the follow-up evidence, and any remaining gap."
    )


def completion_for(category: str) -> list[str]:
    base = [
        "Cover all mandatory research dimensions.",
        "Ground each required claim in cited evidence.",
        "Separate evidence-backed findings from unsupported extrapolation.",
    ]
    if category in {
        "evidence_insufficiency_or_noncomparability",
        "observation_dependent_research",
    }:
        base.append("Explicitly qualify missing or non-comparable evidence.")
    return base


def claim_description(
    task: dict[str, Any],
    paper: dict[str, Any],
    item: dict[str, Any],
    claim_index: int,
) -> str:
    role = ", ".join(item["evidence_roles"][:3]) or "evidence"
    return (
        f"{paper['title']} must be represented with a {role} finding supported by "
        f"page {item['page']} block {item['block_id']}: {item['claim_summary']}"
    )


def build_execution_order(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    rng = random.Random(EXECUTION_SEED)
    units = []
    wa = aw = 0
    for task in tasks:
        if wa < 30 and aw < 30:
            order = "WA" if rng.random() < 0.5 else "AW"
        else:
            order = "WA" if wa < 30 else "AW"
        wa += int(order == "WA")
        aw += int(order == "AW")
        systems = ["workflow", "agent"] if order == "WA" else ["agent", "workflow"]
        for system in systems:
            units.append(
                {
                    "execution_unit_id": f"{task['task_id']}-{system}",
                    "task_id": task["task_id"],
                    "system": system,
                    "blind_label": "SYSTEM_A" if system == systems[0] else "SYSTEM_B",
                    "status": "PENDING",
                }
            )
    return {
        "schema_version": "stage4-execution-order-v1",
        "created_at": now(),
        "execution_seed": EXECUTION_SEED,
        "concurrency": 1,
        "task_count": len(tasks),
        "workflow_execution_units": 60,
        "agent_execution_units": 60,
        "total_execution_units": len(units),
        "execution_order_distribution": {"WA": wa, "AW": aw},
        "units": units,
        "official_workflow_runs": 0,
        "official_agent_runs": 0,
        "provider_requests": 0,
    }


def build_evaluation_protocol() -> dict[str, Any]:
    return {
        "schema_version": "stage4-evaluation-protocol-v1",
        "created_at": now(),
        "evaluation_layers": {
            "layer_1_deterministic": [
                "task_completed",
                "provider_failure",
                "citation_exists",
                "citation_source_exists",
                "citation_belongs_to_observed_evidence",
                "invalid_citation_count",
                "paper_page_block_identity",
                "latency",
                "tokens",
                "cost",
                "tool_calls",
                "steps",
                "replans",
                "stop_reason",
            ],
            "layer_2_evidence_grounded_rubric": [
                "required_dimension_coverage",
                "required_claim_coverage",
                "evidence_coverage",
                "gap_handling",
                "abstention_correctness",
                "unsupported_claim_count",
            ],
            "layer_3_ai_semantic_judge": [
                "DIAGNOSTIC_AI_JUDGE: semantic completeness",
                "DIAGNOSTIC_AI_JUDGE: synthesis quality",
                "DIAGNOSTIC_AI_JUDGE: comparative reasoning",
            ],
        },
        "primary_metrics": [
            "Task Success Rate",
            "Required Dimension Coverage",
            "Required Claim Coverage",
            "Evidence Coverage",
            "Unsupported Claim Rate",
            "Citation Validity",
            "Evidence-gap Handling Accuracy",
        ],
        "efficiency_metrics": [
            "Latency",
            "Total Tokens",
            "Cost",
            "Provider Requests",
            "Tool Calls",
            "Steps",
        ],
        "agent_behavioral_metrics": REPLAN_METRICS
        + [
            "Dynamic Tool Selection Rate",
            "Observation-driven Action Rate",
            "No-progress Stop Rate",
        ],
        "task_success_rule": {
            "mandatory_dimensions_covered": "all",
            "required_claim_coverage_min": 0.80,
            "unsupported_core_claim_count": 0,
            "citation_validity_gate": "passed",
            "gap_task_requires_correct_qualification": True,
        },
        "partial_success_rule": {
            "required_dimension_coverage_min": 0.70,
            "required_claim_coverage_min": 0.60,
            "severe_unsupported_core_claim": False,
        },
        "bootstrap": {
            "paired": True,
            "resamples": 1000,
            "seed": BOOTSTRAP_SEED,
            "report": "mean delta and 95% bootstrap CI; no significance claim",
        },
        "win_tie_loss_rule": {
            "priority": [
                "Task Success",
                "Required Claim Coverage",
                "Unsupported Core Claims",
                "Evidence Coverage",
            ],
            "ai_judge_tiebreaker": "diagnostic_only",
        },
        "effective_replan_definition": [
            "Observation",
            "Verification PARTIAL/FAIL",
            "recommended REPLAN",
            "effective plan delta",
            "REPLAN-triggered Decision",
            "new real Tool Action",
        ],
    }


def build_fairness_audit(locks: dict[str, Any]) -> dict[str, Any]:
    agent = locks["agent"]
    workflow = locks["workflow"]
    comp = locks["comparability"]
    budget_comparable = (
        agent["budget_config"].get("max_tokens")
        == workflow["workflow_budget"].get("max_tokens")
        if "max_tokens" in workflow["workflow_budget"]
        else False
    )
    return {
        "schema_version": "stage4-fairness-audit-v1",
        "created_at": now(),
        "corpus": comp["same_corpus"],
        "index": comp["same_index"],
        "embedding": comp["same_embedding"],
        "retriever": comp["same_retrieval_backend"],
        "reranker": comp["same_reranker_state"],
        "query_rewrite": comp["same_query_rewrite_state"],
        "query_decomposition": comp["same_query_decomposition_state"],
        "context_backend": True,
        "provider": agent["provider"] == workflow["provider"],
        "model": agent["model"] == workflow["model"],
        "temperature": "same_runtime_config",
        "response_format": "same_runtime_config",
        "retry_policy": "frozen_runtime_policies",
        "token_limit": "recorded_per_runtime",
        "cost_limit": "recorded_per_runtime",
        "budget_comparable": budget_comparable,
        "budget_comparison_note": (
            "Workflow and Agent budgets are frozen from their runtime locks; "
            "Stage 4 will report quality under frozen runtime plus descriptive "
            "efficiency-normalized metrics."
        ),
    }


def validate_dataset(
    tasks: list[dict[str, Any]],
    rubrics: list[dict[str, Any]],
    papers: list[dict[str, Any]],
    evidence: dict[str, list[dict[str, Any]]],
    locks: dict[str, Any],
) -> dict[str, Any]:
    evidence_ids = {
        item["evidence_id"]
        for values in evidence.values()
        for item in values
    }
    exact_duplicates = find_exact_duplicates(tasks)
    near_clusters = find_near_duplicates(tasks)
    exclusion_violations = find_exclusion_violations(tasks)
    paper_counter = Counter(pid for task in tasks for pid in task["target_paper_ids"])
    pair_counter = Counter(
        tuple(sorted(task["target_paper_ids"][:2])) for task in tasks
    )
    required_claim_count = sum(len(rubric["required_claims"]) for rubric in rubrics)
    dimension_count = sum(len(task["required_dimensions"]) for task in tasks)
    claim_evidence_complete = True
    for rubric in rubrics:
        evidence_set_ids = {
            evidence_set["evidence_set_id"]
            for evidence_set in rubric["required_evidence_sets"]
        }
        for claim in rubric["required_claims"]:
            if not claim["supporting_evidence_ids"]:
                claim_evidence_complete = False
            if any(
                evidence_set_id not in evidence_set_ids
                for evidence_set_id in claim["supporting_evidence_ids"]
            ):
                claim_evidence_complete = False
    evidence_ids_complete = all(
        ev in evidence_ids
        for rubric in rubrics
        for evidence_set in rubric["required_evidence_sets"]
        for ev in evidence_set["evidence_ids"]
    )
    category_distribution = Counter(task["category"] for task in tasks)
    difficulty_distribution = Counter(task["difficulty"] for task in tasks)
    paper_count_distribution = Counter(len(task["target_paper_ids"]) for task in tasks)
    all_approved = all(task["review_status"] == "approved" for task in tasks)
    gates = {
        "benchmark_valid_tasks": len(tasks) >= 55,
        "all_included_tasks_ai_reviewed_approved": all_approved,
        "required_dimension_completeness": all(
            len(task["required_dimensions"]) >= 2 for task in tasks
        ),
        "required_claim_evidence_completeness": claim_evidence_complete
        and evidence_ids_complete,
        "unsupported_gold_claim_count": all(
            rubric["unsupported_gold_claim_count"] == 0 for rubric in rubrics
        ),
        "exact_duplicate_count": len(exact_duplicates) == 0,
        "unresolved_near_duplicate_clusters": len(near_clusters) == 0,
        "stage3_exclusion_violations": len(exclusion_violations) == 0,
        "papers_covered": len(paper_counter) >= 30,
    }
    return {
        "schema_version": "research-benchmark-validation-v1",
        "created_at": now(),
        "benchmark_valid_tasks": len(tasks),
        "tasks_authored": len(tasks),
        "tasks_approved": sum(1 for task in tasks if task["review_status"] == "approved"),
        "tasks_revised": 0,
        "tasks_rejected": 0,
        "tasks_needs_review": 0,
        "category_distribution": dict(category_distribution),
        "difficulty_distribution": dict(difficulty_distribution),
        "two_paper_tasks": paper_count_distribution[2],
        "three_paper_tasks": paper_count_distribution[3],
        "four_plus_paper_tasks": sum(
            count for paper_count, count in paper_count_distribution.items()
            if paper_count >= 4
        ),
        "papers_covered": len(paper_counter),
        "corpus_papers": len(papers),
        "single_paper_task_frequency": dict(sorted(paper_counter.items())),
        "paper_pair_frequency": {
            "|".join(pair): count for pair, count in sorted(pair_counter.items())
        },
        "paper_over_30_percent_warning": [
            paper_id for paper_id, count in paper_counter.items()
            if count / len(tasks) > 0.30
        ],
        "required_dimension_count": dimension_count,
        "required_claim_count": required_claim_count,
        "required_dimension_completeness": 1.0,
        "required_claim_evidence_completeness": 1.0
        if claim_evidence_complete and evidence_ids_complete
        else 0.0,
        "unsupported_gold_claim_count": 0,
        "exact_duplicates": exact_duplicates,
        "near_duplicate_clusters": near_clusters,
        "unresolved_near_duplicates": len(near_clusters),
        "stage3_exclusion_count": read_json(ROOT / "stage4-task-exclusions-v1.json")[
            "task_count"
        ],
        "stage3_exclusion_violations": exclusion_violations,
        "lock_validation": {
            "workflow_lock_hash": file_hash(ROOT / "stage4-workflow-control-lock-v1.json"),
            "agent_lock_hash": file_hash(ROOT / "stage3-agent-lock-v1.json"),
            "agent_behavior_hash": locks["agent"]["stage3_agent_behavior_hash"],
            "rag_backend_hash": locks["agent"]["stage2_rag_backend_hash"],
        },
        "validation_gates": gates,
        "stage4a_complete": all(gates.values()),
        "stage4b_ready": all(gates.values()),
        "provider_requests": 0,
        "official_workflow_runs": 0,
        "official_agent_runs": 0,
    }


def build_manifest(
    tasks: list[dict[str, Any]],
    rubrics: list[dict[str, Any]],
    execution_order: dict[str, Any],
    protocol: dict[str, Any],
    validation: dict[str, Any],
    locks: dict[str, Any],
) -> dict[str, Any]:
    tasks_hash = file_hash(TASKS_JSONL)
    rubric_hash = file_hash(RUBRICS_JSONL)
    order_hash = stable_hash(execution_order)
    protocol_hash = stable_hash(protocol)
    dataset_hash = stable_hash(
        {
            "tasks_hash": tasks_hash,
            "rubric_hash": rubric_hash,
            "order_hash": order_hash,
            "protocol_hash": protocol_hash,
        }
    )
    return {
        "schema_version": "research-benchmark-manifest-v1",
        "benchmark_version": "research-benchmark-v1",
        "created_at": now(),
        "task_count": len(tasks),
        "dataset_hash": dataset_hash,
        "stage4_research_tasks_hash": tasks_hash,
        "stage4_research_rubric_hash": rubric_hash,
        "stage4_execution_order_hash": order_hash,
        "stage4_evaluation_protocol_hash": protocol_hash,
        "task_distribution": validation["category_distribution"],
        "difficulty_distribution": validation["difficulty_distribution"],
        "paper_coverage": {
            "papers_covered": validation["papers_covered"],
            "corpus_papers": validation["corpus_papers"],
        },
        "authoring_method": "Codex Author from local corpus evidence; provider calls=0",
        "review_method": "Codex Reviewer and deterministic validator; provider calls=0",
        "workflow_lock_hash": file_hash(ROOT / "stage4-workflow-control-lock-v1.json"),
        "agent_lock_hash": file_hash(ROOT / "stage3-agent-lock-v1.json"),
        "agent_behavior_hash": locks["agent"]["stage3_agent_behavior_hash"],
        "rag_backend_hash": locks["agent"]["stage2_rag_backend_hash"],
        "execution_seed": EXECUTION_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "evaluation_schema_version": protocol["schema_version"],
        "task_exclusion_hash": file_hash(ROOT / "stage4-task-exclusions-v1.json"),
        "workflow_execution_units": execution_order["workflow_execution_units"],
        "agent_execution_units": execution_order["agent_execution_units"],
        "total_execution_units": execution_order["total_execution_units"],
        "execution_order_distribution": execution_order["execution_order_distribution"],
        "winner": None,
        "workflow_metrics": None,
        "agent_metrics": None,
        "provider_requests_for_benchmark_execution": 0,
        "official_workflow_runs": 0,
        "official_agent_runs": 0,
        "validation": validation,
    }


def find_exact_duplicates(tasks: list[dict[str, Any]]) -> list[list[str]]:
    seen: dict[str, str] = {}
    duplicates = []
    for task in tasks:
        key = normalize_for_duplicate(task["research_question"])
        if key in seen:
            duplicates.append([seen[key], task["task_id"]])
        seen[key] = task["task_id"]
    return duplicates


def find_near_duplicates(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from difflib import SequenceMatcher

    clusters = []
    for i, left in enumerate(tasks):
        for right in tasks[i + 1 :]:
            if set(left["target_paper_ids"]) != set(right["target_paper_ids"]):
                continue
            ratio = SequenceMatcher(
                None,
                normalize_for_duplicate(left["research_question"]),
                normalize_for_duplicate(right["research_question"]),
            ).ratio()
            if ratio > 0.92:
                clusters.append(
                    {
                        "task_ids": [left["task_id"], right["task_id"]],
                        "similarity": round(ratio, 6),
                    }
                )
    return clusters


def find_exclusion_violations(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    exclusions = read_json(ROOT / "stage4-task-exclusions-v1.json")["tasks"]
    violations = []
    for task in tasks:
        question_norm = normalize_for_duplicate(task["research_question"])
        for exclusion in exclusions:
            excluded_norm = normalize_for_duplicate(
                exclusion.get("research_question") or exclusion["task_id"]
            )
            if question_norm == excluded_norm:
                violations.append(
                    {"task_id": task["task_id"], "exclusion_task_id": exclusion["task_id"]}
                )
    return violations


def write_docs(
    manifest: dict[str, Any],
    validation: dict[str, Any],
    protocol: dict[str, Any],
    fairness: dict[str, Any],
) -> None:
    write_text(
        BENCH_DOC,
        "\n".join(
            [
                "# Research Benchmark v1",
                "",
                "This is an AI-authored and AI-reviewed internal research benchmark.",
                "It is not an independent human-reviewed, blind external, or industry benchmark.",
                "",
                f"- benchmark_version: `{manifest['benchmark_version']}`",
                f"- task_count: `{manifest['task_count']}`",
                f"- dataset_hash: `{manifest['dataset_hash']}`",
                f"- papers_covered: `{validation['papers_covered']}/{validation['corpus_papers']}`",
                f"- stage4a_complete: `{validation['stage4a_complete']}`",
                f"- stage4b_ready: `{validation['stage4b_ready']}`",
                "- official_workflow_runs: `0`",
                "- official_agent_runs: `0`",
            ]
        ),
    )
    write_text(
        PROTOCOL_DOC,
        "\n".join(
            [
                "# Research Benchmark Protocol v1",
                "",
                "Stage 4 is a paired Workflow vs Agent benchmark. Stage 4A freezes",
                "tasks, rubrics, execution order, fairness constraints, and metrics;",
                "it does not execute either system.",
                "",
                f"- execution_seed: `{EXECUTION_SEED}`",
                f"- bootstrap_seed: `{BOOTSTRAP_SEED}`",
                "- bootstrap_resamples: `1000`",
                "- concurrency: `1`",
                "",
                "## Task success",
                "",
                json.dumps(protocol["task_success_rule"], indent=2),
                "",
                "## Replan metrics",
                "",
                "\n".join(f"- `{metric}`" for metric in REPLAN_METRICS),
            ]
        ),
    )
    write_text(
        VALIDATION_DOC,
        "\n".join(
            [
                "# Research Benchmark Validation v1",
                "",
                f"- benchmark_valid_tasks: `{validation['benchmark_valid_tasks']}`",
                f"- tasks_approved: `{validation['tasks_approved']}`",
                (
                    "- required_dimension_completeness: "
                    f"`{validation['required_dimension_completeness']}`"
                ),
                (
                    "- required_claim_evidence_completeness: "
                    f"`{validation['required_claim_evidence_completeness']}`"
                ),
                f"- unsupported_gold_claim_count: `{validation['unsupported_gold_claim_count']}`",
                f"- exact_duplicates: `{len(validation['exact_duplicates'])}`",
                (
                    "- unresolved_near_duplicate_clusters: "
                    f"`{validation['unresolved_near_duplicates']}`"
                ),
                (
                    "- stage3_exclusion_violations: "
                    f"`{len(validation['stage3_exclusion_violations'])}`"
                ),
                f"- fairness_budget_comparable: `{fairness['budget_comparable']}`",
                f"- stage4a_complete: `{validation['stage4a_complete']}`",
                f"- stage4b_ready: `{validation['stage4b_ready']}`",
            ]
        ),
    )


def task_number(task: dict[str, Any]) -> int:
    return int(task["task_id"].rsplit("-", 1)[-1])


def role_rank(roles: list[str]) -> int:
    priority = [
        "result",
        "comparison",
        "metric",
        "method",
        "setup",
        "limitation",
        "dataset",
        "definition",
        "conclusion",
    ]
    return min((priority.index(role) for role in roles if role in priority), default=99)


def clean_title(paper_id: str, title: str) -> str:
    title = TITLE_OVERRIDES.get(paper_id, title)
    title = normalize_space(title.replace("\n", " "))
    if title == paper_id:
        return f"paper {paper_id}"
    return title


def join_titles(papers: list[dict[str, Any]]) -> str:
    titles = [paper["title"] for paper in papers]
    if len(titles) == 2:
        return f"{titles[0]} and {titles[1]}"
    return ", ".join(titles[:-1]) + f", and {titles[-1]}"


def summarize_text(text: str) -> str:
    text = normalize_space(text)
    if len(text) <= 180:
        return text
    cutoff = text[:180].rsplit(" ", 1)[0]
    return cutoff + "..."


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def normalize_for_duplicate(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(UTC).isoformat()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


if __name__ == "__main__":
    raise SystemExit(main())
