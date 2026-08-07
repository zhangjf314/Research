from __future__ import annotations

# ruff: noqa: E501
import argparse
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from paper_research.evaluation.rag_benchmark import read_jsonl
from paper_research.evaluation.rag_gold import (
    normalize_question,
    overlap_ratio,
    write_json_artifact,
)

ROOT = Path("data/evaluation/rag-benchmark")
DOCS = Path("docs/rag-benchmark")


def question_signature(question: str) -> str:
    normalized = normalize_question(question)
    normalized = normalized.replace("paper ", "")
    normalized = normalized.replace("according to ", "")
    return normalized


def connected_components(nodes: set[str], edges: list[tuple[str, str]]) -> list[set[str]]:
    graph: dict[str, set[str]] = defaultdict(set)
    for left, right in edges:
        graph[left].add(right)
        graph[right].add(left)
    components = []
    seen = set()
    for node in sorted(nodes):
        if node in seen or node not in graph:
            continue
        stack = [node]
        component = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(graph[current] - component)
        seen |= component
        components.append(component)
    return components


def classify_cluster(records: list[dict[str, Any]]) -> tuple[str, str]:
    categories = {record.get("category") for record in records}
    evidence_sets = {
        tuple(sorted((item.get("paper_id"), item.get("block_id")) for item in record.get("gold_evidence", [])))
        for record in records
    }
    paper_sets = {tuple(sorted(record.get("gold_paper_ids", []))) for record in records}
    claim_sets = {
        tuple(sorted(str(claim.get("text", "")) for claim in record.get("required_claims", [])))
        for record in records
    }
    searched_sets = {tuple(sorted(record.get("searched_paper_ids", []))) for record in records}
    if all(not record.get("answerable") for record in records) and len(searched_sets) > 1:
        return "DISTINCT_EVIDENCE", "Similar unanswerable template but different searched paper scopes."
    if len(evidence_sets) == 1 and len(claim_sets) == 1:
        return "DUPLICATE", "Same normalized question cluster, same evidence, and same required claims."
    if len(categories) > 1:
        return "DISTINCT_CAPABILITY", "Similar wording but different benchmark categories/capabilities."
    if len(evidence_sets) > 1 or len(paper_sets) > 1:
        return "DISTINCT_EVIDENCE", "Template-like wording but different papers or evidence blocks."
    return "NEEDS_REVIEW", "Similarity could not be resolved deterministically."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "gold-ai-reviewed-full-v1.jsonl")
    parser.add_argument("--threshold", type=float, default=0.82)
    args = parser.parse_args()

    records = read_jsonl(args.input)
    by_id = {str(record["question_id"]): record for record in records}
    edges: list[tuple[str, str]] = []
    ids = set(by_id)
    for left_index, left in enumerate(records):
        for right in records[left_index + 1 :]:
            if overlap_ratio(str(left.get("question", "")), str(right.get("question", ""))) >= args.threshold:
                edges.append((str(left["question_id"]), str(right["question_id"])))
    clusters = []
    for index, component in enumerate(connected_components(ids, edges), start=1):
        cluster_records = [by_id[qid] for qid in sorted(component)]
        classification, reason = classify_cluster(cluster_records)
        clusters.append(
            {
                "cluster_id": f"dup-cluster-{index:03d}",
                "classification": classification,
                "question_ids": sorted(component),
                "size": len(component),
                "reason": reason,
            }
        )

    unresolved = [cluster for cluster in clusters if cluster["classification"] in {"DUPLICATE", "NEEDS_REVIEW"}]
    payload = {
        "schema_version": "rag-gold-duplicate-clusters-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "input": str(args.input),
        "threshold": args.threshold,
        "near_duplicate_pair_count": len(edges),
        "cluster_count": len(clusters),
        "unresolved_cluster_count": len(unresolved),
        "classification_distribution": {
            key: sum(1 for cluster in clusters if cluster["classification"] == key)
            for key in ["DUPLICATE", "DISTINCT_CAPABILITY", "DISTINCT_EVIDENCE", "DISTINCT_COMPARISON", "NEEDS_REVIEW"]
        },
        "clusters": clusters,
    }
    write_json_artifact(ROOT / "gold-duplicate-clusters-v1.json", payload)

    lines = [
        "# RAG Gold duplicate clusters v1",
        "",
        f"- near_duplicate_pair_count: {len(edges)}",
        f"- cluster_count: {len(clusters)}",
        f"- unresolved_cluster_count: {len(unresolved)}",
        "",
        "| classification | count |",
        "| --- | ---: |",
    ]
    for key, value in payload["classification_distribution"].items():
        lines.append(f"| {key} | {value} |")
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "gold-duplicate-clusters-v1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 1 if unresolved else 0


if __name__ == "__main__":
    raise SystemExit(main())
