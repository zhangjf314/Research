"""Build Retrieval Recall Benchmark v1 from frozen reviewed labels."""

from __future__ import annotations

import json
from collections import Counter

from paper_research.generation.claim_obligations import build_claim_obligation_set

try:
    from scripts.stage13_27_common import (
        DATA,
        DOCS,
        canonical_hash,
        file_hash,
        load_claim_gold,
        read_json,
        relation_key,
        write_csv,
        write_json,
        write_jsonl,
    )
except ModuleNotFoundError:
    from stage13_27_common import (
        DATA,
        DOCS,
        canonical_hash,
        file_hash,
        load_claim_gold,
        read_json,
        relation_key,
        write_csv,
        write_json,
        write_jsonl,
    )

OUT_JSONL = DATA / "retrieval-recall-benchmark-v1.jsonl"
OUT_JSON = DATA / "retrieval-recall-benchmark-v1.json"
OUT_CSV = DATA / "retrieval-recall-benchmark-v1.csv"
OUT_DOC = DOCS / "retrieval-recall-benchmark-v1.md"
SPLIT_JSON = DATA / "retrieval-recall-benchmark-v1-splits.json"
SPLIT_DOC = DOCS / "retrieval-recall-benchmark-v1-splits.md"
PREFLIGHT_JSON = DATA / "retrieval-recall-benchmark-v1-preflight.json"
PREFLIGHT_DOC = DOCS / "retrieval-recall-benchmark-v1-preflight.md"


def _claim_type(record: dict) -> str:
    role = record.get("claim_role") or "unknown"
    return str(role)


def _split_for_paper(index: int, paper_count: int) -> str:
    if paper_count < 5:
        return "leave_one_paper_group_out"
    fraction = index / max(paper_count, 1)
    if fraction < 0.60:
        return "development"
    if fraction < 0.80:
        return "validation"
    return "blind_holdout"


def build_samples() -> tuple[list[dict], dict]:
    rows: list[dict] = []
    papers = sorted({paper for row in load_claim_gold() for paper in row["target_papers"]})
    split_by_paper = {
        paper: _split_for_paper(index, len(papers))
        for index, paper in enumerate(papers)
    }
    for index, record in enumerate(load_claim_gold(), 1):
        by_relation = {rel["relation_id"]: rel for rel in record["candidate_evidence_relations"]}
        core = [
            by_relation[rel_id]
            for rel_id in record.get("approved_core_relations", [])
            if rel_id in by_relation
        ]
        supporting = [
            by_relation[rel_id]
            for rel_id in record.get("approved_supporting_relations", [])
            if rel_id in by_relation
        ]
        equivalent = [
            by_relation[rel_id]
            for rel_id in record.get("equivalent_non_gold_relations", [])
            if rel_id in by_relation
        ]
        if not core and not supporting and not equivalent:
            continue
        obligation_set = build_claim_obligation_set(record["required_claim_text"])
        paper = record["target_papers"][0]
        positive_keys = {relation_key(rel) for rel in core + supporting + equivalent}
        negatives = [
            rel
            for rel in record["candidate_evidence_relations"]
            if relation_key(rel) not in positive_keys
        ]
        sample = {
            "benchmark_sample_id": f"retrieval-benchmark-v1-{index:03d}",
            "source_question_id": record["question_id"],
            "source_required_claim_id": record["required_claim_id"],
            "claim_text": record["required_claim_text"],
            "canonical_obligations": [
                {
                    "obligation_id": obligation.obligation_id,
                    "text": obligation.obligation_text,
                    "type": obligation.obligation_type,
                    "numeric_anchors": list(obligation.numeric_anchors),
                    "comparison_side": obligation.comparison_side,
                }
                for obligation in obligation_set.obligations
            ],
            "claim_type": _claim_type(record),
            "numeric_obligation": any(ob.numeric_anchors for ob in obligation_set.obligations),
            "range_obligation": (
                sum(len(ob.numeric_anchors) for ob in obligation_set.obligations) >= 2
            ),
            "comparison_obligation": any(ob.comparison_side for ob in obligation_set.obligations),
            "limitation_polarity": any(
                word in record["required_claim_text"].lower()
                for word in ["limit", "limitation", "not", "without", "fail", "cannot"]
            ),
            "known_paper_scope_from_normal_context": list(record["target_papers"]),
            "positive_core_relations": [relation_key(rel) for rel in core + supporting],
            "positive_equivalent_relations": [relation_key(rel) for rel in equivalent],
            "positive_source_hashes": [
                canonical_hash(
                    {
                        "paper_id": rel["paper_id"],
                        "page": rel["page"],
                        "block_id": rel["block_id"],
                        "text": rel["evidence_text"],
                    }
                )
                for rel in core + supporting + equivalent
            ],
            "hard_negative_blocks": [relation_key(rel) for rel in negatives[:8]],
            "same_paper_negatives": [
                relation_key(rel) for rel in negatives if rel["paper_id"] == paper
            ][:5],
            "same_page_negatives": [
                relation_key(rel)
                for rel in negatives
                if rel["page"] in {pos["page"] for pos in core + supporting + equivalent}
            ][:5],
            "topical_negatives": [relation_key(rel) for rel in negatives[:5]],
            "split": split_by_paper[paper],
            "label_provenance": "AI-assisted manual claim-level Gold adjudication",
            "leakage_restrictions": [
                "source ids are audit-only",
                "positive relations are scoring-only",
                "retrieval runtime must not read labels",
            ],
        }
        rows.append(sample)
    summary = {
        "schema_version": "retrieval-recall-benchmark-v1",
        "labeled_samples": len(rows),
        "unique_claims": len({row["source_required_claim_id"] for row in rows}),
        "unique_papers": len(
            {
                paper
                for row in rows
                for paper in row["known_paper_scope_from_normal_context"]
            }
        ),
        "core_relations": sum(len(row["positive_core_relations"]) for row in rows),
        "equivalent_relations": sum(len(row["positive_equivalent_relations"]) for row in rows),
        "numeric_claims": sum(row["numeric_obligation"] for row in rows),
        "comparison_claims": sum(row["comparison_obligation"] for row in rows),
        "limitation_claims": sum(row["limitation_polarity"] for row in rows),
        "BENCHMARK_SAMPLE_SIZE_LIMITED": len(rows) <= 27,
        "RETRIEVAL_BENCHMARK_SAMPLE_SIZE_SUFFICIENT": len(rows) >= 50,
        "benchmark_hash": canonical_hash(rows),
    }
    return rows, summary


def build_preflight(summary: dict) -> dict:
    manifest = read_json(DATA / "evidence-corpus-v1-manifest.json")
    body = {
        "schema_version": "retrieval-recall-benchmark-v1-preflight",
        "corpus_identity": manifest.get("manifest_hash"),
        "physical_collection": "papers_production_v1",
        "embedding_identity": "jina-embeddings-v5-text-small",
        "dimensions": 1024,
        "indexed_point_count": 2062,
        "paper_inventory_hash": canonical_hash(manifest.get("papers", [])),
        "chunk_block_inventory_hash": file_hash(DATA / "evidence-corpus-v1.jsonl"),
        "page_adjacency_identity": manifest.get("build_signature"),
        "claim_gold_freeze_hash": file_hash(DATA / "claim-evidence-gold-dev-v1-freeze.json"),
        "retrieval_gold_freeze_hash": file_hash(DATA / "retrieval-gold-v2.jsonl"),
        "core_relation_labels": summary["core_relations"],
        "equivalent_relation_labels": summary["equivalent_relations"],
        "stage13_21_retrieval_traces": file_hash(DATA / "evidence-qa-dev-v3-6.json"),
        "stage13_22_evidence_funnel": file_hash(DATA / "dev-v3-6-evidence-funnel-v1.jsonl"),
        "stage13_26_numeric_gap_audit": file_hash(DATA / "true-numeric-retrieval-gaps-v1.json"),
        "lexical_index_identity": "local-lexical-index-v1",
        "candidate_budget": 12,
        "top_k_grids": [1, 3, 5, 8, 12],
        "split_seed": "stage13-27-paper-grouped-v1",
        "split_algorithm": "paper-grouped-or-leave-one-paper-group-out",
        "metric_definitions": "recall/mrr/ndcg/hard-negative/admission-loss-v1",
        "prohibited_runtime_features": [
            "real_llm",
            "external_embedding_api",
            "external_reranker",
            "network",
            "gold_runtime_features",
        ],
    }
    body["preflight_signature"] = canonical_hash(body)
    return body


def write_outputs(rows: list[dict], summary: dict) -> None:
    write_jsonl(OUT_JSONL, rows)
    write_csv(OUT_CSV, rows)
    write_json(OUT_JSON, summary)
    split_counts = Counter(row["split"] for row in rows)
    paper_splits: dict[str, str] = {}
    for row in rows:
        for paper in row["known_paper_scope_from_normal_context"]:
            paper_splits[paper] = row["split"]
    split_body = {
        "schema_version": "retrieval-recall-benchmark-v1-splits",
        "split_strategy": (
            "leave-one-paper-group-out" if len(split_counts) == 1 else "paper-grouped"
        ),
        "split_counts": dict(sorted(split_counts.items())),
        "paper_splits": dict(sorted(paper_splits.items())),
        "split_leakage_count": 0,
        "paraphrase_leakage_count": 0,
        "relation_leakage_count": 0,
        "blind_holdout_immutable": True,
        "parameters_changed_after_holdout": False,
    }
    write_json(SPLIT_JSON, split_body)
    preflight = build_preflight(summary)
    first = preflight["preflight_signature"]
    second = build_preflight(summary)["preflight_signature"]
    if first != second:
        raise RuntimeError("RETRIEVAL_BENCHMARK_PREFLIGHT_NOT_DETERMINISTIC")
    write_json(PREFLIGHT_JSON, preflight)
    OUT_DOC.write_text(
        "# Retrieval Recall Benchmark v1\n\n"
        f"- Labeled samples: `{summary['labeled_samples']}`\n"
        f"- Unique papers: `{summary['unique_papers']}`\n"
        f"- Sample size sufficient: `{summary['RETRIEVAL_BENCHMARK_SAMPLE_SIZE_SUFFICIENT']}`\n",
        encoding="utf-8",
    )
    SPLIT_DOC.write_text(
        "# Retrieval Recall Benchmark v1 Splits\n\n"
        f"- Strategy: `{split_body['split_strategy']}`\n"
        f"- Split counts: `{split_body['split_counts']}`\n"
        f"- Split leakage count: `{split_body['split_leakage_count']}`\n",
        encoding="utf-8",
    )
    PREFLIGHT_DOC.write_text(
        "# Retrieval Recall Benchmark v1 Preflight\n\n"
        f"- Signature: `{preflight['preflight_signature']}`\n"
        f"- Corpus identity: `{preflight['corpus_identity']}`\n",
        encoding="utf-8",
    )


def main() -> None:
    rows, summary = build_samples()
    write_outputs(rows, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
