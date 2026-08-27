"""C2 rerank-only paired evaluation over the sealed C1-R0 candidate snapshot."""
# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from dotenv import load_dotenv
from run_c1_gold_free_execution import h, questions_and_gold

from paper_research.chunking.types import Chunk
from paper_research.config import Settings
from paper_research.retrieval.context_builder import ContextBuilder
from paper_research.retrieval.fusion import FusedResult
from paper_research.retrieval.reranker import RerankerProviderError, SiliconFlowReranker

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ROOT = ROOT
C1 = ROOT / "artifacts" / "rag-quality-v3" / "c1" / "execution"
OUT = ROOT / "artifacts" / "rag-quality-v3" / "c2" / "execution"
FREEZE = "e501e0b42bea4a159c98c7d84189c03bfaabf379"
SNAPSHOT_HASH = "1f6178e54a50002658cd98c885fca3a04547e358313d127a64f0efacf2c62482"
ARMS = ("C2-R0", "C2-R1")


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def mean(rows: list[dict], key: str) -> float:
    return round(sum(row[key] for row in rows) / len(rows), 6) if rows else 0.0


def load_snapshot() -> dict:
    path = C1 / "snapshots" / "c1-r0-candidate-snapshot-v1.json"
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    if snapshot.get("seal_status") != "SEALED" or snapshot.get("global_sha256") != SNAPSHOT_HASH:
        raise RuntimeError("C2_SEALED_SNAPSHOT_INVALID")
    if len(snapshot.get("records", [])) != 176:
        raise RuntimeError("C2_SEALED_SNAPSHOT_INVALID:question_count")
    for record in snapshot["records"]:
        candidates = record.get("candidates", [])
        if not candidates or record.get("candidate_count_actual") != len(candidates):
            raise RuntimeError(
                f"C2_SEALED_SNAPSHOT_INVALID:{record.get('question_id')}:candidate_count"
            )
    return snapshot


def make_result(candidate: dict) -> FusedResult:
    return FusedResult(
        chunk=Chunk(
            chunk_id=candidate["candidate_unit_id"],
            paper_id=candidate["canonical_document_id"],
            block_ids=candidate["neutral_source_block_ids"],
            section_path=["sealed-c1-r0"],
            block_type="sealed_candidate",
            page_start=candidate["source_spans"][0][0],
            page_end=candidate["source_spans"][0][1],
            chunk_text=candidate["text"],
            token_count=max(1, len(candidate["text"].split())),
        ),
        score=float(candidate["fused_score"]),
        dense_rank=candidate.get("dense_rank"),
        sparse_rank=candidate.get("bm25_rank"),
    )


def covered(candidate_sets: list[list[str]], gold_map: dict[str, str]) -> set[str]:
    return {gold_map[item] for values in candidate_sets for item in values if item in gold_map}


def stable_pool_hash(candidates: list[dict]) -> str:
    return hashlib.sha256(
        "\n".join(candidate["candidate_unit_id"] for candidate in candidates).encode()
    ).hexdigest()


def reranker() -> SiliconFlowReranker:
    load_dotenv(CANONICAL_ROOT / ".env", override=True)
    settings = Settings()
    key = settings.embedding_api_key or settings.siliconflow_embedding_api_key
    if not key:
        raise RuntimeError("SILICONFLOW_RERANKER_CREDENTIAL_MISSING")
    return SiliconFlowReranker(
        base_url="https://api.siliconflow.cn/v1",
        api_key=key,
        model="Qwen/Qwen3-Reranker-0.6B",
        max_retries=2,
        allow_fallback=False,
    )


def reordered(
    query: str, candidates: list[dict], provider: SiliconFlowReranker
) -> tuple[list[dict], dict]:
    original = [make_result(candidate) for candidate in candidates]
    outcome = provider.rerank_with_trace(query, original, len(original))
    if outcome.fallback_occurred or outcome.output_count != len(original):
        raise RuntimeError("SILICONFLOW_RERANKER_INVALID_RESPONSE")
    scores = {item.chunk.chunk_id: item.score for item in outcome.results}
    if set(scores) != {item.chunk.chunk_id for item in original}:
        raise RuntimeError("C2_PAIRED_IDENTITY_VIOLATION:provider_membership")
    original_rank = {
        candidate["candidate_unit_id"]: rank for rank, candidate in enumerate(candidates, 1)
    }
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            -float(scores[candidate["candidate_unit_id"]]),
            original_rank[candidate["candidate_unit_id"]],
            candidate["candidate_unit_id"],
        ),
    )
    return ranked, {
        "api_request_count": outcome.api_request_count,
        "latency_ms": outcome.latency_ms,
        "prompt_tokens": outcome.prompt_tokens,
        "total_tokens": outcome.total_tokens,
        "model": outcome.model,
        "provider": outcome.provider,
        "scores": scores,
    }


def score_question(question: dict, candidates: list[dict], gold_map: dict[str, str]) -> dict:
    top = candidates[:5]
    top_results = [make_result(candidate) for candidate in top]
    context = ContextBuilder(include_neighbors=False, max_characters=10**9, max_tokens=12000).build(
        top_results
    )
    candidate_by_id = {candidate["candidate_unit_id"]: candidate for candidate in candidates}
    pool_ids = [candidate["neutral_source_block_ids"] for candidate in candidates]
    top_ids = [candidate["neutral_source_block_ids"] for candidate in top]
    context_ids = [
        candidate_by_id[result.chunk_id]["neutral_source_block_ids"] for result in context
    ]
    gold = question["gold"]
    pool_gold, top_gold, context_gold = (
        covered(pool_ids, gold_map),
        covered(top_ids, gold_map),
        covered(context_ids, gold_map),
    )
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(gold), 10) + 1))
    row = {
        "id": question["id"],
        "dataset": question["dataset"],
        "category": question["category"],
        "doc": question["doc"],
        "gold_recall_5": len(top_gold & gold) / len(gold),
        "mrr": next(
            (1 / rank for rank, ids in enumerate(top_ids, 1) if covered([ids], gold_map) & gold),
            0.0,
        ),
        "ndcg10": sum(
            (1 if covered([ids], gold_map) & gold else 0) / math.log2(rank + 1)
            for rank, ids in enumerate(pool_ids[:10], 1)
        )
        / ideal,
        "context_precision": sum(bool(covered([ids], gold_map) & gold) for ids in context_ids)
        / len(context_ids)
        if context_ids
        else 0.0,
        "context_recall": len(context_gold & gold) / len(gold),
    }
    if question["claims"] is None:
        row.update(
            {
                "claim_status": "METRIC_NOT_COMPUTABLE",
                "required_claim_coverage@5": None,
                "multi_evidence_complete_rate@5": None,
            }
        )
    else:
        claims = question["claims"]
        pool_claims = sum(bool(pool_gold & claim) for claim in claims)
        top_claims = sum(bool(top_gold & claim) for claim in claims)
        context_claims = sum(bool(context_gold & claim) for claim in claims)
        row.update(
            {
                "claim_status": "COMPUTABLE",
                "required_claim_coverage@5": top_claims / len(claims),
                "multi_evidence_complete_rate@5": float(
                    len(claims) > 1 and top_claims == len(claims)
                ),
                "candidate_loss": len(claims) - pool_claims,
                "ranking_loss": pool_claims - top_claims,
                "packing_loss": top_claims - context_claims,
            }
        )
    return row


def metrics(rows: list[dict]) -> tuple[dict, dict]:
    computable = [row for row in rows if row["claim_status"] == "COMPUTABLE"]
    multi = [row for row in computable if row["category"] == "multi_evidence"]
    result = {
        "GoldR@5": mean(rows, "gold_recall_5"),
        "MRR": mean(rows, "mrr"),
        "NDCG@10": mean(rows, "ndcg10"),
        "context_precision": mean(rows, "context_precision"),
        "context_recall": mean(rows, "context_recall"),
        "required_claim_coverage@5": mean(computable, "required_claim_coverage@5"),
        "multi_evidence_complete_rate@5": mean(multi, "multi_evidence_complete_rate@5"),
        "B_D_claim_metrics": "METRIC_NOT_COMPUTABLE",
    }
    losses = {
        name: sum(row.get(name, 0) for row in computable)
        for name in ("candidate_loss", "ranking_loss", "packing_loss")
    }
    return result, losses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=ARMS)
    args = parser.parse_args()
    snapshot = load_snapshot()
    questions, gold_map = questions_and_gold()
    question_by_id = {question["id"]: question for question in questions}
    if set(question_by_id) != {record["question_id"] for record in snapshot["records"]}:
        raise RuntimeError("C2_PAIRED_IDENTITY_VIOLATION:question_set")
    provider = reranker() if args.arm == "C2-R1" else None
    rows, traces = [], []
    calls = failures = 0
    latency_ms = 0.0
    for number, record in enumerate(snapshot["records"], 1):
        question = question_by_id[record["question_id"]]
        original = record["candidates"]
        try:
            if provider is None:
                ranked, trace = original, {"api_request_count": 0, "latency_ms": 0.0, "scores": {}}
            else:
                ranked, trace = reordered(question["query"], original, provider)
        except (RerankerProviderError, RuntimeError) as exc:
            failures += 1
            save(
                OUT / "failures" / f"{args.arm.lower()}-failure-v1.json",
                {
                    "status": "SILICONFLOW_RERANKER_UNAVAILABLE",
                    "question_id": question["id"],
                    "pool_sha256": stable_pool_hash(original),
                    "exception": str(exc),
                    "completed_questions": len(rows),
                    "provider_calls": calls,
                    "retrieval_calls": 0,
                    "embedding_calls": 0,
                },
            )
            raise RuntimeError("SILICONFLOW_RERANKER_UNAVAILABLE") from exc
        calls += trace["api_request_count"]
        latency_ms += trace["latency_ms"]
        if {candidate["candidate_unit_id"] for candidate in original} != {
            candidate["candidate_unit_id"] for candidate in ranked
        }:
            raise RuntimeError(f"C2_PAIRED_IDENTITY_VIOLATION:{question['id']}")
        rows.append(score_question(question, ranked, gold_map))
        traces.append(
            {
                "question_id": question["id"],
                "pool_sha256": stable_pool_hash(original),
                "input_count": len(original),
                "candidate_unit_ids": [candidate["candidate_unit_id"] for candidate in ranked],
                "rerank_scores": trace["scores"],
            }
        )
        print(f"{args.arm} evaluated {number}/176", flush=True)
    result_metrics, losses = metrics(rows)
    output = {
        "arm": args.arm,
        "status": "PASS",
        "attempted_questions": len(rows),
        "paired_identity": {
            "status": "PASS",
            "questions": len(rows),
            "snapshot_global_sha256": SNAPSHOT_HASH,
        },
        "metrics": result_metrics,
        "losses": losses,
        "provider": {
            "name": "siliconflow" if provider else "none",
            "model": "Qwen/Qwen3-Reranker-0.6B" if provider else "none",
            "logical_requests": calls,
            "failures": failures,
            "latency_ms": round(latency_ms, 3),
        },
        "invariants": {
            "retrieval_calls": 0,
            "embedding_calls": 0,
            "blind_data": 0,
            "production_change": "no",
        },
    }
    save(OUT / "runs" / f"{args.arm.lower()}-questions-v1.json", rows)
    save(
        OUT / "traces" / f"{args.arm.lower()}-ordering-v1.json",
        {"arm": args.arm, "records": traces, "global_sha256": h(traces)},
    )
    save(OUT / "summaries" / f"{args.arm.lower()}-summary-v1.json", output)


if __name__ == "__main__":
    main()
