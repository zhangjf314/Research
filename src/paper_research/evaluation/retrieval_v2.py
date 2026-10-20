"""Qualified evidence-block metrics for Phase 1, without invented claim labels."""

import math
from statistics import mean


def block_metrics(ranked: list[str], gold: set[str]) -> dict[str, float]:
    if not gold:
        raise ValueError("answerable block metrics require nonempty gold")
    metrics = {f"recall_at_{k}": len(set(ranked[:k]) & gold) / len(gold) for k in (5, 10, 20)}
    first = next((rank for rank, cid in enumerate(ranked, 1) if cid in gold), None)
    metrics["mrr"] = 1 / first if first else 0.0
    seen: set[str] = set()
    dcg = 0.0
    for rank, cid in enumerate(ranked[:10], 1):
        if cid in gold and cid not in seen:
            dcg += 1 / math.log2(rank + 1)
        seen.add(cid)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(10, len(gold)) + 1))
    metrics["ndcg_at_10"] = dcg / ideal
    metrics["gold_block_recall_at_5"] = metrics["recall_at_5"]
    return metrics


def aggregate(rows: list[dict]) -> dict[str, float]:
    if not rows:
        raise ValueError("benchmark requires at least one scored query")
    keys = list(rows[0]["metrics"])
    metrics = {key: mean(row["metrics"][key] for row in rows) for key in keys}
    metrics["latency_ms"] = mean(row["latency_ms"] for row in rows)
    metrics["rerank_activation_rate"] = mean(float(row["rerank_triggered"]) for row in rows)
    return {key: round(value, 6) for key, value in metrics.items()}
