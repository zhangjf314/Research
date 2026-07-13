import math

from paper_research.evaluation.dataset import EvaluationItem, RetrievalPrediction


def evaluate_retrieval(
    items: list[EvaluationItem],
    predictions: list[RetrievalPrediction],
    k_values: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    prediction_map = {prediction.item_id: prediction for prediction in predictions}
    metrics: dict[str, float] = {}
    for k in k_values:
        hits = []
        recalls = []
        for item in items:
            ranked = prediction_map.get(
                item.id, RetrievalPrediction(item_id=item.id, ranked_paper_ids=[])
            )
            retrieved = ranked.ranked_paper_ids[:k]
            relevant = set(item.relevant_paper_ids)
            matched = len(relevant & set(retrieved))
            hits.append(float(matched > 0))
            recalls.append(matched / len(relevant))
        metrics[f"hit_at_{k}"] = _mean(hits)
        metrics[f"recall_at_{k}"] = _mean(recalls)
    reciprocal_ranks = []
    ndcg_scores = []
    block_hits = []
    for item in items:
        prediction = prediction_map.get(
            item.id, RetrievalPrediction(item_id=item.id, ranked_paper_ids=[])
        )
        relevant = set(item.relevant_paper_ids)
        rank = next(
            (
                index
                for index, paper_id in enumerate(prediction.ranked_paper_ids, start=1)
                if paper_id in relevant
            ),
            None,
        )
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        seen: set[str] = set()
        gains = []
        for paper_id in prediction.ranked_paper_ids[:10]:
            gain = 1.0 if paper_id in relevant and paper_id not in seen else 0.0
            gains.append(gain)
            seen.add(paper_id)
        dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
        ideal = sum(1 / math.log2(index + 2) for index in range(min(len(relevant), 10)))
        ndcg_scores.append(dcg / ideal if ideal else 0.0)
        if item.relevant_block_ids:
            block_hits.append(
                float(bool(set(item.relevant_block_ids) & set(prediction.ranked_block_ids[:10])))
            )
    metrics["mrr"] = _mean(reciprocal_ranks)
    metrics["ndcg_at_10"] = _mean(ndcg_scores)
    metrics["block_hit_at_10"] = _mean(block_hits)
    return {name: round(value, 6) for name, value in metrics.items()}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
