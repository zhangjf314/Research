import json
from pathlib import Path

from paper_research.evaluation.answer_metrics import evaluate_answer
from paper_research.evaluation.dataset import EvaluationItem, RetrievalPrediction
from paper_research.evaluation.observability import UsageRecorder
from paper_research.evaluation.retrieval_metrics import evaluate_retrieval


def test_retrieval_metrics_do_not_reward_duplicate_papers() -> None:
    item = EvaluationItem(
        id="q1",
        question="question",
        question_type="method",
        relevant_paper_ids=["p1"],
    )
    prediction = RetrievalPrediction(
        item_id="q1", ranked_paper_ids=["p1", "p1", "p1", "p2"]
    )

    metrics = evaluate_retrieval([item], [prediction])

    assert metrics["hit_at_1"] == 1
    assert metrics["mrr"] == 1
    assert metrics["ndcg_at_10"] == 1


def test_answer_metrics_detect_supported_and_cited_answer() -> None:
    metrics = evaluate_answer(
        "Self attention connects every token [paper-1, p.3].",
        ["Self attention connects every token in the sequence."],
        [{"valid": True}],
        ["self attention connects tokens"],
    )

    assert metrics["faithfulness"] == 1
    assert metrics["citation_coverage"] == 1
    assert metrics["citation_correctness"] == 1


def test_usage_recorder_persists_latency_tokens_and_cost(tmp_path: Path) -> None:
    path = tmp_path / "usage.jsonl"
    with UsageRecorder(path).record(
        "evaluation", input_tokens=100, output_tokens=20, estimated_cost_usd=0.01
    ):
        pass

    event = json.loads(path.read_text(encoding="utf-8"))
    assert event["operation"] == "evaluation"
    assert event["input_tokens"] == 100
    assert event["output_tokens"] == 20
    assert event["estimated_cost_usd"] == 0.01
    assert event["latency_ms"] >= 0
