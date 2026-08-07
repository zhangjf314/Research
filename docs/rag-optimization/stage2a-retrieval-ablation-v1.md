# Stage 2A retrieval ablation v1

- split: `dev`
- dev_question_count: `98`
- test_questions_evaluated: `0`
- selected_stage2a_candidate: `RR1_hybrid_lexical_rerank`
- llm_requests: `0`

| Config | Recall@5 | Recall@10 | Recall@20 | MRR@10 | nDCG@10 | EvidenceCov@10 | P50 | P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.281629 | 0.469697 | 0.611174 | 0.219467 | 0.250609 | 0.469697 | 2233.879 | 3967.058 |
| Dense | 0.263068 | 0.396212 | 0.547159 | 0.206029 | 0.222883 | 0.396212 | 2403.385 | 5307.235 |
| Sparse | 0.25947 | 0.452652 | 0.607765 | 0.19922 | 0.237179 | 0.452652 | 2403.385 | 5307.235 |
| Hybrid | 0.281629 | 0.469697 | 0.611174 | 0.219467 | 0.250609 | 0.469697 | 2403.385 | 5307.235 |
| Hybrid + Rerank | 0.27822 | 0.473864 | 0.611174 | 0.284776 | 0.276777 | 0.473864 | 2406.822 | 5311.218 |

## Complementarity

{
  "both_failure": 25,
  "both_success": 36,
  "dense_only_success": 10,
  "hybrid_recovers_dense_failure": 12,
  "hybrid_recovers_sparse_failure": 8,
  "sparse_only_success": 17
}

## Selected vs baseline

{
  "paired": {
    "win_count": 10,
    "tie_count": 69,
    "loss_count": 9
  },
  "bootstrap": {
    "recall_at_10": {
      "metric": "recall_at_10",
      "mean_delta": 0.004167,
      "ci95_low": -0.056818,
      "ci95_high": 0.063826,
      "resamples": 1000,
      "seed": 20260807
    },
    "mrr_at_10": {
      "metric": "mrr_at_10",
      "mean_delta": 0.065309,
      "ci95_low": 0.00542,
      "ci95_high": 0.125383,
      "resamples": 1000,
      "seed": 20260807
    },
    "evidence_coverage_at_10": {
      "metric": "evidence_coverage_at_10",
      "mean_delta": 0.004167,
      "ci95_low": -0.056818,
      "ci95_high": 0.063826,
      "resamples": 1000,
      "seed": 20260807
    }
  },
  "bad_case_delta": {
    "before": {
      "PARTIAL_EVIDENCE_RETRIEVAL": 27,
      "RANKING_ERROR": 5,
      "RETRIEVAL_MISS": 20,
      "UNANSWERABLE_FALSE_POSITIVE": 10
    },
    "after": {
      "PARTIAL_EVIDENCE_RETRIEVAL": 27,
      "RANKING_ERROR": 5,
      "RETRIEVAL_MISS": 20,
      "UNANSWERABLE_FALSE_POSITIVE": 10,
      "WRONG_PAPER": 1
    },
    "delta": {
      "PARTIAL_EVIDENCE_RETRIEVAL": 0,
      "RANKING_ERROR": 0,
      "RETRIEVAL_MISS": 0,
      "UNANSWERABLE_FALSE_POSITIVE": 0,
      "WRONG_PAPER": 1
    }
  }
}
