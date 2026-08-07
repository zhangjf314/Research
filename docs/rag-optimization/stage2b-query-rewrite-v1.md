# Stage 2B query rewrite / decomposition v1

- split: `dev`
- dev_question_count: `98`
- test_questions_evaluated: `0`
- test_protocol_violation: `False`
- selected_stage2a_candidate: `Current Hybrid`
- selected_candidate: `Q0_CURRENT_HYBRID`
- reranker_enabled: `False`
- llm_generation_requests: `0`

## Metrics

| Config | Recall@10 | EvidenceCov@10 | ReqClaimCov@10 | FullCov@10 | MRR@10 | nDCG@10 | PaperRecall@10 | P95 latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q0_CURRENT_HYBRID | 0.469697 | 0.469697 | 0.561321 | 0.329545 | 0.219467 | 0.250609 | 0.926136 | 5307.235 |
| Q1_SINGLE_REWRITE_REPLACE | 0.455871 | 0.455871 | 0.528302 | 0.352273 | 0.213515 | 0.239102 | 0.897727 | 5070.701 |
| Q2_ORIGINAL_PLUS_SINGLE_REWRITE | 0.325947 | 0.325947 | 0.433962 | 0.181818 | 0.166797 | 0.279072 | 0.886364 | 5070.701 |
| Q3_ORIGINAL_PLUS_DECOMPOSITION | 0.34072 | 0.34072 | 0.438679 | 0.204545 | 0.164002 | 0.252639 | 0.920455 | 10125.209 |

## Rewrite accounting

```json
{
  "logical_questions": 98,
  "single_rewrite_requests": 0,
  "decomposition_requests": 0,
  "provider_requests": 0,
  "historical_provider_requests_from_cache": 196,
  "effective_provider_requests_for_artifact": 196,
  "provider_failures": 0,
  "rewrite_success_rate": 1.0,
  "decomposition_success_rate": 1.0,
  "input_tokens": 35705,
  "output_tokens": 20851,
  "total_tokens": 56556,
  "estimated_cost": 0.0,
  "cost_accounting_status": "unavailable_cache_schema_gap",
  "usage_sources": [
    "cache_text_estimated_after_interrupted_provider_run"
  ],
  "rewrite_latency_p50": 1124.047,
  "rewrite_latency_p95": 1534.237,
  "average_generated_queries": 1.397959,
  "queries_p50": 2.0,
  "queries_p95": 4.0
}
```

## Selection gate

```json
{
  "Q1_SINGLE_REWRITE_REPLACE": false,
  "Q2_ORIGINAL_PLUS_SINGLE_REWRITE": false,
  "Q3_ORIGINAL_PLUS_DECOMPOSITION": false
}
```

## Interpretation

Q0 Current Hybrid remains selected because no rewrite/decomposition candidate met the pre-registered gate.

This DEV-only experiment did not evaluate TEST, did not invoke QA generation, and did not enable reranking.
