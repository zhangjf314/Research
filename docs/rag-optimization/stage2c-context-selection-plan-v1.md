# Stage 2C context selection plan v1

- dev_questions: `None`
- dev_answerable: `None`
- test_questions_evaluated: `0`
- context_trace_source: `None`

```json
{
  "allowed_features": [
    "retrieval rank",
    "retrieval score",
    "paper_id",
    "section_path",
    "chunk text token estimate",
    "block overlap"
  ],
  "base_chain": "Current Hybrid; no reranker; no query rewrite; no generation LLM",
  "commit": "24cc578df5d8a1cfa2e1a4c0ba64118fd2e3caf1",
  "context_selection_hypothesis_supported": true,
  "created_at": "2026-08-07T17:27:10.469537+00:00",
  "forbidden_features": [
    "gold answers",
    "gold block ids",
    "required claims",
    "LLM selector",
    "prompt changes",
    "retrieval changes",
    "reranker",
    "query rewrite",
    "test split"
  ],
  "hypothesis_source": "stage2c-evidence-funnel-v1",
  "offline_gate": {
    "full_context_coverage_gain_min": 0.05,
    "required_claim_context_coverage_gain_min": 0.05,
    "single_hop_context_coverage_no_obvious_drop": true,
    "token_p95_max_baseline_multiplier": 1.1
  },
  "schema_version": "stage2c-context-selection-plan-v1",
  "selectors": {
    "C0_BASELINE": "Frozen baseline context selection reconstructed deterministically.",
    "C1_SCORE_BUDGETED_DEDUP": "Deterministic score-ordered selector that removes duplicate block coverage within the baseline token budget.",
    "C2_DIVERSITY_AWARE": "Deterministic selector that caps per-paper and per-section early selection, then refills by rank within the baseline token budget."
  },
  "split": "dev",
  "test_questions_allowed": false,
  "test_questions_evaluated": 0
}
```
