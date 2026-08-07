# Stage 2B query rewrite / decomposition plan v1

- split: `dev`
- dev_question_count: `98`
- test_questions_allowed: `False`
- q0_source: `STAGE2A_FROZEN_DEV_RESULT`
- selected_stage2a_candidate: `R3_current_hybrid`
- reranker: `disabled`

## Configs

- `Q0_CURRENT_HYBRID`: Original question, Stage 2A frozen Current Hybrid rows.
- `Q1_SINGLE_REWRITE_REPLACE`: Single rewritten query replaces original.
- `Q2_ORIGINAL_PLUS_SINGLE_REWRITE`: Original query plus single rewrite, fused.
- `Q3_ORIGINAL_PLUS_DECOMPOSITION`: Original query plus 0-3 decomposition queries, fused; max 4 retrieval queries per logical question.

## Decision gate

At least one candidate must satisfy the pre-registered gain threshold without unacceptable new misses or paper recall regression. Otherwise Q0 remains selected.

## Protocol guardrails

- test split evaluation
- reranker
- context selection changes
- chunking changes
- embedding changes
- QA/generation benchmark
- Research Agent
- gold leakage in rewrite prompt
