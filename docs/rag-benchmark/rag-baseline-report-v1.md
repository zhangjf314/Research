# RAG Baseline Benchmark v1

## 1. Benchmark Scope

AI-authored and AI-reviewed internal benchmark. Not a blind benchmark. Semantic claim support is not formally validated.

## 2. Dataset

- 146 questions
- 98 Dev
- 48 Test

## 3. Frozen Baseline Configuration

- system_under_test_commit: `f97746e84b98d6b4e07984a3abbdab206f156839`
- baseline_config_hash: `8817891ed73fc0c0fa2f3a7fc90baf3591b3ab0006934bbb058cbff87e657c94`
- LLM: `deepseek/deepseek-v4-flash`
- Embedding: `jina/jina-embeddings-v5-text-small`
- Reranker: `disabled`
- Query Rewrite: `disabled`

## 4. Retrieval Results

- Recall@5 / @10 / @20: `0.322854` / `0.483586` / `0.635606`
- MRR@10: `0.246414`
- nDCG@10: `0.273671`
- Paper Recall@10: `0.901515`
- Evidence Coverage@10: `0.483586`

## 5. Generation Results

- Required Claim Coverage: `0.058081`
- Supported Claim Ratio: `0.0`
- Citation Precision / Recall: `0.0` / `0.102273`
- Abstention Accuracy: `0.714286`
- Tokens: `569166`
- Cost: `0.0820372`

## 11. Failure Attribution

- largest_overall_bottleneck: `RETRIEVAL_ROOTED`
- distribution: `{'ABSTENTION_FAILURE': 4, 'GENERATION_OMISSION': 61, 'PARTIAL_EVIDENCE_RETRIEVAL': 39, 'RANKING_ERROR': 10, 'RETRIEVAL_MISS': 27, 'RETRIEVAL_ROOTED': 71, 'UNANSWERABLE_FALSE_POSITIVE': 14}`

## 13. Stage 2 Optimization Hypotheses

- H1: If many Gold blocks appear in Top20 but miss Top5/Top10, a reranker may improve ranking.
- H2: If multi-evidence or cross-paper recall is weak, query rewrite or retrieval expansion should be evaluated.
- H3: If Gold evidence enters context but claim coverage remains low, context selection or generation should be studied.
- H4: If dense and sparse errors are complementary, Hybrid weighting deserves ablation.

- stage_1_complete: `True`
- stage_2_ready: `True`
