# RAG benchmark protocol v1

This protocol governs Stage 1C and later RAG optimization work. It protects the benchmark from accidental test-set tuning while keeping the repository honest about its scope.

## Dataset roles

`rag-gold-v1` is intended to be an internal, human-reviewed Gold dataset for this project. It is not a hidden public benchmark and must not be described as an independently authored blind benchmark.

The final dataset is frozen only after enough approved records pass validation. Any later correction must create a new version such as `rag-gold-v1.0.1` or `rag-gold-v2`; scripts must not silently rewrite `rag-gold-v1`.

## Dev split

The dev split may be used repeatedly for:

- Hybrid parameter selection.
- Reranker comparison.
- Query rewrite experiments.
- Context selection experiments.
- Bad-case analysis.
- Regression checks during Stage 2.

Dev results may guide implementation decisions, but every report must disclose that dev was used for selection.

## Test split

The test split is a held-out internal test split. It is not a secret blind benchmark.

Test may be used only for:

- Baseline freeze measurement.
- The final selected Stage 2 configuration.
- Final ablation confirmation.

Test must not be used for:

- Top-K selection.
- RRF weight tuning.
- Reranker threshold tuning.
- Prompt tuning.
- Query rewrite tuning.
- Context selector iteration.

If test performance is poor, record the result and open a new experiment version. Do not continue tuning against the same test split and then claim unbiased test performance.

## Historical 50-question generation artifact

The existing 50-question generation artifact is retained as `HISTORICAL_REFERENCE_ONLY`. It must not be merged with future 100-question results to claim an official 150-question baseline because the dataset version, runtime, and evaluation process differ.

The official baseline must be run once after `rag-gold-v1` is frozen.

## Stage 1C order

Stage 1C Official Baseline must run in this order:

1. Retrieval Benchmark.
2. Retrieval Bad Case Analysis.
3. Generation Benchmark.
4. Generation Bad Case Analysis.
5. Aggregate Baseline Report.

Only after this sequence should Stage 2 choose whether to first test Hybrid, Reranker, Query Rewrite, or Context Selection.
