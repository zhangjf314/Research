# RAG Baseline Report v1

Status: `FRAMEWORK_READY_BASELINE_NOT_RERUN`

- baseline_config_hash: `8817891ed73fc0c0fa2f3a7fc90baf3591b3ab0006934bbb058cbff87e657c94`
- dataset_hash: `01660d3dd734003bcf1bbedebfec30f224524a91190bb6295ea51069a416cffa`
- retrieval_benchmark_ready: `True`
- generation_benchmark_ready: `True`
- bad_case_taxonomy_ready: `True`

## Required answers

- Current Retrieval biggest problem: Formal Stage 1 retrieval benchmark has not been run yet; current evidence shows historical gold-block-present gaps in Full QA artifacts.
- Current Generation biggest problem: Historical Full QA diagnostics show low required-claim and exact citation recall; semantic support is not formally validated.
- Worst question type: paper_contributions
- Best question type: not yet measured by the Stage 1 harness
- Unanswerable: Existing Gold contains two unanswerable questions; formal Stage 1 benchmark run is pending authorization.
- Retrieval vs Generation failures: {'retrieval_failed_items_in_historical_generation_artifact': 29, 'generation_or_citation_failed_items_in_historical_generation_artifact': 21}
- Stage 2 hypothesis: Hypothesis only: first validate whether retrieval misses versus context/citation selection dominate exact-Gold failures before changing algorithms.

No Stage 1 optimization was implemented.
