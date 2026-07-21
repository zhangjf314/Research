# Portfolio Evaluation Policy v1

This project uses a portfolio-oriented evaluation policy. It keeps evaluation
claims honest without requiring research-lab-scale blind benchmark construction.

## Dataset tiers

### gold-dev-v1

`gold-dev-v1` is the existing 50-record, human-processed internal development
evaluation set.

Allowed uses:

- Embedding comparison.
- Retrieval parameter selection.
- Reranker comparison.
- Full QA.
- Claim-level citation evaluation.
- Regression testing.

External wording:

> 基于 50 条人工审核的内部评测数据完成检索和问答评测

Do not call it a blind holdout, strict generalization benchmark, public
benchmark, or production benchmark.

### retrieval-diagnostic-v1

`retrieval-diagnostic-v1` is the existing 27-record claim-level diagnostic
benchmark. It has been inspected during development.

Allowed uses:

- Failure analysis.
- Category diagnostics.
- Retrieval configuration regression checks.
- Obvious degradation checks.

It is not blind and must not be used to prove generalization.

### shadow-holdout-pilot-v1

`shadow-holdout-pilot-v1` is an optional small blind pilot, recommended but not
required for Portfolio v1.0.

Recommended shape:

- 10-15 new human-reviewed samples.
- At least 5 papers not used for retrieval tuning.
- At most 3 samples per paper.
- At least 2 unanswerable questions.
- Built after retrieval configuration freeze.
- Annotation does not show current Retrieval Top-K.
- Marked `revealed` after the first formal run.

If completed and no obvious regression appears, it may support:

> 另外使用 10～15 条未参与调参的新样本进行小规模盲测抽查

It still does not support a statistically strong generalization claim.

## Why Portfolio v1.0 does not require 50 new blind records

A 50-record strict blind holdout with claim-level evidence adjudication is
valuable, but it is disproportionate for a personal job-search portfolio. The
project instead separates:

- development evaluation;
- diagnostic regression analysis;
- optional small blind sanity checks;
- future research-grade benchmark work.

This preserves honesty while keeping the release scope practical.

## Metrics suitable for README/resume

Allowed:

- Metrics on the 50-record human-reviewed internal development set, clearly
  labeled as internal development evaluation.
- Reranker, embedding, retrieval, QA, and citation metrics if their dataset tier
  is named.
- Diagnostic findings from the 27 claim-level benchmark, labeled as diagnostic.
- Optional pilot results, labeled as small-scale pilot if such data is created.

Forbidden:

- "strict generalization benchmark"
- "large-scale independent blind benchmark"
- "production-grade generalization proven"
- "statistically sufficient holdout"
- "在严格盲测集上证明了泛化能力"
- "达到生产级泛化"
- "通过大规模独立 benchmark"
- "public benchmark"

## Full QA gate

Full QA may run when:

- `gold_dev_approved_count > 0`
- `HYBRID_RETRIEVAL_V1_DEV_GATE=PASSED`
- production Embedding is available
- production Collection is available
- real LLM provider preflight has passed
- Claim Validator is available

The optional shadow pilot is not a hard blocker for Full QA.

## Future upgrade path

To support strong generalization claims later:

1. Freeze retrieval and QA configuration.
2. Build a new independent blind holdout.
3. Use enough samples for the intended statistical claim.
4. Ensure annotation does not reveal model outputs or Retrieval Top-K.
5. Run the holdout once, then mark it revealed.
6. Report confidence intervals and limitations.
