# PaperResearch Interview Notes v1

## Short project pitch

PaperResearch is a paper RAG and Deep Research system built around auditable
engineering evidence. It covers ingestion, parsing, hybrid retrieval, structured
QA, citation validation, Deep Research report generation, checkpoint/recovery,
usage accounting, Docker deployment checks, and a frozen Workflow vs Agent
benchmark.

## What is technically strongest

- The system treats evaluation artifacts as first-class release evidence.
- Provider usage, cost, failure, retry, checkpoint, and trace data are persisted
  rather than inferred after the fact.
- Citation validation is strict: generated references must map to allowed
  evidence identifiers rather than free-form guessed page/block triples.
- The Deep Research report path was corrected from evidence-dump reporting to
  structured synthesis with global and section-scoped citation allowlists.
- The Agent path is separate from the frozen Workflow path, preserving a fair
  Stage 4 comparison boundary.

## Workflow vs Agent result

In the frozen 60-unit paired benchmark, the Agent completed 56 tasks while the
Workflow completed 0. The Agent used more provider requests, tokens, and cost.
The result supports the claim that a stateful Agent runtime improved operational
completion under this internal benchmark, but it does not prove semantic
research quality superiority.

## Accurate public wording

Use wording like:

> Built a paper RAG and Deep Research system with production-model QA,
> citation validation, checkpointed execution, Docker deployment checks, and a
> 60-unit internally reviewed Workflow vs Agent benchmark.

For the benchmark:

> The Agent completed 56/60 internally reviewed paired tasks versus 0/60 for the
> frozen Workflow under the same frozen RAG backend, while using more model
> requests and tokens. Metrics are structural proxies; semantic judge evaluation
> remains a documented limitation.

## Wording to avoid

- Strict blind benchmark.
- Production-grade generalization.
- Commercial production-ready system.
- Fully validated semantic benchmark.
- Hallucination-free or hallucination eliminated.
- Agent proved better semantic research quality.
- Long-term stability proven.

## Likely interview questions

### Why keep the failed Workflow?

The Workflow is the frozen control group. Rewriting it after seeing Agent
results would destroy comparability.

### Why not call the Stage 4 result semantic quality?

The scoring artifacts verified structured task completion and proxy metrics, but
not independent content-level rubric judgments for every emitted claim,
dimension, and evidence item.

### Why is budget comparability false?

The Agent made more decisions and tool/model calls. That is part of the Agent
design, but it means the benchmark is not an equal-budget causal ablation.

### What would be needed next?

A future release should add fair output text capture for both paths, run a
semantic blind judge or human rubric audit, and optionally introduce a larger
external holdout set.
