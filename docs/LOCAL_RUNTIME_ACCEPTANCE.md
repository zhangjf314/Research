# Local Runtime Acceptance

## Status

`PORTFOLIO_LOCAL_RUNTIME_VALIDATED`

The validated local Docker runtime used DeepSeek for LLM calls and SiliconFlow
`Qwen/Qwen3-Embedding-0.6B` for embeddings. This document records runtime
capability evidence only; it does not alter the RAG Quality v3 evaluation or
promotion conclusions.

## Validated paths

- UI paper upload, Paper Library display, paper selection, and indexing.
- Direct QA with a non-empty answer and rendered citation/evidence reference.
- Deep Research workflow with local retrieval, terminal `COMPLETED`, and a
  non-empty final report.
- Research Agent with a terminal `COMPLETED`/`SUCCESS` state, real tool calls,
  non-empty evidence state, verifier `PASS`, and a non-empty final report.

## Provider compatibility

DeepSeek uses Chat Completions with forced ordinary function calls for typed
structured output and Agent planning/decisions. The adapter validates function
name and JSON arguments before existing local Pydantic, evidence, and tool
policy validation. Plain Direct QA remains evidence-bound JSON-object output.

The adaptation is confined to the provider boundary. It does not change
chunking, retrieval, BM25, RRF, embedding configuration, Qdrant semantics, or
the production P0 default.

## Promotion boundary

RAG Quality v3 remains closed:

- `NO_CANDIDATE_PROMOTED`
- `PRODUCTION_P0_RETAINED`
- `FULL_QA_NOT_ELIGIBLE`

No experimental reranker or listwise selector is connected to production.
