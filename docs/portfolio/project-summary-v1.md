# PaperResearch Project Summary v1

PaperResearch is a paper-centered RAG and Deep Research system. It emphasizes
real engineering closure: versioned corpora and indexes, reproducible
evaluation artifacts, strict citation validation, provider usage accounting,
checkpointing, recovery drills, Docker deployment checks, and explicit
limitations on every public claim.

## Current release

- Recommended version: `v1.1.0-portfolio`
- Package/runtime version: `1.1.0+portfolio`
- Primary production model used in final QA evidence: `deepseek` /
  `deepseek-v4-flash`
- Default reranker: disabled
- Current release readiness: `READY_WITH_SEMANTIC_EVALUATION_LIMITATION`

## Main capabilities

- PDF ingestion with parsing metadata, OCR fallback validation, chunking, and
  Qdrant indexing.
- Hybrid retrieval over a fixed 34-paper production corpus.
- Production QA over 50 human-reviewed internal development records with
  structured output and strict citation checks.
- Production Deep Research path with Hybrid Retrieval, evidence deduplication,
  section-scoped citation allowlists, structured DeepSeek synthesis, replayable
  raw-response observability, and deterministic Markdown report generation.
- Stateful Research Agent runtime with plan, tool execution, observation,
  evidence state, verification, checkpoint/resume, retry, budget, stop
  conditions, and trace.
- Frozen Workflow vs Agent paired benchmark over 60 internally authored tasks.

## Final benchmark summary

| Metric | Frozen Workflow | Research Agent |
|---|---:|---:|
| Paired units | 60 | 60 |
| Completed | 0 | 56 |
| Failed | 60 | 4 |
| Provider requests | 73 | 414 |
| Total tokens | 159,769 | 414,569 |
| Estimated cost | $0.03092796 | $0.06447728 |
| P50 latency | 10.43 s | 15.99 s |
| P95 latency | 26.84 s | 21.45 s |

The Agent won the structured outcome proxy on 56/60 units and tied on 4/60.
This is an operational and structural-proxy result, not a semantic quality proof.

## Required limitations

- The benchmark tasks were internally authored and reviewed. They are not a
  public benchmark and not a strict blind generalization benchmark.
- Stage 4 is not a strict equal-budget causal ablation:
  `budget_comparable=false`.
- Claim, dimension, evidence coverage, unsupported-claim rate, and citation
  validity are structured proxies. They were not replaced by content-level
  semantic rubric scoring.
- `semantic_judge_complete = false` because fair blind judge input text was not
  available for all compared outputs.
- No effective live replan was observed in the final benchmark:
  `LIVE_EFFECTIVE_REPLAN_NOT_OBSERVED`.
- The release must not claim commercial production readiness, long-term
  endurance, hallucination elimination, or strong grounding proof.
