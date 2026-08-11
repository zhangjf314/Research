# PaperResearch Project Summary v2

PaperResearch is a paper-centered RAG, Deep Research Workflow, and Research
Agent system. It emphasizes real engineering closure: reproducible ingestion and
indexing, frozen retrieval configurations, strict citation validation,
checkpoint/recovery behavior, request ledger accounting, runtime capability
checks, benchmark artifacts, and explicit limitations.

## Current version state

| Item | Value |
| --- | --- |
| Current released tag | `v1.1.0-portfolio` |
| Current package/runtime version | `1.1.0+portfolio` |
| v1.2.0 status | Candidate documentation/readiness audit only |
| v1.2.0 version bump | Not performed in this branch |
| Recommended public wording | Portfolio engineering release with documented limitations |

## Capabilities

- Academic PDF ingestion with parser routing, structured blocks, chunks, and
  index metadata.
- Hybrid retrieval over the frozen Stage 2 backend.
- Production QA evidence with real provider runs and strict citation checks.
- Deep Research Workflow with structured synthesis, citation validation,
  replayable raw-response observability, and deterministic Markdown rendering.
- Research Agent runtime with state, Evidence State, dynamic action selection,
  verification, checkpoint/resume, retry bounds, budget limits, stop conditions,
  trace, and an Agent final-report presentation layer.
- Frozen Workflow-vs-Agent paired benchmark artifacts.

## Benchmark interpretation

The official Stage 4 result remains the frozen v1.1.0 paired benchmark. It
supports an engineering claim about structured task completion and runtime
control, not a strong semantic-quality or generalization claim.

The post-v1.1 Agent final-report synthesis layer is a current runtime capability
and should be demonstrated separately from the Stage 4 benchmark.

## Public limitations

- Internally authored/reviewed evaluation data is not a public benchmark.
- No large-scale strict blind generalization benchmark is included.
- Stage 4 structured coverage metrics are proxies, not human semantic rubric
  judgments over every claim.
- Effective live replan was not observed in the final Stage 4 benchmark.
- Commercial production readiness and long-term endurance are not claimed.
