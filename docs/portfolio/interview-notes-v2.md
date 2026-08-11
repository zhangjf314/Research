# PaperResearch Interview Notes v2

## One-line summary

PaperResearch is an evidence-grounded academic-paper RAG system with a frozen
Workflow baseline and a separate stateful Research Agent runtime, evaluated with
reproducible artifacts and explicit claim boundaries.

## What is technically interesting

- The project separates retrieval quality, citation validity, structured output
  reliability, runtime recovery, and Agent control-flow behavior instead of
  collapsing them into one opaque demo score.
- The Research Agent is not a renamed workflow. It has explicit state,
  observations, Evidence State, policy decisions, verification-before-finish,
  bounded replan support, checkpoint/resume, budget, retry, and trace.
- The final Stage 4 comparison preserves the Workflow baseline and uses a frozen
  RAG backend for comparability.
- Later runtime hotfixes added Agent final-report synthesis and UI mode clarity,
  but those are not retroactively counted as Stage 4 benchmark improvements.

## Safe claims

- "Built a PDF RAG and Deep Research system with Docker deployment, Qdrant,
  PostgreSQL, Redis, strict citation validation, and provider usage accounting."
- "Implemented a separate Research Agent runtime with state/observation-driven
  action selection and checkpointable execution."
- "Ran a frozen Workflow-vs-Agent paired benchmark and published the limitations
  of the benchmark."

## Claims to avoid

- "Production-grade commercial system."
- "Strict blind benchmark proves broad generalization."
- "Agent semantic quality improved by 93.3 percentage points."
- "Hallucinations are eliminated."
- "Long-term stability is proven."
