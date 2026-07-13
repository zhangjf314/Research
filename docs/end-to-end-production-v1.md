# End-to-End Production v1

Status: **BLOCKED — not executed with real models**
Date: 2026-07-13

The code now supports an OpenAI-compatible structured LLM, OpenAI-compatible Embedding,
and HTTP Cross-Encoder through unified provider interfaces. This environment has no
LLM, Embedding, or Reranker API key/base URL configured, and the gold set has 0/50 human
approvals. Running a report under these conditions would be a baseline/template run, not
a Production run, so no real-model report was fabricated.

Production Token, cost, first-token latency, total-model latency, unsupported-claim rate,
and quality comparison are therefore `null`/unavailable. The Stage 9 baseline report and
trace remain unchanged in their original files.

To unblock: configure Production providers, rebuild `papers_production_v1`, complete human
review, then run the production Deep Research and seven-way evaluation.
