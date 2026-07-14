# Evaluation Evidence Index

The machine-readable index is
[`data/evaluation/evaluation-evidence-index.json`](../data/evaluation/evaluation-evidence-index.json).

## Authority rules

1. Original isolated run directories override top-level summaries.
2. Provider-reported usage overrides tokenizer estimates and conservative reservations.
3. Human review overrides token-set semantic signals.
4. `latest-successful` is only a formal selection policy; failed attempts remain visible.
5. Oracle results are diagnostic and never replace Production results.

| Evidence | Authority | Principal limitation |
|---|---|---|
| `production-corpus-v1.json` | Corpus boundary and fixture exclusion | 33 papers plus one release acceptance document |
| `gold-set-v1.jsonl` | Human-approved answers and evidence | Not independent double-blind review |
| `retrieval-gold-v2.jsonl` | Retrieval query/scope/filter protocol | No global-scope items |
| `retrieval-ablation-v2.json` | Corrected Hash/Jina retrieval comparison | Mostly paper-scoped evaluation |
| `reranker-ablation-v1.json` | Reranker negative result | Candidate-set reranking only |
| `qa-production-v1.json` | Real QA outputs, usage and automated metrics | Automated metrics do not establish human support |
| `qa-context-diagnostics-v1.json` | Retrieved/Oracle diagnosis | Token-set signal is not citation correctness |
| `retrieval-context-optimization-v1.json` | Context strategy comparison | Did not clear QA gates |
| `citation-human-audit-summary-v1.json` | Approved AI-assisted citation labels | Failure-stratified 30-item sample |
| `deep-research-smoke-v1/runs/` | Original run, ledger, trace and checkpoint records | Engineering smoke only |
| `deep-research-smoke-v1.json` | Formal latest-successful selection | Subordinate to run directories |
| `stage11d-final-audit-v1.json` | Isolation/accounting/security audit | Successful resumed completion unverified |
| `v1-release-gates.json` | RC/v1 decision thresholds | Threshold changes require review |
| `stage12-test-report-v1.json` | Current pytest/Ruff/compileall/diff execution | Local Windows execution only |

Earlier `results-v1.json` and `results-production-v1.json` remain historical evidence but are
superseded for Production decisions because they were generated while Gold annotations were
pending or before real providers were evaluated.
