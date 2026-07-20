# End-to-End DeepSeek Production v2

Status: `PASSED`

The 50-item DeepSeek Direct Full QA engineering gate remains passed and was not rerun in Stage 13.38.

The Deep Research final retry was executed once after the corrected synthesize provider-smoke passed.

## Final retry

- Run ID: `live-q003-ed900ef2e202`
- Parent run ID: `live-q003-cbc99df5b041`
- Question ID: `q003`
- Status: `completed`
- Nodes: `plan -> retrieve -> assess_evidence -> synthesize -> validate_citations -> persist_trace`
- Retrieval calls: `1`
- LLM request attempts: `1`
- Provider completed requests: `1`
- Usage records: `1`
- Input/output/total tokens: `6714` / `82` / `6796`
- Cost: `$0.00096292`
- Elapsed seconds: `1.337596`
- Claim count: `2`
- Citation count: `2`
- Citation validation: `passed`
- Active reserved tokens: `0`
- Reranker called: `false`
- Template fallback: `false`

Current conclusion: `A. Deep Research has passed and the project can proceed to safety audit, soak, and Portfolio release closure.`
