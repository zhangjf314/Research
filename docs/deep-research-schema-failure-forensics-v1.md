# Deep Research schema failure forensics v1

This is a sanitized forensic summary for failed task
`bef5c92c-e195-49c5-aade-3c6394be6aa6`.

The two provider requests completed, but the previous implementation only returned
usage after successful business schema validation. Therefore the API reported:

- `provider_completed_request_count=2`
- `total_tokens=0`
- `estimated_cost_usd=0`
- `active_reserved_tokens=0`

That is an accounting bug. A completed provider request with returned or estimable
usage must be counted even when JSON parsing, Pydantic validation, citation
allowlist validation, or report quality validation fails.

## Frozen response availability

The raw provider responses for the two attempts were not persisted by the prior
adapter path. No raw response, finish reason, provider response ID, or per-attempt
usage record is available for replay. This report therefore does not include any
raw model text, full prompt, API key, or full paper evidence.

Status:

- `HISTORICAL_RAW_RESPONSE_OBSERVABILITY_GAP=ACCEPTED`
- `historical_frozen_response_replay=NOT_AVAILABLE`
- `acceptance_reason=legacy adapter did not persist raw provider responses`

This must not be restated as historical replay passing or as historical response
validation. The next provider call now persists local-only raw model content under
`.runtime/research-synthesis-provider/<task_id>/attempt-XX.json` before JSON
parsing and schema validation.

## Failure classification

Attempt 1:

- `PROMPT_CONTRACT_FAILURE`
- `ACCOUNTING_BUG`

Attempt 2:

- `PROMPT_CONTRACT_FAILURE`
- `ACCOUNTING_BUG`

More specific schema paths cannot be reconstructed without the missing raw
responses. The hotfix adds future response accounting and sanitized audit support
so that subsequent failures are replayable without committing raw provider output.

## Corrective actions

- Structured JSON transport now extracts usage immediately after HTTP success.
- Business schema failures now carry billable usage records.
- Research synthesis aggregates usage across all attempts.
- The synthesis prompt now uses a short executable JSON skeleton.
- Bounded normalization is limited to fences, single-object extraction, whitelisted
  field aliases, section ID canonicalization, and exact duplicate citation removal.

## Replayable smoke failure update

Task `ce25169e-7ab7-4d1b-92f2-fec77df06f0a` was executed after raw response
audit persistence was added. Its raw provider responses remain local-only under
`.runtime/research-synthesis-provider/ce25169e-7ab7-4d1b-92f2-fec77df06f0a/`
and are not committed.

Offline replay of the persisted responses now gives a narrower failure
classification:

- Attempt 1: JSON parsing passed and `research_gaps` object shape is compatible
  with the updated `ResearchGap` schema, but section-scoped citation validation
  fails. The model cited `[E14]` in `methods` and `[E2]` in `results`, while
  those IDs were not in the corresponding section allowlists recorded in the
  request metadata.
- Attempt 2: JSON parsing passed, but the repair response returned
  `research_gaps` as strings and repeated the same section-scoped citation
  violations.

This is therefore frozen as:

- `RESEARCH_GAP_SCHEMA_MISMATCH_FIXED=true`
- `SECTION_CITATION_CONTRACT_STILL_FAILS_ON_FROZEN_REPLAY=true`
- `FINAL_LIVE_SMOKE_AUTHORIZED_BY_OFFLINE_REPLAY=false`
- `PRODUCTION_DEEP_RESEARCH_STATUS=EXPERIMENTAL_SCHEMA_RELIABILITY_BLOCKED`

No automatic citation replacement, JSON repair, or section citation broadening
was applied to make the frozen responses pass.

## Revised-protocol live smoke

The legacy frozen replay is now explicitly treated as diagnostic-only evidence.
The revised protocol was validated by a new production live smoke:

- `task_id=hotfix-deep-research-20260725220952`
- `request_attempt_count=1`
- `provider_completed_request_count=1`
- `usage_record_count=1`
- `total_tokens=9486`
- `estimated_cost_usd=0.00155736`
- `raw_response_replay=PASSED`
- `research_synthesis_schema=PASSED`
- `citation_global_allowlist=PASSED`
- `citation_section_allowlist=PASSED`
- `report_quality_gate=PASSED`

After this new-protocol smoke, production Deep Research is classified as
`AVAILABLE`, while `SEMANTIC_CLAIM_SUPPORT_AUDIT=NOT_FORMALLY_VALIDATED` and
`STRONG_GROUNDING_CLAIM_ALLOWED=false` remain in force.
