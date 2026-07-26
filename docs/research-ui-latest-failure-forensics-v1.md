# Research UI Latest Failure Forensics v1

Date: 2026-07-26

## Summary

The latest observed Deep Research UI failure was not a retrieval outage or a provider
availability failure. The graph reached synthesis, received provider-completed responses,
and failed because the generated structured payload violated the section-scoped citation
allowlist.

The UI also had a contract bug: a terminal failed graph was returned over HTTP 200 and
persisted with an empty Markdown report, so the browser presented the ambiguous message
`Research completed without a report` instead of surfacing the actual
`FAILED_PROVIDER_SCHEMA` status.

## Evidence

- task_id: `c88e94e6-a8b1-41d5-8c6f-b75cc90778db`
- HTTP endpoint: `POST /api/v1/research/deep`
- HTTP status: `200`
- graph status: `FAILED_PROVIDER_SCHEMA`
- stop reason: `research synthesis schema validation failed`
- nodes visited: `understand -> plan -> local_search -> assess -> synthesize_llm -> render_report -> validate_report_quality`
- evidence gaps: none
- global evidence count: 16
- request attempts: 2
- provider completed requests: 2
- usage records: 2
- input/output/total tokens: 9671 / 3582 / 13253
- usage source: `provider_reported`
- estimated cost: `0.0023569000000000003`
- active reserved tokens: 0
- report length: 0
- report quality: null

The report artifacts were stored in the Docker `app_data` volume under
`/app/data/reports/research/`, not in the host `data/reports/research/` directory.

## Replay result

Raw response replay was performed from:

`.runtime/research-synthesis-provider/c88e94e6-a8b1-41d5-8c6f-b75cc90778db`

Attempt 1:

- JSON parse: passed
- schema validation: failed
- failure type: `CITATION_NOT_ALLOWED_FOR_SECTION`
- offending evidence IDs: `E02`, `E07`, `E15`, `E16`

Attempt 2:

- JSON parse: passed
- schema validation: failed
- failure type: `CITATION_NOT_ALLOWED_FOR_SECTION`
- offending evidence ID: `E02`

## Root cause

The second full retry improved the output but still allowed an invalid section citation.
The deterministic root cause is therefore a section-scoped citation allowlist violation
inside the provider synthesis payload, plus a UI/API failure contract that hid the true
terminal failure from the user.

## Fix

- The API response now includes `succeeded`, `terminal`, `error_code`, and
  `report_available`.
- Failed terminal tasks no longer create blank Markdown reports.
- The UI now displays failed graph status, stop reason, task ID, attempts, token usage,
  and estimated cost instead of treating failures as completed reports.
- Provider attempt 2 now performs component-level repair for invalid sections only, with
  each repair prompt restricted to the invalid section evidence allowlist.

No new live LLM call was made while producing this forensic audit.
