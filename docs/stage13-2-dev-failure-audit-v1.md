# Stage 13.2 Dev Failure Audit v1

Historical Stage 13.2 runs and their 60,000 conservative token reservations remain unchanged.

## q001

The Provider response returned, but neither raw JSON nor usage was durably recorded before the local `claim_text`/`text` adapter mismatch. The original raw response therefore cannot be replayed. Deterministic fixtures for `claim_text`, legacy `text`, and equal dual fields pass; missing or conflicting fields fail closed. The new response envelope settles usage and persists raw response before parsing.

## q008

The request ended in `ReadTimeout` after 120 seconds with unknown Provider completion. The old client used one aggregate timeout and had no pre-batch DNS/TCP/TLS/models probe. A one-shot health checker is implemented; it is not executed live in Phase A and never retries a QA request.

## q019

The strict validator correctly rejected a page/block mismatch. The exact model JSON was not persisted, so the emitted page cannot be reconstructed. The historical context did contain complete allowed triples and block-page mappings. The citation-id-v2 fixture rejects free-form triples and unknown IDs; valid IDs resolve deterministically and are triple-validated again.

## Status

`WAITING_FOR_DEV_CITATION_AUDIT`. Dev v2, Full QA, and Deep Research were not run.
