# Stage 13.7 Hash Root Cause

- Target: `data/evaluation/evidence-qa-dev-v1.json`
- Historical raw SHA-256: `23126a87deb978216fe56bc8518e25d99fb8c9b2e461640ebe727c80ea73d170`
- Current LF raw SHA-256: `c2df9e5aee765159ecaf03956923d2c652758a95086be22ac878b24eef2109bd`
- CRLF reconstruction SHA-256: `23126a87deb978216fe56bc8518e25d99fb8c9b2e461640ebe727c80ea73d170`
- Canonical JSON v1 SHA-256: `f1e8d8c3a5ab3a922a08cdef918c3ccc4b85d2465491dab72c3d3b1f5f4f7357`
- Root cause: the historical review hashed CRLF bytes while the checkout uses LF. Parsed JSON is identical.
- The mismatch is therefore fully explained by newline representation; no semantic mismatch was accepted.
