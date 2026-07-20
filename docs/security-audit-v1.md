# Security Audit v1

Status: `PASSED_WITH_HISTORY_REVIEW_REQUIRED`

## What was checked

- Tracked text files were scanned for API-key-like values, authorization headers,
  cookie headers, private keys, and database URLs containing inline credentials.
- Current ignored/local-only artifacts were reviewed for publication risk.
- Git history was queried for broad secret-related strings without printing any
  credential value.

## Findings

- Actual tracked secret findings: `0`.
- Raw scan hits: `14`.
- Classification:
  - `.env.example` contains placeholder/local development configuration only.
  - `data/evaluation/evidence-corpus-v1.jsonl` produced a false positive inside
    paper text.
  - Remaining hits are docs/scripts/tests that mention authorization strings for
    safety checks; they are not credentials.
- `.env` is ignored and was not committed.
- Human review ZIP files remain ignored and local-only.

## Git history boundary

`git log -G` returned matching commits for secret-related words. These are not
automatically credential leaks; the repository intentionally contains security
tests and configuration-key names. A line-level history audit should be done
before any public release. No key contents were printed during this audit.

## Public demo handling

Do not publish raw provider responses, long trace files, private review ZIPs,
database dumps, Qdrant snapshots, or `.env` files. Publish only curated summaries,
hash manifests, and sanitized audit documents.
