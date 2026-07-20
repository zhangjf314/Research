# Security Audit v1

Status: `PASSED`

## What was checked

- Tracked text files were scanned for API-key-like values, authorization headers,
  cookie headers, private keys, and database URLs containing inline credentials.
- Current ignored/local-only artifacts were reviewed for publication risk.
- Git history was queried for broad secret-related strings and every hit was
  reviewed without printing any credential value.

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

## Git history review

Stage 13.40 completed a line-level git-history review:

- Total hits: `36`
- Confirmed real secrets: `0`
- Unresolved hits: `0`
- Classification counts:
  - `DOCUMENTATION_EXAMPLE`: `10`
  - `EMPTY_VALUE`: `24`
  - `FALSE_POSITIVE`: `1`
  - `PLACEHOLDER`: `1`

Gate: `GIT_HISTORY_SECRET_GATE=PASSED`.

No key contents were printed or written during this audit. The review report is
[`docs/git-history-secret-review-v1.md`](git-history-secret-review-v1.md).

## Public demo handling

Do not publish raw provider responses, long trace files, private review ZIPs,
database dumps, Qdrant snapshots, or `.env` files. Publish only curated summaries,
hash manifests, and sanitized audit documents.
