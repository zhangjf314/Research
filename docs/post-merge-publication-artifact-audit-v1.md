# Post-merge publication artifact audit v1

This audit checks the local post-merge release tree for large tracked artifacts, raw traces, local-only review packages, and secret-like content.

## Summary

- Tracked `artifacts/` files: 22.
- Git object store size: 15.79 MiB.
- Confirmed real secrets: 0.
- Unresolved secret hits: 0.
- Local absolute path hits in tracked files: 0.
- Gate: `PASSED_WITH_SIZE_LIMITATIONS`.

## Largest tracked files

| Path | Size | Classification |
| --- | ---: | --- |
| `data/evaluation/evidence-corpus-v1.jsonl` | 25.3 MB | `PUBLIC_SAFE_BUT_LARGE` |
| `data/evaluation/retrieval-context-optimization-v1.json` | 21.4 MB | `PUBLIC_SAFE_BUT_LARGE` |
| `data/ocr-audit-v1/mixed-native-scanned.pdf` | 13.5 MB | `PUBLIC_SAFE_BUT_LARGE` |
| `data/ocr-audit-v1/fully-scanned.pdf` | 13.5 MB | `PUBLIC_SAFE_BUT_LARGE` |
| `data/evaluation/reranker-ablation-v1.json` | 4.7 MB | `PUBLIC_SAFE_BUT_LARGE` |

The large files are retained because they are part of the reproducible Portfolio evidence. They are not raw private run directories, database dumps, Qdrant snapshots, or local human-review ZIP packages.

## Secret scan

Only file-level counts were recorded to avoid printing secrets. The matches are code paths, test fixtures, configuration placeholders, and existing redacted secret-review reports.

- `Authorization`: 36 file hits.
- `Bearer `: 25 file hits.
- `LLM_API_KEY`: 13 file hits.
- `postgresql://`: 2 file hits.
- `postgresql+psycopg://`: 5 file hits.
- `Cookie`: 4 file hits.
- Confirmed real secret values: 0.

The existing tracked `data/evaluation/git-history-secret-review-v1.json` reports `confirmed_real_secret=0`, `unresolved=0`, and `gate=PASSED`.

## Local-only files

These local review packages are intentionally ignored and were not added:

- `artifacts/stage13-9-human-citation-review-results.zip`
- `artifacts/stage13-10-human-claim-gold-review-results.zip`

No `git rm --cached` action was required in this pass.
