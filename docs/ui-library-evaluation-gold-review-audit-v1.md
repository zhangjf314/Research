# UI Library / Evaluation / Gold Review Audit v1

## Scope

Post-release hotfix for Library, PDF import, Evaluation report visibility, Gold
Review loading behavior, and audit fixture cleanup.

No QA prompt, Retrieval algorithm, Gold content, formal evaluation result, real
LLM, Embedding API, Reranker, or Deep Research run is part of this audit.

## Initial runtime observations

- Docker services were running: API, Nginx, PostgreSQL, Qdrant, and Redis.
- API `/health` responded healthy but the running container still reported
  runtime version `1.0.0+portfolio`; this required an API container rebuild after
  the hotfix.
- `/api/v1/papers?limit=100` returned 42 papers.
- The first records included OCR audit fixtures with `source_type=upload`:
  - `fully-scanned-20260720152244`
  - `mixed-native-scanned-20260720152244`
  - `text-native-20260720152244`
  - older `fully-scanned-*`, `mixed-native-scanned-*`, and `text-native-*`
    records.

## Confirmed root causes

1. OCR audit scripts uploaded synthetic PDFs through the normal user upload API
   and did not clean DB/Qdrant/raw/parsed artifacts.
2. `/api/v1/papers` and the Library UI did not exclude `audit_fixture` records.
3. Evaluation UI was hard-coded to stale report paths and displayed missing
   reports as empty/generated placeholders.
4. Gold Review used CWD-relative paths and the UI did not handle failed or empty
   loads, so it could remain stuck at `Loading...`.
5. Manual uploads only persisted title/authors from parsing; year and other
   metadata fields had no parser-level extraction path.

## Implemented safeguards

- Added `PaperCleanupService` with dry-run support and ordered DB/Qdrant/raw
  PDF/parsed artifact cleanup.
- Added Qdrant `delete_by_paper_id` and `count_by_paper_id`.
- Added `source_type=audit_fixture` support to uploads while preserving default
  `source_type=upload`.
- `/api/v1/papers` defaults to excluding audit fixtures and supports
  `include_fixtures`, source, title, missing metadata, and not-indexed filters.
- Added metadata `PATCH /api/v1/papers/{paper_id}`.
- Added a public-safe evaluation report catalog and report detail route.
- Updated Gold Review path resolution and evidence warnings.

## Safety boundaries

- Fixture cleanup only auto-applies to deterministic OCR fixture names or records
  proven by audit artifacts.
- Real user papers are not removed merely because their title mentions scanned
  content.
- Gold records are not modified by this hotfix.
- Raw provider responses, user PDFs, and database dumps are not added to Git.

## Final Docker verification

- API image rebuilt and API/Nginx containers force-recreated.
- `/api/v1/health` returned `version=1.0.1+portfolio` and
  `display_version=1.0.1-portfolio`.
- `/api/v1/capabilities` showed production DeepSeek configuration present and
  `template_fallback=false`; no model call was made for this hotfix.
- UI routes returned HTTP 200:
  - `/api/v1/ui`
  - `/api/v1/ui/library`
  - `/api/v1/ui/search`
  - `/api/v1/ui/evaluation`
  - `/api/v1/ui/gold-review`
- Evaluation report detail route returned HTTP 200 for
  `/api/v1/ui/evaluation/portfolio-release-audit-v1`.
- Gold Review API returned `total=50`; first records `q001` and `q002` loaded
  successfully after mounting `data/evaluation` into the API container.

## Fixture cleanup result

- Cleanup dry-run identified exactly 9 `CONFIRMED_AUDIT_FIXTURE` records and 0
  formal evaluation references.
- Cleanup apply removed those 9 records.
- Library count changed from 42 to 33.
- Post-cleanup fixture-like title count: 0.
- Qdrant point count for the 9 old fixture records was 0 in both
  `papers_hash_v1` and `papers_production_v1`.
- Raw PDFs, parsed directories, and database records were deleted for all 9
  confirmed fixtures.
- The cleanup evidence is stored in
  `data/evaluation/audit-fixture-cleanup-v1.json` and
  `docs/audit-fixture-cleanup-v1.md`.

## Human Gold status

- `gold-set-v1.jsonl` records: 50.
- `review_status=approved`: 50.
- `answerable=true`: 48.
- `answerable=false`: 2.
- No Gold content, reviewer, or reviewed timestamp was modified.

## Validation

- Targeted UI/import/cleanup tests: 10 passed.
- Full release test entrypoint: 656 passed, 1 warning.
- Ruff: passed.
- Compileall: passed.
- `git diff --check`: passed; only pre-existing CRLF warnings were emitted.
- `docker compose config --quiet`: passed.
