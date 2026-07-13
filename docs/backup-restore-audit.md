# Backup and Restore Audit

Date: 2026-07-13
Overall result: **PASS with deterministic-tie ordering note**

## PostgreSQL

Actual commands included `pg_dump -Fc`, `createdb`, and `pg_restore` into the isolated
database `paper_research_restore_stage10`. The dump is stored at
`artifacts/postgres-stage10.dump` (45,761 bytes).

Restored counts:

- papers: 34
- LangGraph checkpoints: 9
- checkpoint writes: 50

The source database was not destroyed; restore validation used a separate database.

## Qdrant

- Source collection: `papers_hash_v1__20260713104355`
- Snapshot size: 22,468,608 bytes
- Restored collection: `papers_hash_v1_restore_stage10`
- Source/restored point count: 2,062 / 2,062
- Vector size/distance: 384 / Cosine

For query `retrieval augmented generation`, the restored Top-5 contained exactly the
same IDs and scores. Positions 4 and 5 exchanged order because both scores were exactly
`0.57735026`; this is a tie-order difference, not a retrieval-content difference.
