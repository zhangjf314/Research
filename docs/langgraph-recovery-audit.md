# LangGraph Recovery Audit

Date: 2026-07-13
Result: **PASS for pause → API replacement → resume**

## Configuration

- Compose provider: `postgres`
- Database: PostgreSQL 16 (`paper_research`)
- Test thread: `stage10-recovery-001`
- Pause point: after `plan`

## Actual procedure and evidence

1. `POST /api/v1/research/deep` with a fixed `task_id` and
   `pause_after_node=plan` returned history `understand, plan`.
2. PostgreSQL contained `checkpoint_blobs`, `checkpoint_migrations`,
   `checkpoint_writes`, and `checkpoints`; the thread had 4 checkpoints.
3. The API image was rebuilt and the API container replaced.
4. `POST /api/v1/research/deep/stage10-recovery-001/resume` completed using the same
   thread ID.
5. Final history was `understand, plan, local_search, assess, synthesize, report,
   validate`; committed `understand` and `plan` were not repeated.
6. Final status was `COMPLETED`, stop reason `research_complete`, and the thread had
   9 checkpoints.

The import path also retains database/file-hash idempotency and a Redis import lock.
This run did not simulate a process kill in the middle of an uncommitted external import;
exactly-once semantics for arbitrary mid-node crashes are therefore not claimed.
