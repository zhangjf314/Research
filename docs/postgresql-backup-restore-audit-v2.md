# PostgreSQL Backup Restore Audit v2

Status: `NOT_EXECUTED`

The Stage 10 backup/restore note remains historical evidence only. Stage 13.39
requires a fresh production restore verification against the current `0.9.0rc3`
runtime and final DeepSeek evidence. That fresh restore was not executed here.

Required remaining proof:

- Create a current PostgreSQL backup.
- Restore into an isolated test database or disposable container.
- Verify papers, task rows, trace rows, checkpoint rows, and checkpoint writes.
- Keep dump files local-only and out of Git.
