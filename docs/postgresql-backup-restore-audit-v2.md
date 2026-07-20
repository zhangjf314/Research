# PostgreSQL Backup Restore Audit v2

- Gate: `PASSED`
- Backup created: `true`
- Restore database: `paper_research_restore_v2_20260720150633`
- Backup SHA-256: `03bb409ac975d24b662b0bb0744b4b5b876dd1bc7b01ff99b29400dba57b8d38`
- Schema compatible: `True`
- Critical table counts match: `True`
- Checkpoint records present: `True`
- Checkpoint writes present: `True`
- Source database unchanged: `True`
- Full QA and Deep Research run IDs are artifact-backed in this project, not PostgreSQL rows.
- Redis was not compared because it is outside PostgreSQL.

## Table counts

| Table | Source | Restored |
|---|---:|---:|
| papers | 36 | 36 |
| checkpoints | 36 | 36 |
| checkpoint_writes | 200 | 200 |
| checkpoint_blobs | 100 | 100 |
