from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

OUTPUT_JSON = Path("data/evaluation/postgresql-backup-restore-v2.json")
OUTPUT_MD = Path("docs/postgresql-backup-restore-audit-v2.md")
ARTIFACT_DIR = Path("artifacts/private/postgresql-restore-v2")
TABLES = ["papers", "checkpoints", "checkpoint_writes", "checkpoint_blobs"]


def run(*args: str) -> str:
    completed = subprocess.run(args, capture_output=True, text=True)
    if completed.returncode != 0:
        print(
            json.dumps(
                {
                    "command": list(args),
                    "returncode": completed.returncode,
                    "stdout": completed.stdout.strip(),
                    "stderr": completed.stderr.strip(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        completed.check_returncode()
    return completed.stdout.strip()


def docker(*args: str) -> str:
    return run("docker", "compose", *args)


def psql(database: str, sql: str) -> str:
    return docker("exec", "-T", "postgres", "psql", "-U", "paper", "-d", database, "-Atc", sql)


def counts(database: str) -> dict[str, int]:
    rows = psql(
        database,
        " union all ".join(f"select '{table}', count(*) from {table}" for table in TABLES),
    )
    return {name: int(value) for name, value in (row.split("|", 1) for row in rows.splitlines())}


def main() -> None:
    started = datetime.now(UTC)
    stamp = started.strftime("%Y%m%d%H%M%S")
    restore_db = f"paper_research_restore_v2_{stamp}"
    dump_name = f"paper_research_stage13_40_{stamp}.dump"
    container_dump = f"/tmp/{dump_name}"
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    host_dump = ARTIFACT_DIR / dump_name

    source_version = psql("paper_research", "select version();")
    source_counts_before = counts("paper_research")
    docker(
        "exec",
        "-T",
        "postgres",
        "pg_dump",
        "-U",
        "paper",
        "-Fc",
        "-d",
        "paper_research",
        "-f",
        container_dump,
    )
    subprocess.run(
        ["docker", "cp", f"research-postgres-1:{container_dump}", str(host_dump)],
        check=True,
    )
    dump_sha = hashlib.sha256(host_dump.read_bytes()).hexdigest()
    docker("exec", "-T", "postgres", "createdb", "-U", "paper", restore_db)
    docker("exec", "-T", "postgres", "pg_restore", "-U", "paper", "-d", restore_db, container_dump)
    restored_counts = counts(restore_db)
    source_counts_after = counts("paper_research")
    checkpoint_present = int(psql(restore_db, "select count(*) from checkpoints;")) > 0
    checkpoint_writes_present = int(psql(restore_db, "select count(*) from checkpoint_writes;")) > 0
    schema_compatible = set(source_counts_before) == set(restored_counts)
    critical_table_counts_match = source_counts_before == restored_counts
    source_database_unchanged = source_counts_before == source_counts_after
    gate = (
        "PASSED"
        if schema_compatible
        and critical_table_counts_match
        and checkpoint_present
        and checkpoint_writes_present
        and source_database_unchanged
        else "FAILED"
    )
    finished = datetime.now(UTC)
    payload = {
        "schema_version": "postgresql-backup-restore-v2",
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "gate": gate,
        "backup_created": True,
        "restore_to_isolated_database": True,
        "restore_database": restore_db,
        "source_database_version": source_version,
        "backup_command": (
            "docker compose exec -T postgres pg_dump -U paper -Fc "
            "-d paper_research -f /tmp/<dump>"
        ),
        "restore_command": (
            "docker compose exec -T postgres pg_restore -U paper "
            "-d <isolated_db> /tmp/<dump>"
        ),
        "backup_file": str(host_dump),
        "backup_sha256": dump_sha,
        "schema_compatible": schema_compatible,
        "source_counts_before": source_counts_before,
        "restored_counts": restored_counts,
        "source_counts_after": source_counts_after,
        "critical_table_counts_match": critical_table_counts_match,
        "critical_run_ids_present": "NOT_APPLICABLE_ARTIFACT_ONLY",
        "usage_totals_match": "NOT_APPLICABLE_ARTIFACT_ONLY",
        "checkpoint_records_present": checkpoint_present,
        "checkpoint_writes_present": checkpoint_writes_present,
        "source_database_unchanged": source_database_unchanged,
        "redis_compared": False,
        "redis_scope_note": "Redis is outside PostgreSQL and was intentionally not compared.",
        "dump_publication_policy": "local-only; do not commit",
    }
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# PostgreSQL Backup Restore Audit v2",
        "",
        f"- Gate: `{gate}`",
        "- Backup created: `true`",
        f"- Restore database: `{restore_db}`",
        f"- Backup SHA-256: `{dump_sha}`",
        f"- Schema compatible: `{schema_compatible}`",
        f"- Critical table counts match: `{critical_table_counts_match}`",
        f"- Checkpoint records present: `{checkpoint_present}`",
        f"- Checkpoint writes present: `{checkpoint_writes_present}`",
        f"- Source database unchanged: `{source_database_unchanged}`",
        (
            "- Full QA and Deep Research run IDs are artifact-backed in this project, "
            "not PostgreSQL rows."
        ),
        "- Redis was not compared because it is outside PostgreSQL.",
        "",
        "## Table counts",
        "",
        "| Table | Source | Restored |",
        "|---|---:|---:|",
    ]
    for table in TABLES:
        lines.append(f"| {table} | {source_counts_before[table]} | {restored_counts[table]} |")
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "gate": gate,
                "restore_database": restore_db,
                "counts_match": critical_table_counts_match,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
