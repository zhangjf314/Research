# ruff: noqa: E501,E701,E702
"""Classify the Stage 13 checkpoint worktree and scan commit candidates."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/evaluation"
DOCS = ROOT / "docs"
OUTPUT = DATA / "stage13-checkpoint-file-audit-v1.json"
OUTPUT_DOC = DOCS / "stage13-checkpoint-file-audit-v1.md"
SECRET_OUTPUT = DATA / "stage13-checkpoint-secret-scan-v1.json"


def git(*args: str) -> list[str]:
    result = subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True)
    return [line for line in result.stdout.splitlines() if line]


def category(path: str) -> str:
    if path == ".gitignore" or path.startswith("src/"):
        if any(token in path for token in ("prompts.py", "required_claim_output.py", "response_normalization.py", "citation_id_output.py", "citation_registry.py")):
            return "prompt_or_schema"
        return "production_code"
    if path.startswith("scripts/"):
        return "evaluation_code"
    if path.startswith("tests/"):
        return "test"
    if path.endswith(".bak") or ".bak" in path:
        return "backup"
    if path.startswith("artifacts/imports/") or "human-review-results" in path:
        return "temporary_import"
    if path.startswith("artifacts/"):
        return "audit_evidence"
    if path.startswith("data/evaluation/"):
        return "canonical_evaluation_data"
    if path.startswith("docs/"):
        return "canonical_report"
    return "uncertain"


def stage(path: str) -> str:
    if any(token in path for token in ("v3-1", "response-shape", "response-normalization", "stage13-5-schema", "checkpoint")):
        return "13.6/checkpoint"
    if "v3" in path or "stage13_4" in path or "stage13-4" in path or "required_claim" in path:
        return "13.4-13.5"
    if "v2" in path or "stage13_3" in path or "stage13-3" in path:
        return "13.3"
    if "gap" in path or "pilot" in path or "stage13_1" in path or "stage13-1" in path:
        return "13.1"
    return "13.2/common"


def recommendation(path: str, cat: str) -> tuple[str, str, bool]:
    local = (
        cat in {"backup", "temporary_import"}
        or path.startswith("artifacts/")
        or "/runs/" in path
        or path.endswith(".zip")
    )
    if local:
        return "retain_local_only", "runtime, backup, or duplicate external-review evidence retained locally and ignored", True
    if cat == "uncertain":
        return "manual_review", "path does not match a reviewed Stage 13 category", False
    return "commit", "canonical Stage 13 code, test, data, or report", False


def main() -> None:
    modified = {line[3:].replace("\\", "/"): line[:2].strip() or "M" for line in git("status", "--porcelain=v1", "--untracked-files=all")}
    ignored = {
        path
        for path in git("ls-files", "--others", "--ignored", "--exclude-standard")
        if path.startswith(("artifacts/", "data/evaluation/"))
        or path.endswith(".bak")
    }
    paths = sorted(set(modified) | ignored | {"scripts/audit_stage13_checkpoint_v1.py", "data/evaluation/stage13-checkpoint-file-audit-v1.json", "data/evaluation/stage13-checkpoint-secret-scan-v1.json", "docs/stage13-checkpoint-file-audit-v1.md"})
    env_key = ""
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("LLM_API_KEY="):
                env_key = line.split("=", 1)[1].strip()
    hashes: dict[str, list[str]] = defaultdict(list)
    raw_records = []
    secret_hits = []
    absolute_hits = []
    for rel in paths:
        path = ROOT / rel
        if not path.is_file():
            continue
        body = path.read_bytes()
        digest = hashlib.sha256(body).hexdigest()
        hashes[digest].append(rel)
        text = body.decode("utf-8", errors="ignore")
        actual_secret = bool(env_key and len(env_key) > 8 and env_key in text)
        secret_patterns = []
        for name, pattern in {"sk_prefix": r"\bsk-[A-Za-z0-9_-]{12,}", "bearer_value": r"Bearer\s+[A-Za-z0-9._-]{12,}", "authorization_value": r'Authorization["\s:]+Bearer\s+[A-Za-z0-9._-]{12,}', "cookie_value": r"Cookie[" + "'\"\\s:]+[^\r\n]{12,}"}.items():
            if re.search(pattern, text, re.IGNORECASE): secret_patterns.append(name)
        if actual_secret or secret_patterns:
            secret_hits.append({"path": rel, "actual_configured_key": actual_secret, "patterns": secret_patterns, "classification": "actual_secret" if actual_secret else "requires_manual_context_review"})
        local_windows_path = "\\".join(("D:", "Agents", "Codex", "research"))
        json_windows_path = "\\\\".join(("D:", "Agents", "Codex", "research"))
        local_posix_path = "/".join(("D:", "Agents", "Codex", "research"))
        if local_windows_path in text or json_windows_path in text or local_posix_path in text:
            absolute_hits.append(rel)
        cat = category(rel)
        rec, reason, retained = recommendation(rel, cat)
        raw_records.append({"path": rel, "git_status": modified.get(rel, "ignored" if rel in ignored else "generated"), "size": len(body), "category": cat, "stage": stage(rel), "commit_recommendation": rec, "reason": reason, "duplicate_of": None, "contains_sensitive_data": actual_secret, "retained_outside_git": retained, "sha256": digest, "contains_local_absolute_path": rel in absolute_hits})
    for record in raw_records:
        duplicates = hashes[record["sha256"]]
        record["duplicate_of"] = next((item for item in duplicates if item != record["path"]), None)
    counts = Counter(record["category"] for record in raw_records)
    recommendations = Counter(record["commit_recommendation"] for record in raw_records)
    duplicates = [items for items in hashes.values() if len(items) > 1]
    large = [{"path": row["path"], "size": row["size"], "recommendation": row["commit_recommendation"]} for row in raw_records if row["size"] > 1024 * 1024]
    payload = {"schema_version": "stage13-checkpoint-file-audit-v1", "initial_commit": "09fe3a17e1bbb6fd1694b55fe11a1618d0eae864", "branch": "eval/evidence-gap-adjudication-v1", "file_count": len(raw_records), "category_counts": dict(counts), "recommendation_counts": dict(recommendations), "uncertain_count": counts.get("uncertain", 0), "large_files": large, "duplicate_groups": duplicates, "absolute_path_hits": sorted(absolute_hits), "records": raw_records}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True); DOCS.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    secret = {"schema_version": "stage13-checkpoint-secret-scan-v1", "candidate_file_count": sum(row["commit_recommendation"] == "commit" for row in raw_records), "actual_secret_hits": [row for row in secret_hits if row["classification"] == "actual_secret"], "context_review_hits": [row for row in secret_hits if row["classification"] != "actual_secret"], "absolute_path_hits": sorted(absolute_hits), "api_key_field_names_are_not_secret_values": True, "safe_to_commit": not any(row["classification"] == "actual_secret" for row in secret_hits)}
    SECRET_OUTPUT.write_text(json.dumps(secret, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_DOC.write_text(f"# Stage 13 Checkpoint File Audit\n\n- Initial files audited: {len(raw_records)}\n- Commit candidates: {recommendations.get('commit', 0)}\n- Retain local only: {recommendations.get('retain_local_only', 0)}\n- Uncertain: {counts.get('uncertain', 0)}\n- Files over 1 MiB: {len(large)}\n- Duplicate hash groups: {len(duplicates)}\n- Actual secret hits: {len(secret['actual_secret_hits'])}\n- Absolute-path hits: {len(absolute_hits)}\n\nLocal-only files remain in place under explicit `.gitignore` rules; nothing was deleted or archived.\n", encoding="utf-8")
    print(json.dumps({"files": len(raw_records), "categories": counts, "recommendations": recommendations, "large": len(large), "duplicates": len(duplicates), "actual_secret_hits": len(secret["actual_secret_hits"]), "absolute_path_hits": len(absolute_hits)}))
    if counts.get("uncertain") or secret["actual_secret_hits"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
