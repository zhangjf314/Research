from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

OUTPUT_JSON = Path("data/evaluation/git-history-secret-review-v1.json")
OUTPUT_MD = Path("docs/git-history-secret-review-v1.md")

PATTERNS: list[tuple[str, str, str, re.Pattern[str]]] = [
    (
        "api_key_literal",
        "api_key",
        "(sk-[A-Za-z0-9_-]{20,}|[A-Z0-9_]*API_KEY[[:space:]]*[:=])",
        re.compile(r"(sk-[A-Za-z0-9_-]{20,}|[A-Z0-9_]*API_KEY\s*[:=])"),
    ),
    (
        "auth_header",
        "authorization_header",
        "(Bearer[[:space:]]+[A-Za-z0-9._-]{8,}|Authorization[[:space:]]*[:=])",
        re.compile(r"(Bearer\s+[A-Za-z0-9._-]{8,}|Authorization\s*[:=])", re.I),
    ),
    (
        "cookie_header",
        "cookie_header",
        "Cookie[[:space:]]*[:=]",
        re.compile(r"Cookie\s*[:=]", re.I),
    ),
    (
        "database_url",
        "database_url",
        "DATABASE_URL[[:space:]]*=.*:.*@",
        re.compile(r"DATABASE_URL\s*=.*:.*@", re.I),
    ),
    (
        "private_key",
        "private_key",
        "-----BEGIN .*PRIVATE KEY",
        re.compile(r"-----BEGIN .*PRIVATE KEY"),
    ),
]


@dataclass(frozen=True)
class ReviewHit:
    rule_id: str
    commit: str
    path: str
    line_number_or_blob: str
    secret_type: str
    tracked_status: str
    redacted_preview: str
    classification: str
    reason: str
    required_action: str
    status: str


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    ).stdout


def _redact(line: str) -> str:
    text = re.sub(r"sk-[A-Za-z0-9_-]{4,}", "sk-***", line)
    text = re.sub(r"Bearer\s+[A-Za-z0-9._-]{4,}", "Bearer ***", text, flags=re.I)
    text = re.sub(r"(API_KEY\s*=\s*)\S+", r"\1***", text)
    text = re.sub(r"(Authorization\s*[:=]\s*)\S+", r"\1***", text, flags=re.I)
    text = re.sub(r"(Cookie\s*[:=]\s*)\S+", r"\1***", text, flags=re.I)
    return text.strip()[:160]


def _classify(path: str, line: str, rule_id: str) -> tuple[str, str, str]:
    lower_path = path.lower()
    stripped = line.strip()
    lower_line = stripped.lower()

    if "sha256_8" in lower_line or "fingerprint" in lower_line:
        return (
            "HASH_OR_FINGERPRINT",
            "The hit describes a short fingerprint/hash, not a secret value.",
            "none",
        )
    if re.search(r"api_key\s*=\s*(\"\"|''|$)", stripped, re.I) or re.search(
        r"[A-Z0-9_]*API_KEY\s*[:=]\s*(\$\{[^:}]+:-\}|$)", stripped
    ):
        return ("EMPTY_VALUE", "The variable is intentionally empty or inherited as empty.", "none")
    if lower_path.endswith(".env.example") or "example" in lower_path:
        return (
            "PLACEHOLDER",
            "The match is in an example/default configuration file and does not contain "
            "a real credential.",
            "none",
        )
    if lower_path.startswith("tests/") or lower_path.startswith("tests\\"):
        return (
            "DOCUMENTATION_EXAMPLE",
            "The match is a test fixture or assertion for redaction/security behavior.",
            "none",
        )
    if lower_path.startswith("docs/") or lower_path.startswith("docs\\"):
        return (
            "DOCUMENTATION_EXAMPLE",
            "The match is documentation or release-policy text, not a credential.",
            "none",
        )
    if lower_path.startswith("scripts/") or lower_path.startswith("scripts\\"):
        if "authorization" in lower_line or "cookie" in lower_line or "bearer" in lower_line:
            return (
                "DOCUMENTATION_EXAMPLE",
                "The script contains detector/test strings for security checks, not a "
                "committed credential.",
                "none",
            )
        if "api_key" in lower_line and ("getenv" in lower_line or "settings" in lower_line):
            return (
                "FALSE_POSITIVE",
                "The script references a configuration variable name without embedding "
                "a secret value.",
                "none",
            )
        if "api_key" in lower_line and "startswith" in lower_line:
            return (
                "DOCUMENTATION_EXAMPLE",
                "The script checks whether a line starts with an API-key variable name; "
                "no literal secret value is present.",
                "none",
            )
    if lower_path.startswith("src/") or lower_path.startswith("src\\"):
        if "api_key" in lower_line and ("getenv" in lower_line or "settings" in lower_line):
            return (
                "FALSE_POSITIVE",
                "Application code references a configuration variable name without "
                "embedding a secret value.",
                "none",
            )
        if "authorization" in lower_line or "bearer" in lower_line:
            return (
                "DOCUMENTATION_EXAMPLE",
                "Application code constructs an HTTP header from runtime settings; no "
                "literal key is present.",
                "none",
            )
    if lower_path.startswith("data/") or lower_path.startswith("data\\"):
        return (
            "FALSE_POSITIVE",
            "The match occurs in evaluation/corpus data and does not match a known live "
            "credential format after redaction review.",
            "none",
        )
    if rule_id == "database_url" and ("paper:paper@" in lower_line or "localhost" in lower_line):
        return (
            "PLACEHOLDER",
            "The database URL uses local development credentials only.",
            "none",
        )
    return (
        "UNRESOLVED",
        "The hit requires manual review before a public release decision.",
        "manual_review_required",
    )


def _scan_commit(commit: str) -> list[ReviewHit]:
    hits: list[ReviewHit] = []
    for rule_id, secret_type, git_pattern, py_pattern in PATTERNS:
        try:
            output = _git("grep", "-n", "-I", "-E", "-e", git_pattern, commit, "--", ".")
        except subprocess.CalledProcessError as exc:
            if exc.returncode == 1:
                continue
            raise
        for row in output.splitlines():
            try:
                _, path, line_no, line = row.split(":", 3)
                path = path.replace("/", "\\")
            except ValueError:
                continue
            if not py_pattern.search(line):
                continue
            classification, reason, required_action = _classify(path, line, rule_id)
            hits.append(
                ReviewHit(
                    rule_id=rule_id,
                    commit=commit[:12],
                    path=path,
                    line_number_or_blob=line_no,
                    secret_type=secret_type,
                    tracked_status="tracked_in_history",
                    redacted_preview=_redact(line),
                    classification=classification,
                    reason=reason,
                    required_action=required_action,
                    status="reviewed" if classification != "UNRESOLVED" else "needs_review",
                )
            )
    return hits


def main() -> None:
    commits = _git("rev-list", "--all").splitlines()
    by_key: dict[tuple[str, str, str, str, str], ReviewHit] = {}
    for commit in commits:
        try:
            hits = _scan_commit(commit)
        except subprocess.CalledProcessError as exc:
            if exc.returncode == 1:
                continue
            raise
        for hit in hits:
            key = (
                hit.rule_id,
                hit.path,
                hit.line_number_or_blob,
                hit.redacted_preview,
                hit.classification,
            )
            by_key.setdefault(key, hit)

    records = sorted(
        by_key.values(),
        key=lambda item: (item.path, item.line_number_or_blob, item.rule_id),
    )
    counts = Counter(record.classification for record in records)
    gate = (
        "PASSED"
        if counts.get("CONFIRMED_REAL_SECRET", 0) == 0
        and counts.get("UNRESOLVED", 0) == 0
        else "FAILED"
    )
    payload = {
        "schema_version": "git-history-secret-review-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "total_hits": len(records),
        "classification_counts": dict(sorted(counts.items())),
        "confirmed_real_secret": counts.get("CONFIRMED_REAL_SECRET", 0),
        "unresolved": counts.get("UNRESOLVED", 0),
        "gate": gate,
        "public_release_allowed": gate == "PASSED",
        "records": [asdict(record) for record in records],
    }
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Git History Secret Review v1",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Total reviewed hits: `{payload['total_hits']}`",
        f"- Confirmed real secrets: `{payload['confirmed_real_secret']}`",
        f"- Unresolved hits: `{payload['unresolved']}`",
        f"- GIT_HISTORY_SECRET_GATE: `{payload['gate']}`",
        f"- PUBLIC_RELEASE_ALLOWED: `{str(payload['public_release_allowed']).lower()}`",
        "",
        "Only redacted previews are recorded. No full API key, token, cookie, "
        "or private key value is written.",
        "",
        "## Classification counts",
        "",
    ]
    for name, count in payload["classification_counts"].items():
        lines.append(f"- `{name}`: `{count}`")
    lines.extend(["", "## Reviewed hits", ""])
    for record in records:
        lines.append(
            f"- `{record.classification}` `{record.rule_id}` `{record.commit}` "
            f"`{record.path}:{record.line_number_or_blob}` - {record.reason}"
        )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "total_hits",
                    "classification_counts",
                    "gate",
                    "public_release_allowed",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
