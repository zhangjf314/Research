# Post-merge release fix audit v1

## Starting state

- `starting_head`: `87ac5aa91d9238563bb51bbb82b9d9a9cf4baaaf`
- `release_branch_head`: `3920aa96c26218493d3e14f4d07aa95cd8cae4ac`
- `backup_branch`: `backup/main-merged-v1.0.0-portfolio`
- `fix_branch`: `release/post-merge-fix-v1.0.0-portfolio`
- `tag_created`: `false`
- `push_executed`: `false`

The local merge commit was preserved. No reset, tag, push, or remote release was performed.

## Baseline reproduction

The full baseline pytest command was started with project-local temp directories:

```powershell
$env:TEMP = (Resolve-Path .runtime\pytest-temp).Path
$env:TMP = $env:TEMP
.\.venv\Scripts\python.exe -m pytest -q --basetemp "$PWD\.runtime\pytest-base" -p no:cacheprovider
```

The command did not expose a Windows `PermissionError`; it timed out in the Codex tool after progressing through the suite. A targeted historical hash run reproduced a deterministic failure:

```text
RuntimeError: source hashes changed: cl-q001-b93fe64266ba6940d2a1
```

Root cause and fix are documented in `docs/historical-freeze-hash-stability-audit-v1.md`.

## Windows pytest temp fix

Added `scripts/run_release_tests.ps1`, which:

- creates `.runtime\pytest-temp` and `.runtime\pytest-release`;
- sets `TEMP` and `TMP` to the project-local temp directory;
- performs a write probe before running tests;
- runs pytest, Ruff, compileall, `git diff --check`, and `docker compose config --quiet`;
- exits non-zero on the first failing gate.

`.runtime/` is ignored in `.gitignore`.

## Merge comparison

`git rev-list --parents -n 1 HEAD` reported:

```text
87ac5aa91d9238563bb51bbb82b9d9a9cf4baaaf b657dcc2a9f7201b7674d6d4142f5817d818a639 3920aa96c26218493d3e14f4d07aa95cd8cae4ac
```

No unexpected diff was observed between the release branch head and the local merge commit.

## Publication artifacts

See `docs/post-merge-publication-artifact-audit-v1.md`.

## Final verification status

Final release verification was run with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_release_tests.ps1
```

Results:

- pytest: `599 passed, 1 warning`.
- pytest permission errors: `0`.
- historical freeze hash failures after fix: `0`.
- Ruff: `passed`.
- compileall: `passed`.
- `git diff --check`: `passed` with CRLF materialization warnings only.
- `docker compose config --quiet`: `passed`; Docker emitted a local config-file access warning, but returned exit code 0.

The full local log is retained at `artifacts/post-merge-release-tests-v1.txt` and is intentionally ignored because it contains local machine paths.

## Conclusion

`A. Post-merge release verification passed. Fix branch is ready to merge into main; tag/push still require user authorization.`

Tag creation and push remain unauthorized.
