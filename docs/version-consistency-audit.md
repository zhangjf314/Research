# Version Consistency Audit

Generated on: 2026-07-18  
Branch: `eval/retrieval-recall-benchmark-v1`  
HEAD at audit start: `98058ceef882b317afa9b6f2086b9da9ffdac3d0`

## Result

Status: **PASS for rebuilt API container**

The source tree now uses `pyproject.toml` as the runtime version source for the local development tree and falls back to installed package metadata when the source tree is unavailable. This fixes the previous source-code mismatch where:

- package version: `0.9.0rc3`
- runtime `paper_research.__version__`: `0.9.0-rc1`

After Stage 13.40 hard gates passed, the local release-preparation tree now uses:

- internal package version: `1.0.0+portfolio`
- display version: `1.0.0-portfolio`

The internal package version uses the PEP 440 local-version form because Python
package metadata does not accept `1.0.0-portfolio` as a canonical distribution
version. User-facing docs, health/capabilities display fields, and the Docker OCI
label use `1.0.0-portfolio`.

The project is **not** upgraded to `1.0.0-portfolio`.

## Version source

Implementation:

- `src/paper_research/version.py`
- `src/paper_research/__init__.py`

Runtime behavior:

1. Read `[project].version` from source-tree `pyproject.toml` when available.
2. Fall back to `importlib.metadata.version("paper-research-agent")` when running from an installed package without the source tree.
3. Expose:
   - `paper_research.__version__`
   - `paper_research.__display_version__`
   - FastAPI/OpenAPI version
   - `/api/v1/health` version fields
   - `/api/v1/capabilities` version fields

## Known local environment caveat

The current virtual environment metadata still reports `0.9.0rc1` for:

```powershell
.\.venv\Scripts\python.exe -c "import importlib.metadata as m; print(m.version('paper-research-agent'))"
```

This is stale installed metadata and should be refreshed only by a normal package reinstall/build step. The source tree avoids using that stale metadata while `pyproject.toml` is present.

## Docker/OpenAPI impact

The API image was rebuilt after the Stage 13.40 release-preparation version
update, and the API container was force-recreated.

Runtime checks:

- `GET /api/v1/health`: `version=1.0.0+portfolio`, `display_version=1.0.0-portfolio`
- `GET /api/v1/capabilities`: `version=1.0.0+portfolio`, `display_version=1.0.0-portfolio`
- `docker compose ps`: API healthy after rebuild/recreate

## Tests

Added:

- `tests/test_version_consistency.py`

Coverage:

- package/source version equals runtime version
- root endpoint version
- OpenAPI version
- `/api/v1/health` version
- `/api/v1/capabilities` version

## Release decision

Version consistency is now verified for the local release-preparation tree.
This does not authorize merge, tag, push, or remote release; those actions still
require explicit user authorization.
