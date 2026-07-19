# Version Consistency Audit

Generated on: 2026-07-18  
Branch: `eval/retrieval-recall-benchmark-v1`  
HEAD at audit start: `98058ceef882b317afa9b6f2086b9da9ffdac3d0`

## Result

Status: **PASS for rebuilt API container**

The source tree now uses `pyproject.toml` as the runtime version source for the local development tree and falls back to installed package metadata when the source tree is unavailable. This fixes the previous source-code mismatch where:

- package version: `0.9.0rc3`
- runtime `paper_research.__version__`: `0.9.0-rc1`

The current source-tree runtime version is now expected to be:

- internal version: `0.9.0rc3`
- display version: `0.9.0-rc3`

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

The API image was rebuilt after the version-source fix. The Docker build produced `paper-research-agent-0.9.0rc3`, and the API container was force-recreated successfully after the optional-numeric-env parsing fix.

Runtime checks:

- `GET /api/v1/health`: `version=0.9.0rc3`, `display_version=0.9.0-rc3`
- `GET /api/v1/capabilities`: `version=0.9.0rc3`, `display_version=0.9.0-rc3`
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

Version consistency is now verified for the rebuilt API container. This does not authorize `1.0.0-portfolio`; the version remains RC (`0.9.0rc3` / `0.9.0-rc3`).
