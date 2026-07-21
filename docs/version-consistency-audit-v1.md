# Version Consistency Audit v1

Status: `PASSED`

## Observed values

- `pyproject.toml`: `0.9.0rc3`
- `paper_research.__version__`: `0.9.0rc3`
- Display version: `0.9.0-rc3`
- FastAPI OpenAPI version: `0.9.0rc3`
- `/api/v1/health`: `0.9.0rc3`
- `/api/v1/capabilities`: `0.9.0rc3`
- Docker OCI label source: Dockerfile `ARG APP_VERSION=0.9.0rc3`

The editable package metadata was refreshed with:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

No upgrade to `1.0.0-portfolio` was performed.

## Enforcement

`tests/test_version_consistency.py` verifies pyproject/runtime/OpenAPI/root
health/capabilities semantic consistency. Stage 13.39 adds a Dockerfile label
check so container metadata cannot silently drift from the RC version.
