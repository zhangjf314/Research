# Quickstart

This guide starts the local Docker Compose stack for PaperResearch.

## Requirements

- Windows PowerShell or a compatible shell
- Docker Desktop with Linux containers
- DeepSeek-compatible LLM credentials
- Jina embedding credentials

## Start

```powershell
git clone https://github.com/zhangjf314/Research.git
cd Research

Copy-Item .env.example .env
# Fill provider keys in .env. Never commit .env.

docker compose up -d --build
docker compose ps

Invoke-RestMethod http://localhost/api/v1/health
Invoke-RestMethod http://localhost/api/v1/capabilities
```

## Local endpoints

- UI: <http://localhost/api/v1/ui>
- OpenAPI: <http://localhost/docs>
- Health: <http://localhost/api/v1/health>
- Qdrant: <http://localhost:6333>

Default Compose credentials are for local development only.

## UI and runtime checks

UI page GETs (`/api/v1/ui` and `/api/v1/ui/*`) do not consume Search, Upload, or
Deep Research rate-limit buckets. Business APIs are rate-limited separately:
read APIs, search, upload, metadata enrichment, and Deep Research. A 429 response
includes `Retry-After`, `error_code=RATE_LIMITED`, and `request_id`.

Check runtime capabilities before a demo:

```powershell
Invoke-RestMethod http://localhost/api/v1/capabilities |
  ConvertTo-Json -Depth 20
```

For Production Deep Research, confirm `capabilities.deep_research` reports the
expected provider/model, `response_format=json_object`, and
`template_fallback=false`.

Run the executable UI JavaScript gate:

```powershell
.\.venv\Scripts\python.exe scripts\check_ui_javascript_syntax_v1.py
```

Optional real browser smoke requires Playwright. If Playwright is unavailable,
the smoke script reports `BLOCKED_NOT_FAKED` instead of pretending the browser
test passed.

```powershell
.\.venv\Scripts\python.exe scripts\run_ui_browser_smoke.py
```

## Import a local PDF

Use the browser UI:

```text
Library -> choose PDF -> Upload PDF -> optional auto-index
```

Equivalent API flow:

```text
POST /api/v1/papers/upload
POST /api/v1/papers/{paper_id}/index
```

The upload form sends the PDF file only; it must not send the full local file
path.

## Import an external paper

Use the browser UI:

```text
Search -> search arXiv/Semantic Scholar -> Import PDF -> optional Index
```

Equivalent API flow:

```text
POST /api/v1/search/papers
POST /api/v1/search/import
```

The app intentionally does not provide arbitrary URL download. External imports
must come from configured search providers and candidates with a validated
downloadable PDF URL.
