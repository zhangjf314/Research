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
