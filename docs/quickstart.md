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
