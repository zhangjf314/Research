# Docker Access Audit

Generated on: 2026-07-18  
Branch: `eval/retrieval-recall-benchmark-v1`

## Result

Status: **PARTIAL**

The user's normal desktop PowerShell can fully access Docker Desktop through the `desktop-linux` context. The current Codex sandbox process still cannot read Docker Desktop context metadata or connect through its default Docker context, but project-level `docker compose` commands in this workspace can operate the stack.

This distinction matters: the existing containers are real, but this process cannot complete daemon-level `docker version`, `docker info`, `docker compose config`, rebuild, or container-env inspection.

## Commands and outcomes

| Command | Outcome |
|---|---|
| `whoami` | `laptop-4armhd5s\codexsandboxoffline` |
| `docker context ls` | failed: cannot read `C:\Users\ZHJF\.docker\contexts\meta` |
| `docker context show` | returned `default`, with config-file warning |
| `Get-ChildItem Env:DOCKER_HOST` | not present |
| `docker version` | client info printed, then permission denied connecting to `npipe:////./pipe/docker_engine` |
| `docker info` | client info printed, then permission denied connecting to Docker API |
| `docker compose ps` | succeeded earlier and listed API, Nginx, PostgreSQL, Qdrant, Redis running |
| `docker compose exec -T api ...` | failed with Docker API named-pipe permission denied |

## User-provided desktop PowerShell evidence

The user ran the required commands from `D:\Agents\Codex\research` in a normal PowerShell session:

- `docker context ls`: `desktop-linux` selected
- `docker context show`: `desktop-linux`
- `docker version`: client/server both `29.5.3`; Docker Desktop `4.78.0`
- `docker info`: server accessible; 5 containers running; Linux/amd64 Docker Desktop engine
- `docker compose config`: rendered the expected Production profile and provider variables
- `docker compose build --no-cache api`: completed and built `paper-research-agent-0.9.0rc3`
- `docker compose up -d --force-recreate api nginx`: initially failed because API was unhealthy
- Root cause was later confirmed from logs as optional numeric environment empty-string parsing.

The user-provided Compose config contained real secrets. They are intentionally not reproduced here.

## Post-fix Codex evidence

After fixing optional numeric environment parsing, Codex successfully executed:

- `docker compose build --no-cache api`
- `docker compose up -d --force-recreate api nginx`
- `docker compose ps`
- `docker compose logs --tail=80 api`

`docker compose ps` showed:

- API up and healthy
- Nginx up
- PostgreSQL healthy
- Qdrant up
- Redis healthy

API HTTP checks also passed via Nginx on localhost.

Later in this audit, after provider preflight, Codex successfully repeated:

- `GET http://localhost/api/v1/health`
- `GET http://localhost/api/v1/capabilities`

Those HTTP checks confirmed the API was still healthy and configured for `siliconflow/Qwen/Qwen3-8B`. In the same command batch, `docker compose ps` failed again with `Access is denied` and Docker named-pipe permission denied, confirming the remaining issue is Codex subprocess Docker CLI access rather than the HTTP-visible API runtime.

## Diagnosis

Observed issue is not simply “Docker Desktop down.” Evidence:

- `docker compose ps` previously listed running project containers.
- API endpoints on `localhost` respond.
- Qdrant and Redis ports respond.

The remaining Codex-only issue is a permissions/context mismatch for the Codex sandbox user:

- current process user: `codexsandboxoffline`
- Docker config path belongs to `C:\Users\ZHJF\.docker`
- named pipe access to `docker_engine` is denied

No system permission changes were attempted.

## Required user action

For final daemon-level audits, continue to use a normal Docker-authorized PowerShell session. Minimal verification:

```powershell
cd D:\Agents\Codex\research
whoami
docker context ls
docker context show
docker version
docker info
docker compose ps
```

If those pass, continue with:

```powershell
docker compose config
docker compose build --no-cache api
docker compose up -d --force-recreate api nginx
docker compose ps
```

Do not loosen Docker named-pipe permissions broadly just to satisfy this audit.

## Release decision

Docker Desktop access is proven in the user's PowerShell session and Compose operations are usable from Codex for this workspace. The Codex sandbox still cannot run raw `docker version/info` against `desktop-linux`, so final daemon-level evidence should continue to be captured from the user's Docker-authorized PowerShell.
