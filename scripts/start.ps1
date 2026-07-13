param(
    [switch]$Build,
    [switch]$SkipMigrations
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example"
}

$composeArgs = @("compose", "up", "-d")
if ($Build) {
    $composeArgs += "--build"
}

docker @composeArgs

if (-not $SkipMigrations) {
    docker compose exec -T api python -m alembic upgrade head
}

Write-Host "PaperResearch Agent: http://localhost"
Write-Host "API documentation: http://localhost/docs"
