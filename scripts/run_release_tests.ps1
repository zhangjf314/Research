$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root
$Runtime = Join-Path $Root ".runtime"
$TempDir = Join-Path $Runtime "pytest-temp"
$BaseTemp = Join-Path $Runtime "pytest-release"

New-Item -ItemType Directory -Force -Path $TempDir | Out-Null
New-Item -ItemType Directory -Force -Path $BaseTemp | Out-Null

$env:TEMP = (Resolve-Path $TempDir).Path
$env:TMP = $env:TEMP

$Probe = Join-Path $env:TEMP "write-test.txt"
"ok" | Set-Content -LiteralPath $Probe -Encoding UTF8
$ProbeValue = Get-Content -LiteralPath $Probe -Raw
Remove-Item -LiteralPath $Probe -Force
if ($ProbeValue.Trim() -ne "ok") {
    throw "Project pytest temp directory is not writable."
}

$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    $Python = "python"
}

& powershell -ExecutionPolicy Bypass `
    -File (Join-Path $Root "scripts\run_full_test_suite_sharded.ps1") `
    -PythonPath $Python `
    -TestsPath "tests" `
    -OutputDirectory (Join-Path $Runtime "full-pytest-shards-release")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python -m ruff check .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python -m compileall -q src scripts tests
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

git diff --check
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

docker compose config --quiet
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
