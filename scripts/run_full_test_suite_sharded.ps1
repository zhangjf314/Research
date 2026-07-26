param(
    [string]$PythonPath = ".\.venv\Scripts\python.exe",
    [string]$TestsPath = "tests",
    [string]$OutputDirectory = ".runtime\full-pytest-shards-v1",
    [switch]$ContinueOnFailure = $true,
    [switch]$Resume,
    [int]$PerShardTimeoutSeconds = 0
)

$ErrorActionPreference = "Stop"

function New-Directory([string]$Path) {
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

function ConvertTo-RepoRelativePath([string]$Path) {
    $full = [System.IO.Path]::GetFullPath($Path)
    $root = [System.IO.Path]::GetFullPath((Get-Location).Path)
    if ($full.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $full.Substring($root.Length).TrimStart('\', '/') -replace '\\', '/'
    }
    return $Path -replace '\\', '/'
}

function Invoke-CapturedProcess(
    [string]$FilePath,
    [string[]]$Arguments,
    [string]$StdoutPath,
    [string]$StderrPath,
    [int]$TimeoutSeconds = 0
) {
    $startedAt = Get-Date
    $timedOut = $false
    & $FilePath @Arguments 1> $StdoutPath 2> $StderrPath
    $exitCode = $LASTEXITCODE

    $endedAt = Get-Date
    if (-not (Test-Path -LiteralPath $StdoutPath)) {
        [System.IO.File]::WriteAllText($StdoutPath, "", [System.Text.UTF8Encoding]::new($false))
    }
    if (-not (Test-Path -LiteralPath $StderrPath)) {
        [System.IO.File]::WriteAllText($StderrPath, "", [System.Text.UTF8Encoding]::new($false))
    }

    return [ordered]@{
        exit_code = if ($timedOut) { 124 } else { $exitCode }
        timed_out = $timedOut
        duration_seconds = [Math]::Round(($endedAt - $startedAt).TotalSeconds, 3)
    }
}

function Read-JUnitSummary([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        return [ordered]@{
            tests = 0; failures = 0; errors = 1; skipped = 0
            passed = 0; xfailed = 0; xpassed = 0
        }
    }
    [xml]$xml = Get-Content -LiteralPath $Path -Raw
    $suites = @()
    if ($xml.testsuites) {
        $suites = @($xml.testsuites.testsuite)
    } elseif ($xml.testsuite) {
        $suites = @($xml.testsuite)
    }
    $tests = 0
    $failures = 0
    $errors = 0
    $skipped = 0
    foreach ($suite in $suites) {
        $tests += [int]$suite.tests
        $failures += [int]$suite.failures
        $errors += [int]$suite.errors
        $skipped += [int]$suite.skipped
    }
    $passed = $tests - $failures - $errors - $skipped
    return [ordered]@{
        tests = $tests
        failures = $failures
        errors = $errors
        skipped = $skipped
        passed = $passed
        xfailed = 0
        xpassed = 0
    }
}

function Write-JsonFile([string]$Path, $Value) {
    $json = $Value | ConvertTo-Json -Depth 20
    [System.IO.File]::WriteAllText($Path, $json, [System.Text.UTF8Encoding]::new($false))
}

$outputFull = [System.IO.Path]::GetFullPath($OutputDirectory)
if ((Test-Path -LiteralPath $outputFull) -and -not $Resume) {
    Remove-Item -LiteralPath $outputFull -Recurse -Force
}
New-Directory $outputFull

$collectStdout = Join-Path $outputFull "collect-stdout.txt"
$collectStderr = Join-Path $outputFull "collect-stderr.txt"
$collectStartedAt = Get-Date
$collect = Invoke-CapturedProcess -FilePath $PythonPath -Arguments @("-m", "pytest", "--collect-only", "-q", $TestsPath) -StdoutPath $collectStdout -StderrPath $collectStderr
$collectEndedAt = Get-Date
$collectLines = Get-Content -LiteralPath $collectStdout
$nodeids = @($collectLines | Where-Object { $_ -match '^.+\.py::.+$' })
$duplicateNodeids = @($nodeids | Group-Object | Where-Object { $_.Count -gt 1 } | ForEach-Object { $_.Name })
$collectedFiles = @($nodeids | ForEach-Object { ($_ -split '::')[0] } | Sort-Object -Unique)
$testFiles = @(Get-ChildItem -LiteralPath $TestsPath -Recurse -File -Filter "test_*.py" | ForEach-Object { ConvertTo-RepoRelativePath $_.FullName } | Sort-Object -Unique)
$missingTestFiles = @($testFiles | Where-Object { $collectedFiles -notcontains $_ })

if ($collect.exit_code -ne 0) {
    $summary = [ordered]@{
        status = "FAILED"
        collection_errors = 1
        collected_total = $nodeids.Count
        executed_total = 0
        passed = 0
        failed = 0
        errors = 1
        skipped = 0
        xfailed = 0
        xpassed = 0
        shard_count = 0
        passed_shards = 0
        failed_shards = 0
        timeout_shards = 0
        missing_test_files = $missingTestFiles
        duplicate_nodeids = $duplicateNodeids
        collection_duration_seconds = [Math]::Round(($collectEndedAt - $collectStartedAt).TotalSeconds, 3)
    }
    Write-JsonFile (Join-Path $outputFull "summary.json") $summary
    exit 1
}

$manifest = [ordered]@{
    generated_at = (Get-Date).ToUniversalTime().ToString("o")
    python_path = $PythonPath
    tests_path = $TestsPath
    output_directory = $OutputDirectory
    collected_total = $nodeids.Count
    collected_test_files = $collectedFiles
    all_test_files = $testFiles
    missing_test_files = $missingTestFiles
    duplicate_nodeids = $duplicateNodeids
    shards = @()
}

$shardResults = New-Object System.Collections.Generic.List[object]
$index = 0
foreach ($file in $testFiles) {
    $index += 1
    $shardName = "shard-{0:D3}" -f $index
    $shardDir = Join-Path $outputFull $shardName
    New-Directory $shardDir
    $stdoutPath = Join-Path $shardDir "stdout.txt"
    $stderrPath = Join-Path $shardDir "stderr.txt"
    $junitPath = Join-Path $shardDir "junit.xml"
    $exitPath = Join-Path $shardDir "exit-code.txt"
    $resultPath = Join-Path $shardDir "result.json"

    if ($Resume -and (Test-Path -LiteralPath $resultPath)) {
        $existing = Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json
        if ($existing.exit_code -eq 0 -and -not $existing.timed_out) {
            [void]$shardResults.Add($existing)
            $manifest.shards += [ordered]@{ shard = $shardName; file = $file; resumed = $true }
            continue
        }
    }

    $run = Invoke-CapturedProcess -FilePath $PythonPath -Arguments @("-m", "pytest", "-q", $file, "--junitxml=$junitPath") -StdoutPath $stdoutPath -StderrPath $stderrPath -TimeoutSeconds $PerShardTimeoutSeconds
    $junit = Read-JUnitSummary $junitPath
    Set-Content -LiteralPath $exitPath -Value ([string]$run.exit_code)
    $result = [ordered]@{
        shard = $shardName
        file = $file
        exit_code = $run.exit_code
        timed_out = $run.timed_out
        duration_seconds = $run.duration_seconds
        collected_count = $junit.tests
        passed = $junit.passed
        failed = $junit.failures
        errors = $junit.errors
        skipped = $junit.skipped
        xfailed = $junit.xfailed
        xpassed = $junit.xpassed
        junit_xml = $junitPath
        stdout = $stdoutPath
        stderr = $stderrPath
    }
    Write-JsonFile $resultPath $result
    [void]$shardResults.Add([pscustomobject]$result)
    $manifest.shards += [ordered]@{ shard = $shardName; file = $file; resumed = $false }

    if ($run.exit_code -ne 0 -and -not $ContinueOnFailure) {
        break
    }
}

$executedTotal = 0
$passedTotal = 0
$failedTotal = 0
$errorsTotal = 0
$skippedTotal = 0
$xfailedTotal = 0
$xpassedTotal = 0
foreach ($result in $shardResults) {
    $executedTotal += [int]$result.collected_count
    $passedTotal += [int]$result.passed
    $failedTotal += [int]$result.failed
    $errorsTotal += [int]$result.errors
    $skippedTotal += [int]$result.skipped
    $xfailedTotal += [int]$result.xfailed
    $xpassedTotal += [int]$result.xpassed
}

$passedShardCount = @($shardResults | Where-Object { $_.exit_code -eq 0 -and -not $_.timed_out }).Count
$timeoutShardCount = @($shardResults | Where-Object { $_.timed_out }).Count
$failedShardCount = @($shardResults | Where-Object { $_.exit_code -ne 0 -or $_.timed_out }).Count
$missingCollectedNodeids = @()

$status = "PASSED"
if (
    $duplicateNodeids.Count -gt 0 -or
    $missingTestFiles.Count -gt 0 -or
    $missingCollectedNodeids.Count -gt 0 -or
    $failedShardCount -gt 0 -or
    $failedTotal -gt 0 -or
    $errorsTotal -gt 0 -or
    $timeoutShardCount -gt 0 -or
    [int]$executedTotal -ne $nodeids.Count
) {
    $status = "FAILED"
}

$summary = [ordered]@{
    status = $status
    collection_errors = 0
    collected_total = $nodeids.Count
    executed_total = [int]$executedTotal
    passed = [int]$passedTotal
    failed = [int]$failedTotal
    errors = [int]$errorsTotal
    skipped = [int]$skippedTotal
    xfailed = [int]$xfailedTotal
    xpassed = [int]$xpassedTotal
    shard_count = $testFiles.Count
    passed_shards = $passedShardCount
    failed_shards = $failedShardCount
    timeout_shards = $timeoutShardCount
    missing_test_files = $missingTestFiles
    duplicate_nodeids = @($duplicateNodeids | Sort-Object -Unique)
    missing_collected_nodeids = $missingCollectedNodeids
    collection_duration_seconds = [Math]::Round(($collectEndedAt - $collectStartedAt).TotalSeconds, 3)
    total_duration_seconds = [Math]::Round((($shardResults | ForEach-Object { [double]$_.duration_seconds } | Measure-Object -Sum).Sum), 3)
    slowest_shards = @($shardResults | Sort-Object -Property duration_seconds -Descending | Select-Object -First 10 | ForEach-Object {
        [ordered]@{ shard = $_.shard; file = $_.file; duration_seconds = $_.duration_seconds }
    })
}

Write-JsonFile (Join-Path $outputFull "manifest.json") $manifest
Write-JsonFile (Join-Path $outputFull "summary.json") $summary

$summaryMarkdown = @(
    "# Full Pytest Sharded Summary",
    "",
    "- Status: $($summary.status)",
    "- Collected tests: $($summary.collected_total)",
    "- Executed tests: $($summary.executed_total)",
    "- Passed: $($summary.passed)",
    "- Failed: $($summary.failed)",
    "- Errors: $($summary.errors)",
    "- Skipped: $($summary.skipped)",
    "- Shards: $($summary.shard_count)",
    "- Passed shards: $($summary.passed_shards)",
    "- Failed shards: $($summary.failed_shards)",
    "- Timeout shards: $($summary.timeout_shards)",
    "- Missing test files: $($summary.missing_test_files.Count)",
    "- Duplicate node IDs: $($summary.duplicate_nodeids.Count)",
    "- Total shard runtime seconds: $($summary.total_duration_seconds)",
    "",
    "## Slowest shards",
    ""
)
foreach ($shard in $summary.slowest_shards) {
    $summaryMarkdown += "- $($shard.file): $($shard.duration_seconds)s"
}
[System.IO.File]::WriteAllText((Join-Path $outputFull "summary.md"), ($summaryMarkdown -join "`n"), [System.Text.UTF8Encoding]::new($false))

if ($summary.status -ne "PASSED") {
    exit 1
}
exit 0
