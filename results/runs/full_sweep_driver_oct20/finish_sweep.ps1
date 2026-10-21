$workspace = 'C:\Personal\cse291a-memory-aware-serving'
$runnerPid = 18996
$serverPid = 48700
$runDir = Join-Path $workspace 'results\runs\full_cold_oct20_01'
$logDir = Join-Path $workspace 'results\runs\full_sweep_driver_oct20'
$summaryPath = Join-Path $runDir 'summary_trials.csv'
$statusPath = Join-Path $logDir 'finish_status.json'

while (Get-CimInstance Win32_Process -Filter "ProcessId = $runnerPid" -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 30
}

$completed = 0
if (Test-Path -LiteralPath $summaryPath) {
    $completed = @(Import-Csv -LiteralPath $summaryPath).Count
}

$analysisExitCode = $null
if ($completed -eq 54) {
    Push-Location $workspace
    try {
        & (Join-Path $workspace '.venv\Scripts\python.exe') (Join-Path $workspace 'analysis\analyze.py') --results-dir $runDir *> (Join-Path $logDir 'analysis.log')
        $analysisExitCode = $LASTEXITCODE
    } finally {
        Pop-Location
    }
}

$server = Get-CimInstance Win32_Process -Filter "ProcessId = $serverPid" -ErrorAction SilentlyContinue
if ($server -and $server.CommandLine -match 'llama-server' -and $server.CommandLine -match '8081') {
    Stop-Process -Id $serverPid -ErrorAction SilentlyContinue
}

@{
    finished_at = (Get-Date).ToString('o')
    completed_trials = $completed
    expected_trials = 54
    analysis_exit_code = $analysisExitCode
} | ConvertTo-Json | Set-Content -LiteralPath $statusPath
