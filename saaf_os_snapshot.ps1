# saaf_os_snapshot.ps1

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$snapshotDir = "saaf_os_diag_$timestamp"
$subdirs = @("tests", "demo_logs", "models", "memory", "diagnostics")

# Create directories
foreach ($subdir in $subdirs) {
    New-Item -Path "$snapshotDir\$subdir" -ItemType Directory -Force | Out-Null
}

Write-Host "Running full test suite..."
pytest -v --tb=short | Out-File "$snapshotDir\diagnostics\pytest.log" -Encoding utf8

Write-Host "Running scenario simulations..."
$scenarios = @("solar_conflict", "veto_loop", "alienation_drift")
foreach ($scenario in $scenarios) {
    Write-Host "▶️ Running scenario: $scenario"
    python scripts\run_demo.py --scenario $scenario *> "$snapshotDir\demo_logs\$scenario.log"
}

Write-Host "Copying memory logs..."
Get-ChildItem -Path memory\*_episodes.jsonl -ErrorAction SilentlyContinue | Copy-Item -Destination "$snapshotDir\memory" -Force
Get-ChildItem -Path memory\*.log -ErrorAction SilentlyContinue | Copy-Item -Destination "$snapshotDir\memory" -Force

Write-Host " Copying trained models..."
Get-ChildItem -Path models\*.pt -ErrorAction SilentlyContinue | Copy-Item -Destination "$snapshotDir\models" -Force

Write-Host "Saving environment metadata..."
git rev-parse HEAD | Out-File "$snapshotDir\diagnostics\git_commit.txt"
python --version | Out-File "$snapshotDir\diagnostics\python_version.txt"
pip freeze | Out-File "$snapshotDir\diagnostics\pip_freeze.txt"
tree /F /A | Out-File "$snapshotDir\diagnostics\file_tree.txt"
Write-Host "Copying reports and plots..."
if (Test-Path "reports") {
    Copy-Item -Path reports\*.json -Destination "$snapshotDir\diagnostics" -Force -ErrorAction SilentlyContinue
}
if (Test-Path "plots") {
    Copy-Item -Path plots\*.png -Destination "$snapshotDir\diagnostics" -Force -ErrorAction SilentlyContinue
}

Write-Host "Copying additional logs..."
$logFiles = @("synthesis_log.jsonl", "governance\audit_log.jsonl")
foreach ($file in $logFiles) {
    if (Test-Path $file) {
        Copy-Item $file "$snapshotDir\diagnostics" -Force
    }
}

$agentLogs = Get-ChildItem -Path *.jsonl -Recurse | Where-Object { $_.Name -like "*_bus*.jsonl" }
foreach ($log in $agentLogs) {
    Copy-Item $log.FullName "$snapshotDir\diagnostics" -Force
}

Write-Host "Zipping everything..."
# Zip the snapshot directory
$zipPath = "saaf_os_diag_$timestamp.zip"
Compress-Archive -Path "$snapshotDir\*" -DestinationPath $zipPath -Force

# Clean up the directory
Remove-Item "$snapshotDir" -Recurse -Force

# Final message
Write-Host ("[OK] Snapshot complete: {0}" -f $zipPath)
