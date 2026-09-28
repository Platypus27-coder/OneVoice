param(
    [string]$Python = 'D:\MINICONDA\envs\onevoice\python.exe',
    [string]$BundleRoot = 'D:\OneVoiceDesktop\onevoice-v2-rc1',
    [string]$OutputRoot = '',
    [switch]$PrepareOnly,
    [int]$SmokeTestCases = 0
)

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if (-not $OutputRoot) { $OutputRoot = Join-Path $RepoRoot 'reports\pc_pipeline_afterfix_v2' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python not found: $Python" }
if ($SmokeTestCases -lt 0) { throw 'SmokeTestCases must be non-negative' }
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'

function Invoke-Benchmark([string[]]$CommandArgs) {
    Write-Host ('> ' + $Python + ' ' + ($CommandArgs -join ' '))
    & $Python @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        # Functional failures are retained in the full report; complete the
        # other direction too, but never label a failed stage as PASS.
        Write-Warning "Benchmark exit code $LASTEXITCODE. Inspect the saved report."
        $script:StageFailures += 1
    }
}

$StageFailures = 0
$OriginalDirectory = Get-Location
try {
    foreach ($Direction in @('vi2en', 'en2vi')) {
        $Language = if ($Direction -eq 'vi2en') { 'vi' } else { 'en' }
        $Bundle = Join-Path $BundleRoot $Direction
        $Config = Join-Path $Bundle 'runtime_config.yaml'
        $Manifest = Join-Path $BundleRoot ('test_audio\' + $Language + '_noise\' + $Language + '_manifest.jsonl')
        Set-Location -LiteralPath $Bundle
        $DirectionOutput = Join-Path $OutputRoot $Direction
        $PipelineArgs = @('-u', (Join-Path $RepoRoot 'scripts\benchmark_desktop_pipeline.py'),
            '--config', $Config, '--manifest', $Manifest, '--direction', $Direction,
            '--output-dir', (Join-Path $DirectionOutput 'pipeline_full_noisy'), '--tts-backend', 'auto', '--resume')
        if ($PrepareOnly) { $PipelineArgs += '--prepare-only' }
        if ($SmokeTestCases -gt 0) { $PipelineArgs += @('--limit-test-cases', "$SmokeTestCases", '--limit-safety-cases', "$SmokeTestCases") }
        Invoke-Benchmark $PipelineArgs
        if ($PrepareOnly -or $SmokeTestCases -gt 0) { continue }
        Invoke-Benchmark @('-u', (Join-Path $RepoRoot 'scripts\benchmark_offline_tts.py'),
            '--config', $Config, '--prompt-csv', (Join-Path $RepoRoot 'data\onevoice_construction_v2\test.csv'),
            '--direction', $Direction, '--profile', 'edge', '--tts-backend', 'auto',
            '--report-dir', (Join-Path $DirectionOutput 'tts_full_test'), '--resume')
        $NormalInput = Join-Path $BundleRoot $(if ($Direction -eq 'vi2en') { 'test_audio\OV2_000001_clean.wav' } else { 'test_audio\OV2_000001_en_clean.wav' })
        $SafetyInput = Join-Path $Bundle $(if ($Direction -eq 'vi2en') { 'artifacts\safety_audio\SAFE2_0001_en2vi.wav' } else { 'artifacts\safety_audio\SAFE2_0001_vi2en.wav' })
        Invoke-Benchmark @('-u', (Join-Path $RepoRoot 'scripts\profile_release_runtime.py'),
            '--config', $Config, '--direction', $Direction, '--profile', 'edge',
            '--normal-input', $NormalInput, '--safety-input', $SafetyInput, '--repeats', '5',
            '--hardware-profile', 'desktop_tuf_f15_rtx2050', '--max-rss-mb', '8192',
            '--normal-target-ms', '3000', '--safety-target-ms', '300',
            '--report-dir', (Join-Path $DirectionOutput 'profile'))
    }
} finally {
    Set-Location -LiteralPath $OriginalDirectory.Path
}
Write-Host "Reports: $OutputRoot; failed stages: $StageFailures"
if ($StageFailures -gt 0) { exit 1 }
