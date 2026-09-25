param(
    [Parameter()]
    [string]$Prefix = $env:CONDA_PREFIX
)

$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($Prefix)) {
    throw 'Pass -Prefix with the Conda environment path, for example D:\MINICONDA\envs\onevoice.'
}

$resolvedPrefix = (Resolve-Path -LiteralPath $Prefix).Path
$condaMeta = Join-Path $resolvedPrefix 'conda-meta'
$espeakDir = Join-Path $resolvedPrefix 'espeak-ng-runtime\eSpeak NG'
$espeakExe = Join-Path $espeakDir 'espeak-ng.exe'
$espeakData = Join-Path $espeakDir 'espeak-ng-data'

if (-not (Test-Path -LiteralPath $condaMeta -PathType Container)) {
    throw "Not a Conda environment (missing conda-meta): $resolvedPrefix"
}
if (-not (Test-Path -LiteralPath $espeakExe -PathType Leaf)) {
    throw "The expected eSpeak NG executable is missing: $espeakExe"
}
if (-not (Test-Path -LiteralPath $espeakData -PathType Container)) {
    throw "The expected eSpeak NG data directory is missing: $espeakData"
}

$activateDir = Join-Path $resolvedPrefix 'etc\conda\activate.d'
$deactivateDir = Join-Path $resolvedPrefix 'etc\conda\deactivate.d'
$null = New-Item -ItemType Directory -Path $activateDir -Force
$null = New-Item -ItemType Directory -Path $deactivateDir -Force

$activateHook = Join-Path $activateDir 'onevoice_espeak_ng.ps1'
$deactivateHook = Join-Path $deactivateDir 'onevoice_espeak_ng.ps1'
$activateContent = @'
# Added by OneVoice scripts/install_espeak_conda_hooks.ps1.
$oneVoiceEspeakDir = Join-Path $env:CONDA_PREFIX 'espeak-ng-runtime\eSpeak NG'
$oneVoiceEspeakData = Join-Path $oneVoiceEspeakDir 'espeak-ng-data'
$oneVoiceEspeakExe = Join-Path $oneVoiceEspeakDir 'espeak-ng.exe'
if (-not (Test-Path -LiteralPath $oneVoiceEspeakExe -PathType Leaf) -or
    -not (Test-Path -LiteralPath $oneVoiceEspeakData -PathType Container)) {
    throw "OneVoice eSpeak NG files are missing under $oneVoiceEspeakDir"
}

if (-not (Test-Path Env:ONEVOICE_ESPEAK_DATA_PATH_WAS_SET)) {
    if (Test-Path Env:ESPEAK_DATA_PATH) {
        $env:ONEVOICE_ESPEAK_DATA_PATH_WAS_SET = '1'
        $env:ONEVOICE_PREVIOUS_ESPEAK_DATA_PATH = $env:ESPEAK_DATA_PATH
    }
    else {
        $env:ONEVOICE_ESPEAK_DATA_PATH_WAS_SET = '0'
    }
}

$oneVoicePathEntries = @($env:PATH -split ';')
if (-not ($oneVoicePathEntries | Where-Object { $_.TrimEnd('\') -ieq $oneVoiceEspeakDir.TrimEnd('\') })) {
    $env:PATH = "$oneVoiceEspeakDir;$env:PATH"
}
$env:ESPEAK_DATA_PATH = $oneVoiceEspeakData
'@
$deactivateContent = @'
# Added by OneVoice scripts/install_espeak_conda_hooks.ps1.
$oneVoiceEspeakDir = Join-Path $env:CONDA_PREFIX 'espeak-ng-runtime\eSpeak NG'
$env:PATH = (@($env:PATH -split ';') | Where-Object { $_.TrimEnd('\') -ine $oneVoiceEspeakDir.TrimEnd('\') }) -join ';'

if ($env:ONEVOICE_ESPEAK_DATA_PATH_WAS_SET -eq '1') {
    $env:ESPEAK_DATA_PATH = $env:ONEVOICE_PREVIOUS_ESPEAK_DATA_PATH
}
elseif ($env:ONEVOICE_ESPEAK_DATA_PATH_WAS_SET -eq '0') {
    Remove-Item Env:ESPEAK_DATA_PATH -ErrorAction SilentlyContinue
}
Remove-Item Env:ONEVOICE_ESPEAK_DATA_PATH_WAS_SET -ErrorAction SilentlyContinue
Remove-Item Env:ONEVOICE_PREVIOUS_ESPEAK_DATA_PATH -ErrorAction SilentlyContinue
'@

$hooks = @(
    @{ Path = $activateHook; Content = $activateContent },
    @{ Path = $deactivateHook; Content = $deactivateContent }
)
$encoding = [System.Text.UTF8Encoding]::new($false)
foreach ($hook in $hooks) {
    if (Test-Path -LiteralPath $hook.Path -PathType Leaf) {
        $existingContent = [System.IO.File]::ReadAllText($hook.Path)
        if ($existingContent -cne $hook.Content) {
            throw "Refusing to overwrite an existing non-OneVoice Conda hook: $($hook.Path)"
        }
        Write-Output "Already configured: $($hook.Path)"
        continue
    }
    [System.IO.File]::WriteAllText($hook.Path, $hook.Content, $encoding)
    Write-Output "Installed: $($hook.Path)"
}

Write-Output "eSpeak NG Conda hooks are ready for: $resolvedPrefix"
Write-Output 'Open a new PowerShell session, run conda activate onevoice, then verify with Get-Command espeak-ng and $env:ESPEAK_DATA_PATH.'
