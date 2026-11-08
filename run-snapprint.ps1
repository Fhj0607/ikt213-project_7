[CmdletBinding()]
param([ValidateSet('auto', 'cpu', 'gpu:0')][string]$Device)
$ErrorActionPreference = 'Stop'
$projectPython = Join-Path $PSScriptRoot '.venv-vl\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) { throw 'Run .\setup.ps1 first.' }
$configPath = Join-Path $PSScriptRoot '.models\setup.json'
if (-not $Device -and (Test-Path -LiteralPath $configPath)) {
    $Device = (Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json).ocr_device
}
$previousDevice = $env:SNAPPRINT_OCR_DEVICE
Push-Location $PSScriptRoot
try {
    if ($Device) { $env:SNAPPRINT_OCR_DEVICE = $Device }
    & $projectPython (Join-Path $PSScriptRoot 'main.py')
    if ($LASTEXITCODE -ne 0) { throw "SnapPrint exited with code $LASTEXITCODE." }
} finally {
    $env:SNAPPRINT_OCR_DEVICE = $previousDevice
    Pop-Location
}
