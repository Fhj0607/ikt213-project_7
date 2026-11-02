$ErrorActionPreference = 'Stop'
$vlPython = Join-Path $PSScriptRoot '.venv-vl/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $vlPython)) {
    throw 'The VL environment is missing. Follow PADDLEOCR_VL.md to install it.'
}
Push-Location $PSScriptRoot
try {
    & $vlPython (Join-Path $PSScriptRoot 'main.py')
} finally {
    Pop-Location
}
