<#
.SYNOPSIS
Install SnapPrint on Windows and prepare its local models.
.EXAMPLE
powershell -ExecutionPolicy Bypass -File .\setup.ps1
.EXAMPLE
.\setup.ps1 -TorchDevice cpu -SkipOcrModels -SkipSmokeTest
#>
[CmdletBinding()]
param(
    [ValidateSet('auto', 'cpu', 'cuda')][string]$TorchDevice = 'auto',
    [string]$PythonPath,
    [string]$HogModelPath,
    [switch]$TrainHog,
    [switch]$SkipOcrModels,
    [switch]$SkipLlm,
    [switch]$SkipSmokeTest,
    [switch]$CheckOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$envPython = Join-Path $projectRoot '.venv-vl\Scripts\python.exe'
$utf8 = New-Object System.Text.UTF8Encoding($false)

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed (exit $LASTEXITCODE): $Executable $($Arguments -join ' ')"
    }
}

function Find-Python {
    if ($PythonPath) {
        if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
            throw "Python executable does not exist: $PythonPath"
        }
        return (Resolve-Path -LiteralPath $PythonPath).Path
    }
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python311\python.exe'),
        (Join-Path $env:ProgramFiles 'Python311\python.exe')
    )
    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($pythonCommand -and $pythonCommand.Source -notlike '*\WindowsApps\*') {
        $candidates += $pythonCommand.Source
    }
    $pyCommand = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($pyCommand) {
        $previous = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        $located = & $pyCommand.Source -3.11 -c 'import sys; print(sys.executable)' 2>$null
        $ErrorActionPreference = $previous
        if ($LASTEXITCODE -eq 0 -and $located) { $candidates += [string]$located }
    }
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            & $candidate -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 11) else 1)' 2>$null
            if ($LASTEXITCODE -eq 0) { return $candidate }
        }
    }
    return $null
}

function Find-Ollama {
    $command = Get-Command ollama.exe -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $candidate = Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe'
    if (Test-Path -LiteralPath $candidate) { return $candidate }
    return $null
}

function Install-WingetPackage {
    param([string]$Id)
    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "Install $Id manually, then rerun setup. Automatic installation needs Windows App Installer (winget)."
    }
    Invoke-Checked $winget.Source @('install', '--id', $Id, '--exact', '--source', 'winget',
        '--silent', '--accept-package-agreements', '--accept-source-agreements', '--disable-interactivity')
}

function Test-OllamaReady {
    try {
        $null = Invoke-RestMethod -Uri 'http://localhost:11434/api/version' -TimeoutSec 3
        return $true
    } catch { return $false }
}

function Get-QwenModel {
    $folder = Join-Path $projectRoot '.models\llm\qwen3.5-2b'
    $target = Join-Path $folder 'Qwen_Qwen3.5-2B-Q4_K_M.gguf'
    $expectedHash = '57a1085840f497d764a7fc5d346922dbde961efb54cc792ea81d694fd846a1d8'
    $expectedSize = 1396198496
    $url = 'https://huggingface.co/bartowski/Qwen_Qwen3.5-2B-GGUF/resolve/7d26695454df6de5fbcce2e58681e62dae06ce43/Qwen_Qwen3.5-2B-Q4_K_M.gguf'
    $null = New-Item -ItemType Directory -Force -Path $folder
    if (Test-Path -LiteralPath $target) {
        if ((Get-Item -LiteralPath $target).Length -eq $expectedSize -and
            (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -eq $expectedHash) {
            Write-Host 'Qwen download already verified.'
            return
        }
        throw "Existing Qwen file failed verification: $target. Move it aside and rerun setup."
    }
    $partial = "$target.partial"
    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if (-not $curl) { throw 'curl.exe is required for the resumable model download.' }
    Write-Host 'Downloading Qwen (1.4 GB). Interrupted downloads resume on the next run.'
    Invoke-Checked $curl.Source @('--fail', '--location', '--retry', '3', '--continue-at', '-',
        '--output', $partial, $url)
    if ((Get-Item -LiteralPath $partial).Length -ne $expectedSize -or
        (Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash -ne $expectedHash) {
        throw "Qwen download failed size/SHA-256 verification. Move $partial aside and retry."
    }
    Move-Item -LiteralPath $partial -Destination $target
}

if ($env:OS -ne 'Windows_NT' -or -not [Environment]::Is64BitOperatingSystem) {
    throw 'SnapPrint setup requires 64-bit Windows.'
}
if ($HogModelPath -and $TrainHog) { throw 'Choose either -HogModelPath or -TrainHog.' }
if ($HogModelPath -and -not (Test-Path -LiteralPath $HogModelPath -PathType Leaf)) {
    throw "HOG model not found: $HogModelPath"
}
# Resolve before changing directory so caller-relative paths work.
if ($HogModelPath) { $HogModelPath = (Resolve-Path -LiteralPath $HogModelPath).Path }
if ($PythonPath) { $PythonPath = (Resolve-Path -LiteralPath $PythonPath).Path }
if ($TrainHog -and -not $CheckOnly) {
    foreach ($required in @('cnn/datasets/raw/cocotext.v2.json', 'cnn/datasets/text', 'cnn/datasets/noText')) {
        if (-not (Test-Path -LiteralPath (Join-Path $projectRoot $required))) {
            throw "HOG training requires prepared data: $required. See SETUP.md."
        }
    }
}

Push-Location $projectRoot
try {
    if ($CheckOnly) {
        Write-Host "Python 3.11: $(Find-Python)"
        Write-Host "Project interpreter exists: $(Test-Path -LiteralPath $envPython)"
        Write-Host "Ollama executable: $(Find-Ollama)"
        Write-Host "Ollama server running: $(Test-OllamaReady)"
        Write-Host "Qwen file exists: $(Test-Path -LiteralPath '.models\llm\qwen3.5-2b\Qwen_Qwen3.5-2B-Q4_K_M.gguf')"
        Write-Host "HOG model exists: $(Test-Path -LiteralPath 'hog\models\hog_svm.joblib')"
        Write-Host 'Check only: nothing installed or downloaded; model validity and inference were not tested.'
        return
    }

    Write-Host 'SnapPrint setup: Python, dependencies, local models, and validation.'
    Write-Host 'Internet and several GB of free disk space are needed. Existing environments are reused.'
    if (-not (Test-Path -LiteralPath $envPython)) {
        $basePython = Find-Python
        if (-not $basePython) {
            Install-WingetPackage 'Python.Python.3.11'
            $basePython = Find-Python
        }
        if (-not $basePython) { throw 'Python 3.11 not found after installation. Rerun with -PythonPath C:\path\python.exe.' }
        Invoke-Checked $basePython @('-c', 'import sys; assert sys.version_info[:2] == (3, 11), "Python 3.11 is required"; assert sys.maxsize > 2**32, "64-bit Python is required"')
        Invoke-Checked $basePython @('-m', 'venv', (Join-Path $projectRoot '.venv-vl'))
    }
    Invoke-Checked $envPython @('-c', 'import sys; assert sys.version_info[:2] == (3, 11), "Existing .venv-vl must use Python 3.11"; assert sys.maxsize > 2**32')
    Invoke-Checked $envPython @('-m', 'pip', 'install', '--upgrade', 'pip')

    $selectedDevice = $TorchDevice
    if ($selectedDevice -eq 'auto') {
        $selectedDevice = 'cpu'
        $smi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
        if ($smi) {
            & $smi.Source --query-gpu=name --format=csv,noheader 2>$null | Out-Host
            if ($LASTEXITCODE -eq 0) { $selectedDevice = 'cuda' }
        }
    }
    Write-Host "Installing PyTorch for $selectedDevice. Use -TorchDevice cpu to force CPU."
    # pip considers a CPU wheel to satisfy the same version as a CUDA wheel.
    # Force replacement only when an existing environment has the other backend.
    $torchFlags = @()
    $backend = & $envPython -c 'import importlib.util; print("missing" if importlib.util.find_spec("torch") is None else __import__("torch").version.cuda or "cpu")'
    if ($LASTEXITCODE -ne 0) { throw 'Existing PyTorch cannot be inspected. Repair or move aside .venv-vl and rerun setup.' }
    if (($selectedDevice -eq 'cuda' -and $backend -eq 'cpu') -or
        ($selectedDevice -eq 'cpu' -and $backend -notin @('cpu', 'missing'))) {
        $torchFlags += '--force-reinstall'
    }
    if ($selectedDevice -eq 'cuda') {
        Invoke-Checked $envPython (@('-m', 'pip', 'install', '--no-compile', '-r', 'requirements-gpu.txt') + $torchFlags)
    } else {
        Invoke-Checked $envPython (@('-m', 'pip', 'install', '--no-compile', '--index-url',
            'https://download.pytorch.org/whl/cpu', 'torch==2.11.0', 'torchvision==0.26.0') + $torchFlags)
    }
    Invoke-Checked $envPython @('-m', 'pip', 'install', '--no-compile', '-r', 'requirements.txt')
    Invoke-Checked $envPython @('-m', 'pip', 'check')

    if ($HogModelPath) {
        $destination = Join-Path $projectRoot 'hog\models\hog_svm.joblib'
        $null = New-Item -ItemType Directory -Force -Path (Split-Path $destination)
        if ($HogModelPath -ne $destination) { Copy-Item -LiteralPath $HogModelPath -Destination $destination }
    }
    if ($TrainHog) {
        Invoke-Checked $envPython @('-m', 'hog.makeDataset')
        Invoke-Checked $envPython @('-m', 'hog.trainHog')
    }

    $verifyArgs = @('scripts/setup_verify.py', '--device', $selectedDevice)
    if (-not $SkipOcrModels) { $verifyArgs += '--preload-ocr' }
    if (-not $SkipLlm) {
        $ollama = Find-Ollama
        if (-not $ollama) {
            Install-WingetPackage 'Ollama.Ollama'
            $ollama = Find-Ollama
        }
        if (-not $ollama) { throw 'Ollama was not found after installation. Reopen the terminal and rerun setup.' }
        Invoke-Checked $ollama @('--version')
        if (-not (Test-OllamaReady)) {
            $logFolder = Join-Path $projectRoot 'results\setup'
            $null = New-Item -ItemType Directory -Force -Path $logFolder
            $null = Start-Process -FilePath $ollama -ArgumentList 'serve' -WindowStyle Hidden -PassThru `
                -RedirectStandardOutput (Join-Path $logFolder 'ollama-stdout.log') `
                -RedirectStandardError (Join-Path $logFolder 'ollama-stderr.log')
            $deadline = (Get-Date).AddSeconds(60)
            while (-not (Test-OllamaReady)) {
                if ((Get-Date) -ge $deadline) { throw "Ollama did not start. Check $logFolder\ollama-stderr.log." }
                Start-Sleep -Seconds 1
            }
        }
        Get-QwenModel
        # Relative FROM paths are resolved against the Modelfile directory.
        Invoke-Checked $ollama @('create', 'snapprint-summary', '-f', (Join-Path $projectRoot 'llm\Modelfile'))
        $verifyArgs += '--check-llm'
        if (-not $SkipSmokeTest) { $verifyArgs += '--smoke-llm' }
    }
    Invoke-Checked $envPython $verifyArgs

    $modelDir = Join-Path $projectRoot '.models'
    $null = New-Item -ItemType Directory -Force -Path $modelDir
    $ocrDevice = if ($selectedDevice -eq 'cuda') { 'gpu:0' } else { 'cpu' }
    $config = @{ ocr_device = $ocrDevice } | ConvertTo-Json
    [IO.File]::WriteAllText((Join-Path $modelDir 'setup.json'), $config, $utf8)
    Write-Host "Setup completed. Start SnapPrint with: .\run-snapprint.ps1"
    if ($selectedDevice -eq 'cpu') { Write-Host 'CPU OCR is available but may be slow and memory intensive.' }
    if ($SkipOcrModels) { Write-Warning 'OCR model preload skipped. First startup still needs internet to download OCR models.' }
    if ($SkipLlm) { Write-Warning 'LLM setup skipped. Word summarization needs Ollama and snapprint-summary.' }
    if (-not (Test-Path -LiteralPath 'hog/models/hog_svm.joblib')) {
        Write-Warning 'Optional HOG filter has no weights. Use PaddleOCR or CNN, or rerun with -HogModelPath / -TrainHog.'
    }
} finally {
    Pop-Location
}
