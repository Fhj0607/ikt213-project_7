# SnapPrint

Run these commands in PowerShell on 64-bit Windows. You need Git, an internet connection, several GB of free disk space, and Windows App Installer (`winget`) if Python 3.11 or Ollama is not already installed.

## First install

```powershell
git clone --branch main https://github.com/Fhj0607/ikt213-project_7.git
cd ikt213-project_7
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\run-snapprint.ps1
```

Wait for `Setup completed` before running the launcher. If setup is interrupted, rerun the same `setup.ps1` command.
