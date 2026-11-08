# Windows setup

From the cloned repository, open PowerShell and run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
```

After setup succeeds, start the application with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run-snapprint.ps1
```

No Conda activation is needed. These commands apply the execution policy only
to that PowerShell process. Setup uses the repository directory even when run
from another working directory. A camera is needed only for camera capture.

## What setup does

1. Reuses `.venv-vl`, or finds 64-bit Python 3.11 and creates it. If Python is
   missing, installs it using Windows App Installer (`winget`). An existing
   `.venv-vl` must use Python 3.11; setup does not delete environments.
2. Installs the pinned PyTorch/TorchVision pair plus the application requirements,
   including CPU PaddlePaddle, OpenCV, Word export, and HOG dependencies. Automatic
   selection uses the CUDA 12.8 wheel when `nvidia-smi` detects an NVIDIA GPU,
   otherwise CPU wheels. It replaces an existing opposite-backend torch build.
3. Installs Ollama through winget if needed and starts its local server if it is
   not already running. The server stays running for the app.
4. Downloads the approximately 1.4 GB Qwen GGUF from the revision recorded in
   `LLM_SETUP.md`, checks its exact size and SHA-256, and imports it into Ollama as
   `snapprint-summary`. Existing verified files are reused; partial downloads
   resume. An existing file with the wrong checksum causes an error, not deletion.
5. Checks imports, CUDA/bfloat16 support when selected, and the included CNN
   weights. Downloads and loads the PaddleOCR detector and OCR-VL models using
   the same code as the app, then releases them.
6. Checks Ollama model registration, sends a short CPU summarization request,
   validates the response, and creates/reopens a temporary Word document to
   check export and source preservation. The temporary document is removed.
7. Saves the selected OCR device in ignored `.models/setup.json`. The launcher
   uses this setting and the project interpreter.

Initial setup needs internet and several GB of free disk space. OCR model
downloads add to the Qwen download, and Ollama maintains its own model store,
which can use additional space. Models remain on this machine; setup does not
upload captured images or notes. Downloading a model includes no assurance
about its accuracy on your documents.

Setup can take a long time during downloads, OCR loading, and CPU inference.
Rerun the same command after an interruption. pip checks and failures stop
setup with an error; a completion message is printed only after validation.
It does not install graphics drivers or guarantee compatibility with every GPU.

## Options

```powershell
# Inspect installed prerequisites; no installs or model downloads.
.\setup.ps1 -CheckOnly

# Force CPU, including on a machine with an NVIDIA GPU.
.\setup.ps1 -TorchDevice cpu

# Require CUDA; fails validation if CUDA or bfloat16 is unavailable.
.\setup.ps1 -TorchDevice cuda

# Use a particular Python 3.11 executable when creating the environment.
.\setup.ps1 -PythonPath 'C:\Users\you\miniconda3\envs\snapprint\python.exe'

# Defer OCR downloads until the app's first startup.
.\setup.ps1 -SkipOcrModels

# Import/register Qwen, but skip the inference request.
.\setup.ps1 -SkipSmokeTest

# Set up only OCR; Word summarization still needs a later full setup.
.\setup.ps1 -SkipLlm

# Override the saved OCR device for one launch.
.\run-snapprint.ps1 -Device cpu
```

`-CheckOnly` reports presence and server availability, not readiness for inference.
The skip options deliberately leave parts of the normal full validation undone.
`-SkipSmokeTest` still checks model registration and Word export.

If winget is unavailable, install Python 3.11 and Ollama manually from their
official sites, reopen PowerShell, and rerun setup. Installers may request
Windows permissions or show their own prompts. A newer Ollama with Qwen3.5
support is needed; if model import fails on an old installation, update Ollama.

If automatic GPU selection fails, install a driver compatible with the pinned
CUDA 12.8 build, or rerun with `-TorchDevice cpu`. CPU OCR uses float32 and can
require substantial memory and time. Successful model loading is not a camera,
OCR accuracy, or full-session performance test.

## Optional HOG filter

The repository includes CNN weights and the trained HOG model at
`hog/models/hog_svm.joblib`. Setup checks that available weights load. Training
data is not included. You can replace the HOG weights or retrain them if needed.

To install a HOG model supplied by a trusted project contributor:

```powershell
.\setup.ps1 -HogModelPath 'C:\Downloads\hog_svm.joblib'
```

This replaces `hog/models/hog_svm.joblib` and checks that it loads.
Joblib models execute Python during loading, so use files from a trusted source.
The scikit-learn version should match the version used to train those weights.

To train from the project's COCO-Text dataset, first provide the annotations at
`cnn/datasets/raw/cocotext.v2.json` and prepare the images using the existing
dataset tool (this may download a large dataset):

```powershell
.\.venv-vl\Scripts\python.exe -m pip install datasets
.\.venv-vl\Scripts\python.exe -m cnn.prepareDataset
.\setup.ps1 -TrainHog
```

`-TrainHog` requires the annotation JSON and populated `cnn/datasets/text` and
`cnn/datasets/noText` directories. It builds patches, trains and tunes the SVM,
and loads the resulting model. Training takes additional time; its quality
depends on the training and validation data. Do not combine it with
`-HogModelPath`. Training data, environments, and downloaded OCR/LLM models stay out of Git.

## References

- [Ollama on Windows](https://docs.ollama.com/windows)
- [Importing GGUF models](https://docs.ollama.com/import)
- [Python Windows downloads](https://www.python.org/downloads/windows/)
- [Qwen provenance, checksum, license, and limitations](LLM_SETUP.md)
