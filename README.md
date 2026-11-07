# SnapPrint

SnapPrint is a local Python desktop application for printed and handwritten text recognition using PaddleOCR-VL, with spoken camera guidance.

## Requirements

- Windows (the desktop app currently uses Windows-specific speech support).
- Python 3.11 and Miniconda or Anaconda for environment management.
- For the desktop app: a CUDA-capable NVIDIA GPU supported by the installed PyTorch build and the application's bfloat16 inference settings. A specific GPU model is not required.
- A camera for live capture or taking photos. Uploaded images do not require a camera.
- Internet access for dependency installation and the initial model downloads, plus disk space for the environment and model weights.

Memory use and recognition speed depend on image resolution, document complexity, and inference settings. No general minimum RAM or VRAM requirement has been established. Hardware measurements in [PADDLEOCR_VL.md](PADDLEOCR_VL.md) describe one tested configuration, not a required specification.

CPU-only saved-image recognition is also available through `vl_scan.py --device cpu`; the desktop app currently defaults to `gpu:0` and does not automatically fall back to CPU. AMD/Intel GPU and macOS/Linux desktop support have not been validated.

## Installation

### 1. Install Miniconda

If Conda is not already installed, download and install Miniconda:

https://www.anaconda.com/docs/getting-started/miniconda/install

After installation, open **Miniconda Prompt** or **Anaconda Prompt**.

### 2. Clone or Download SnapPrint

Clone the repository:

```bash
git clone --branch fabian https://github.com/Fhj0607/ikt213-project_7.git
cd ikt213-project_7
```

Alternatively, download the repository as a ZIP file and extract it.

Then navigate to the extracted SnapPrint directory.

### 3. Create the Conda Environment

Create a new environment using Python 3.11:

```bash
conda create -n snapprint python=3.11
```

Activate it:

```bash
conda activate snapprint
```

### 4. Install Dependencies

With the environment activated, install PyTorch and TorchVision using the command from the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). Choose Windows, Pip, and a CUDA build compatible with your GPU and driver. Use `python -m pip` in the active environment. Keep Torch and TorchVision on matching releases; the versions used for local validation are recorded in `requirements-gpu.txt`.

For machines compatible with the repository's pinned CUDA 12.8 build, this shortcut reproduces the tested PyTorch installation:

```bash
python -m pip install --no-compile -r requirements-gpu.txt
```

That file is an optional tested configuration, not a universal GPU requirement. If you selected a different compatible PyTorch build, skip the shortcut above.

Then install the CPU PaddlePaddle runtime used by the lightweight text detector and the application dependencies:

```bash
python -m pip install numpy opencv-python paddlepaddle
python -m pip install --no-compile -r requirements.txt
```

`requirements.txt` includes PaddleOCR's document-parser dependencies for OCR-VL; it does not create an environment or install your selected PyTorch build. Its pinned application-library versions provide a reproducible baseline; alternative PyTorch/CUDA combinations need validation on the target machine.

For the desktop app, verify that PyTorch can access the GPU:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

The result should be `True`. If it is `False`, check the installed PyTorch build and GPU driver before launching the desktop app.

For CPU-only saved-image recognition, choose the CPU build in the PyTorch selector, install the same application dependencies, then run:

```bash
python vl_scan.py path/to/page.jpg --device cpu
```

CPU recognition is expected to be slower and still requires enough system memory for the models. It has not been benchmarked as a supported desktop configuration.

The environment and model weights are not included in Git. The application downloads missing model weights during first startup and caches them in `.models/`; later launches reuse them locally. Startup shows loading stages and keeps session controls disabled until the models are ready. Download and loading times vary by connection and hardware.

## Running SnapPrint

Navigate to the SnapPrint directory and make sure the environment is active:

```bash
conda activate snapprint
```

Then start the application:

```bash
python main.py
```

## Running SnapPrint Again

After the initial installation, you only need to open a Conda prompt, navigate to the SnapPrint directory, and run:

```bash
conda activate snapprint
python main.py
```

## Session menu and controls

Start a session, then choose **Live capture** for automatic page capture with spoken positioning guidance, or **Take photos or upload images** for manual OCR input. Uploads are processed locally and do not require a camera. Photos/uploads mode is silent, including OCR status and page-mode changes; spoken guidance is reserved for live capture. In photo mode, the first **Take photo** opens the preview; press it again to capture a page.

Recognized pages accumulate in order. **End session** stops capture and speech and keeps completed text visible. Choose **Create Word document** to summarize the collected text locally with Ollama and save a consistently formatted `.docx` under `results/documents/`. The saved path appears in the app. The document contains a title, Summary, Key points, Detailed notes, Unclear passages, and an Original OCR text appendix. It is a generated draft for review. Ending during OCR discards the unfinished page. Session text stays in memory until replaced by a later session; exported documents remain on disk. See [local summarization setup](LLM_SETUP.md) for model setup and CLI usage.

- **S:** start a session.
- **L / P:** choose live capture / photos and uploads.
- **Space:** stop spoken guidance.
- **C:** scan the next page in live mode, or take a photo in manual mode.
- **U:** choose an image in manual mode.
- **Esc:** end the session, including while OCR is processing.
- **2:** toggle single-page/two-page mode. Align the page divide with the image center; pages are read left then right.

Keep PyCharm's interpreter set to the `snapprint` Conda environment. At startup, recognition models load on a background worker with visible loading stages. Model loading can take over a minute. Session controls become available when initialization completes; a loading failure offers a retry. Models stay loaded across scans and sessions, and are released when the application exits. Ending a session stops capture and suppresses late results; an inference already running completes before the shared worker processes a new session.

Development tests and generated results are kept locally and are excluded from the published repository.

See [PaddleOCR-VL setup and validation](PADDLEOCR_VL.md) for saved-image testing, memory settings, and limitations.

## Experimental live text gate

Automatic live capture checks stability and then runs the CPU-only `PP-OCRv5_mobile_det` text detector on a copy capped at 960 pixels on the longest side. Detection runs on a worker, at most once per 500 ms camera check, with no overlapping detector jobs. Full OCR recognition is skipped at this stage.

At least one text region with score >= 0.55 and area >= 16 pixels squared must appear in two consecutive stable checks. A negative result, a changed page, or instability when detection returns resets confirmation. Transient movement during a detector job does not invalidate it if the same page is stable again when the result returns. VL receives the original full-resolution candidate, then scanning pauses until **Scan next page / C**. These are initial tuning values, not measured accuracy guarantees.

**Capture now / F** bypasses the detector for handwriting or faint text it misses. Photos and uploaded images also bypass it. If detection fails, automatic capture remains blocked and manual capture stays available. Initial startup may download detector weights; later launches use the PaddleX cache under `.models`. Detector loading now happens during application startup. Ending a session suppresses pending results, but cannot interrupt model inference already running.

Tune `TextGate` defaults in `ocr/text_gate.py`. Validate latency and missed detections with representative camera samples on the target machine. The UI displays detector loading/checking status; timings measured on the development machine are not performance guarantees.

Session lifecycle, mode selection, shortcut actions, upload eligibility, and the future formatting payload live in `controller/sessionController.py`. Camera capture, text detection, OCR, page mode, and speech policy live in `controller/snapPrintController.py`. `main.py` wires UI events and owns the file picker; `ui/mainScreen.py` owns presentation and styling.

`ocr/model_service.py` owns the application-lifetime detector, VL reader, and serial model worker. Initialization, inference, and final cleanup share that worker so sessions cannot use or unload the same models concurrently.
