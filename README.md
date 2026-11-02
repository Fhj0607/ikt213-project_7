# SnapPrint

SnapPrint is a local Python desktop application for printed and handwritten text recognition using PaddleOCR-VL, with spoken camera guidance.

## Requirements

- Windows
- Miniconda or Anaconda
- Python 3.11
- NVIDIA GPU and a CUDA 12.8-compatible driver for the camera app (tested on an RTX 4050 Laptop GPU, 6 GB)

## Installation

### 1. Install Miniconda

If Conda is not already installed, download and install Miniconda:

https://www.anaconda.com/docs/getting-started/miniconda/install

After installation, open **Miniconda Prompt** or **Anaconda Prompt**.

### 2. Clone or Download SnapPrint

Clone the repository:

```bash
git clone <repository-url>
cd SnapPrint
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

Install the required packages:

```bash
python -m pip install --no-compile -r requirements-gpu.txt
python -m pip install --no-compile -r requirements.txt
```

The GPU packages are several gigabytes. First recognition also downloads model weights; subsequent recognition runs locally using cached models.

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

Recognized pages accumulate in order. **End session** stops capture and speech, keeps completed text visible, and prepares an in-memory `SessionController.completed_session` payload (`pages`, `text`, `formatting_status`) for future local LLM integration. Document creation and formatting are not implemented. Ending during OCR discards the unfinished page. Session data is not saved to disk; starting and ending another session replaces the previous payload.

- **S:** start a session.
- **L / P:** choose live capture / photos and uploads.
- **Space:** stop spoken guidance.
- **C:** scan the next page in live mode, or take a photo in manual mode.
- **U:** choose an image in manual mode.
- **Esc:** end the session, including while OCR is processing.
- **2:** toggle single-page/two-page mode. Align the page divide with the image center; pages are read left then right.

Keep PyCharm's interpreter set to the `snapprint` Conda environment. At startup, recognition models load on a background worker with visible loading stages. Model loading can take over a minute. Session controls become available when initialization completes; a loading failure offers a retry. Models stay loaded across scans and sessions, and are released when the application exits. Ending a session stops capture and suppresses late results; an inference already running completes before the shared worker processes a new session.

Run checks with `python -m unittest discover -s tests -v`. Run the simulated-camera UI check with `python tests/ui_smoke.py`.

See [PaddleOCR-VL setup and validation](PADDLEOCR_VL.md) for saved-image testing, memory settings, and limitations.

## Experimental live text gate

On `test/live-text-gate`, automatic live capture checks stability and then runs the CPU-only `PP-OCRv5_mobile_det` text detector on a copy capped at 960 pixels on the longest side. Detection runs on a worker, at most once per 500 ms camera check, with no overlapping detector jobs. Full OCR recognition is skipped at this stage.

At least one text region with score >= 0.55 and area >= 16 pixels squared must appear in two consecutive stable checks. A negative result, a changed page, or instability when detection returns resets confirmation. Transient movement during a detector job does not invalidate it if the same page is stable again when the result returns. VL receives the original full-resolution candidate, then scanning pauses until **Scan next page / C**. These are initial tuning values, not measured accuracy guarantees.

**Capture now / F** bypasses the detector for handwriting or faint text it misses. Photos and uploaded images also bypass it. If detection fails, automatic capture remains blocked and manual capture stays available. Initial startup may download detector weights; later launches use the PaddleX cache under `.models`. Detector loading now happens during application startup. Ending a session suppresses pending results, but cannot interrupt model inference already running.

Tune `TextGate` defaults in `ocr/text_gate.py`. Validate latency and missed detections on real camera samples before relying on this gate. `python tests/live_gate_smoke.py` tests automatic capture with the real CPU detector, a saved printed page as the camera feed, and mocked VL recognition. In the local check, initial detector loading plus the first two detections triggered VL in about 38 seconds; an already-loaded detector checked a saved handwriting image in about 2 seconds. The UI displays detector loading/checking status during this wait.

Session lifecycle, mode selection, shortcut actions, upload eligibility, and the future formatting payload live in `controller/sessionController.py`. Camera capture, text detection, OCR, page mode, and speech policy live in `controller/snapPrintController.py`. `main.py` wires UI events and owns the file picker; `ui/mainScreen.py` owns presentation and styling.

`ocr/model_service.py` owns the application-lifetime detector, VL reader, and serial model worker. Initialization, inference, and final cleanup share that worker so sessions cannot use or unload the same models concurrently.
