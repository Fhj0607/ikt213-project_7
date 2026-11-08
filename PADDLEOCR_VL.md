# Local PaddleOCR-VL setup and validation

The app uses the complete PaddleOCR-VL-1.6 pipeline, including layout
analysis, through the Transformers engine on native Windows.

## SnapPrint Conda environment

```powershell
conda activate snapprint
python -m pip install --no-compile -r requirements-gpu.txt
python -m pip install --no-compile -r requirements.txt
python main.py
```

In PyCharm, select the `snapprint` Conda interpreter. The app no longer
requires launching from `.venv-vl`; that environment remains an optional
isolated installation for comparison and troubleshooting.

## Optional isolated setup

From the project directory in PowerShell:

```powershell
conda activate snapprint
python -m venv .venv-vl
.\.venv-vl\Scripts\python.exe -m pip install --no-compile -r requirements-gpu.txt
.\.venv-vl\Scripts\python.exe -m pip install --no-compile -r requirements.txt
```

Initial installation and model downloads require internet. The models run
locally; captured images are not sent to an external recognition service.
Model caches are in `.models/`, excluded from Git. Keep this directory to
reuse the downloaded models offline.

## Saved-image test

```powershell
conda activate snapprint
python vl_scan.py path\to\page.jpg
python vl_scan.py path\to\spread.jpg --two-pages
# Optional image enhancement:
python vl_scan.py path\to\page.jpg --enhance
```

Results are saved to `results/vl-output.md`. The console reports model loading
and recognition times separately. `--device cpu` is available but will be
slower. Default GPU settings use bfloat16, one region at a time, SDPA attention,
approximately one megapixel per region, and a 2048-token limit per region.
If text gets cut off, increase `--max-tokens`; if detail is lost, increase
`--max-pixels`. Both increase memory use. Neither limit guarantees a complete
page transcription. Compare with the original before trusting the result.

## Camera app

```powershell
conda activate snapprint
python main.py
# Automated setup environment: .\run-snapprint.ps1
```

- Every captured or uploaded page is aligned before VL. Optional **Enhance
  images** is Off by default. When enabled, brightness correction, sharpening,
  and CLAHE are applied when quality thresholds indicate a problem. The enhanced
  image is sent to VL as three channels; sharpening/CLAHE uses grayscale content.
  Photos and uploads receive the same processing. Quality guidance stays separate.
- Image-classified regions are also sent through OCR. The app displays raw
  transcription content as plain text, excluding HTML image placeholders;
  a placeholder alone is treated as no readable text.
- Only one document is processed at a time, keeping the preview responsive.
- Space: stop spoken guidance. Recognized text remains editable in the notes preview.
- 2: toggle single-page/two-page mode, with spoken confirmation.
- C: scan again without having to move the camera to a different scene.
- After each capture, scanning remains paused even if the camera moves. Use
  Scan next page (C) to begin another capture; the button is disabled during OCR.
- Two-page mode splits exactly at the image center, left page first. Align
  the page divide vertically with the camera center. Each half then passes
  through page-border detection, perspective correction when a border is found,
  deskewing, and conditional image enhancement before OCR-VL recognition.
- The camera requests 1920x1080; actual resolution depends on the camera.
- Live-mode completion announces whether text was recognized and the elapsed
  time. Photos and uploads are silent. No OCR confidence percentage is invented.

For handwriting evaluation, test photographed A4 pages from multiple writers
and compare omitted lines, character errors, reading order, time, and memory.
Full-page handwriting accuracy is not established by a printed sample test.

Official guide: https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PaddleOCR-VL.html
