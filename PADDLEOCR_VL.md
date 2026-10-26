# Local PaddleOCR-VL setup and validation

Integrated on `fabian` after testing on `experiment/paddleocr-vl`.
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
# Optional isolated environment: .\run-vl.ps1
```

- Quality warnings and capture run as before; VL uses the original color frame.
- Image-classified regions are also sent through OCR. The app displays raw
  transcription content as plain text, excluding HTML image placeholders;
  a placeholder alone is treated as no readable text.
- Only one document is processed at a time, keeping the preview responsive.
- Space: stop speech. R: read the latest recognized text.
- The Read text button does the same as R. During processing, it says to wait;
  without a result, it says there is no text and asks you to scan again.
- 2: toggle single-page/two-page mode, with spoken confirmation.
- C: scan again without having to move the camera to a different scene.
- After each capture, scanning remains paused even if the camera moves. Use
  Scan again (C) to begin another capture; the button is disabled during OCR.
- Two-page mode splits exactly at the image center, left page first. Align
  the page divide vertically with the camera center; automatic page detection
  and perspective correction are not implemented here.
- The camera requests 1920x1080; actual resolution depends on the camera.
- Completion announces that text was recognized, the elapsed time, and that R
  reads the text aloud. VL has no equivalent of the previous line confidence
  scores, so the app does not invent a percentage.

For handwriting evaluation, test photographed A4 pages from multiple writers
and compare omitted lines, character errors, reading order, time, and memory.
Full-page handwriting accuracy is not established by a printed sample test.

## Local validation results

Tested on the RTX 4050 Laptop GPU (6 GB) with 16 GB system RAM:

| Input | Recognition time | Observed result |
| --- | --- | --- |
| Generated printed page | 14.0 s | All five sections recovered in order |
| Seven-line handwritten IAM sample | 16.6 s | All seven lines recovered; at least one word substitution |
| Two identical printed pages side by side | 15.0 s | Both pages recovered, left then right |

The first measured cold model load took 83.2 seconds. Recognition times exclude
model loading; the camera app keeps the loaded model for subsequent captures.
Peak PyTorch GPU allocation during the handwriting/spread run was 2.57 GiB;
this is not total device memory usage. These are individual smoke tests, not
an accuracy benchmark or a latency guarantee. Webcam photography and two
handwritten pages still need manual validation.

The integrated `snapprint` Conda environment also passed all 11 unit tests,
the simulated-camera UI/scan-control check, and the handwriting recognition
test (18.5 seconds after a 146.2-second first load in that environment).

The handwritten test image is from the public
[IAM sample collection](https://huggingface.co/datasets/kenza-ily/iam_disco),
file `images/handwritten/iam_000_a01-000u_handwritten.png`. Generated fixtures,
downloaded samples, output Markdown, and `vl-validation.json` live in the
ignored `results/` directory. They are not committed to the repository.

Official guide: https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PaddleOCR-VL.html
