# T4 — Live text-gate comparison

Run `python -m tests.T4.masterDetectorBenchmark` from the repository root. It compares HOG + SVM, CNN, and the PaddleOCR text detector on the same held-out images and prints their classification and timing results. Prepare the excluded `cnn/datasets/text` and `cnn/datasets/noText` directories first. Model loading and one warmup image are excluded from per-image timing.
