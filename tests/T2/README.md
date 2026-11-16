# T2 — OCR-VL benchmark

Run `python -m tests.T2.vl_benchmark` from the repository root. The test requires three images: the included generated printed page and two-page spread, plus IAM form `a01-000u` as a local handwriting image. Place the handwriting image at `tests/T2/data/iam_000_a01-000u_handwritten.png`, or pass `--handwritten PATH`. The IAM image is ignored by Git because redistribution permission has not been confirmed.
