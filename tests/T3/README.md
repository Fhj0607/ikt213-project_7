# T3 — CNN threshold sweep

Run `python -m tests.T3.sweepThresholds` from the repository root. It prints accuracy, text recall, no-text rejection, and false negatives for thresholds 0.1 through 0.9 on the held-out COCO-Text image set. The dataset is excluded from Git, so prepare `cnn/datasets/text` and `cnn/datasets/noText` first.
