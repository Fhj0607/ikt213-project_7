import argparse
import statistics
import time

import cv2

from cnn.trainTextCNN import loadDataset
from cnn.textClassifier import CNNTextGate
from hog.hogTextGate import HogTextGate
from ocr.text_gate import TextGate

MODEL_NAMES = ("HOG + SVM", "CNN", "PaddleOCR")

def metrics(rows):
    tp = sum(r["actual"] == 1 and r["predicted"] == 1 for r in rows)
    tn = sum(r["actual"] == 0 and r["predicted"] == 0 for r in rows)
    fp = sum(r["actual"] == 0 and r["predicted"] == 1 for r in rows)
    fn = sum(r["actual"] == 1 and r["predicted"] == 0 for r in rows)
    durations = [r["elapsed_ms"] for r in rows]

    def div(n, d):
        return n / d if d else None

    return {
        "images": len(rows), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": div(tp + tn, len(rows)),
        "precision": div(tp, tp + fp),
        "recall": div(tp, tp + fn),
        "specificity": div(tn, tn + fp),
        "f1": div(2 * tp, 2 * tp + fp + fn),
        "skipped": div(tn + fn, len(rows)),
        "total_seconds": sum(durations) / 1000,
        "mean_ms": statistics.mean(durations),
        "median_ms": statistics.median(durations),
        "p95_ms": sorted(durations)[min(len(durations) - 1, int(0.95 * len(durations)))]
    }

def evaluate(name, gate, images):
    print(f"\nLoading {name}...", flush=True)
    start = time.perf_counter()
    gate.load()
    load_seconds = time.perf_counter() - start

    warmup = cv2.imread(str(images[0][0]))
    if warmup is None:
        raise ValueError(f"Cannot read warmup image: {images[0][0]}")
    gate.detect(warmup)

    rows = []
    for index, (path, actual) in enumerate(images, start=1):
        start = time.perf_counter()
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read test image: {path}")
        prediction = int(gate.detect(image))
        elapsed_ms = (time.perf_counter() - start) * 1000
        rows.append({"image": str(path), "actual": int(actual),
                     "predicted": prediction, "elapsed_ms": elapsed_ms})
        if index % 100 == 0 or index == len(images):
            print(f"{name}: {index}/{len(images)}", flush=True)
    return rows, load_seconds

def pct(value):
    return "n/a" if value is None else f"{value * 100:.2f}%"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="First N held-out test images (smoke test)")
    parser.add_argument("--hog-threshold", type=float, default=None,
                        help="Override HOG patch threshold (default: tuned model value)")
    parser.add_argument("--hog-min-windows", type=int, default=None,
                        help="Override HOG minimum region size (default: tuned model value)")
    parser.add_argument("--cnn-threshold", type=float, default=0.2)
    parser.add_argument("--paddle-threshold", type=float, default=0.55)
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    if any(v is not None and not 0 <= v <= 1
           for v in (args.hog_threshold, args.cnn_threshold, args.paddle_threshold)):
        parser.error("thresholds must be in [0, 1]")
    if args.hog_min_windows is not None and args.hog_min_windows < 1:
        parser.error("--hog-min-windows must be >= 1")

    train, validation, test = loadDataset()
    if args.limit is not None:
        test = test[:args.limit]
    if not test:
        parser.error("No test images found under cnn/datasets/text or cnn/datasets/noText")

    print(f"Dataset: {len(train)} training, {len(validation)} validation, {len(test)} test")
    print("Same held-out test images and order for all detectors.")
    print("Per-image time includes cv2.imread and detect; model loading is measured separately.")

    gates = {
        "HOG + SVM": HogTextGate(probability_threshold=args.hog_threshold,
                                 min_region_windows=args.hog_min_windows),
        "CNN": CNNTextGate(threshold=args.cnn_threshold),
        "PaddleOCR": TextGate(score_threshold=args.paddle_threshold),
    }
    runs, summaries, loads = {}, {}, {}
    hog_settings = {}
    try:
        for name in MODEL_NAMES:
            rows, load_seconds = evaluate(name, gates[name], test)
            runs[name] = rows
            summaries[name] = metrics(rows)
            loads[name] = load_seconds
            if name == "HOG + SVM":
                hog = gates[name]
                hog_settings = {"probability_threshold": float(hog.probability_threshold),
                                "min_region_windows": int(hog.min_region_windows)}
    finally:
        for gate in gates.values():
            gate.close()

    print(f"\nThresholds: HOG patch={hog_settings['probability_threshold']}, "
          f"HOG min region={hog_settings['min_region_windows']} windows, "
          f"CNN={args.cnn_threshold}, PaddleOCR={args.paddle_threshold}")
    print("\n--- MASTER DETECTOR BENCHMARK ---")
    print(f"{'Model':16} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Rejection':>11} {'ms/image':>10}")
    for name in MODEL_NAMES:
        s = summaries[name]
        print(f"{name:16} {pct(s['accuracy']):>10} {pct(s['precision']):>10} "
              f"{pct(s['recall']):>10} {pct(s['f1']):>10} {pct(s['specificity']):>11} {s['mean_ms']:10.2f}")
        print(f"  TP={s['tp']} TN={s['tn']} FP={s['fp']} FN={s['fn']} Load={loads[name]:.2f}s")
    print("\n--- Timing (model loading excluded from inference) ---")
    for name in MODEL_NAMES:
        s = summaries[name]
        print(f"{name:16} load={loads[name]:.2f}s  total={s['total_seconds']:.2f}s  "
              f"mean={s['mean_ms']:.2f}ms  median={s['median_ms']:.2f}ms  "
              f"p95={s['p95_ms']:.2f}ms")

if __name__ == "__main__":
    main()
