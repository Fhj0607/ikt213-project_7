import argparse
import random
from pathlib import Path

import cv2

from cnn.benchmark import benchmarkDetector, evaluateResults
from cnn.trainTextCNN import loadDataset
from hog.hogTextGate import HogTextGate



OVERLAY_FOLDER = Path("results/hog_overlays")
RESULTS_FILE = Path("results/hog_benchmark.txt")


def summary(name, results, total_time):
    tp = sum(1 for actual, predicted in results if actual == 1 and predicted == 1)
    tn = sum(1 for actual, predicted in results if actual == 0 and predicted == 0)
    text = sum(1 for actual, _ in results if actual == 1)

    return {
        "name": name,
        "accuracy": (tp + tn) / len(results) * 100,
        "recall": tp / text * 100,
        "rejection": tn / (len(results) - text) * 100,
        "ms": total_time / len(results) * 1000
    }



def benchmarkCnn(test_images):
    import torch

    from cnn.benchmark import MODEL_PATH, benchmarkCNN
    from cnn.trainTextCNN import createModel, prepareImage

    model = createModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()

    with torch.no_grad():
        model(prepareImage(test_images[0][0]).unsqueeze(0))

    return benchmarkCNN(model, test_images)



def benchmarkPaddle(test_images):
    from ocr.text_gate import TextGate

    detector = TextGate()
    detector.load()
    detector.detect(cv2.imread(str(test_images[0][0])))

    try:
        return benchmarkDetector(detector, test_images)
    finally:
        detector.close()



def saveOverlays(gate, test_images, count):
    OVERLAY_FOLDER.mkdir(parents=True, exist_ok=True)

    random.seed(42)

    for image_path, label in random.sample(test_images, min(count, len(test_images))):
        frame = cv2.imread(str(image_path))

        prefix = "text" if label == 1 else "noText"

        cv2.imwrite(str(OVERLAY_FOLDER / f"{prefix}_{image_path.name}"), gate.drawDetections(frame))

    print(f"Saved {count} detection images to {OVERLAY_FOLDER}")



def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--compare", action="store_true", help="also run the CNN and the PaddleOCR detector")
    parser.add_argument("--overlays", type=int, default=0, help="save N HOG detection images")
    args = parser.parse_args()

    _, _, test_images = loadDataset()

    print("Test images:", len(test_images))

    gate = HogTextGate()
    gate.load()
    gate.detect(cv2.imread(str(test_images[0][0])))

    print(f"HOG settings: probability threshold {gate.probability_threshold}, "
          f"min region {gate.min_region_windows} windows")

    print("\nBenchmarking HOG + SVM...")

    hog_results, hog_time = benchmarkDetector(gate, test_images)

    evaluateResults(hog_results, "HOG + SVM")

    table = [summary("HOG + SVM", hog_results, hog_time)]

    if args.overlays:
        saveOverlays(gate, test_images, args.overlays)

    gate.close()

    if args.compare:
        # One method failing should not throw away the others' results
        for name, benchmark in (("CNN", benchmarkCnn), ("PaddleOCR detector", benchmarkPaddle)):
            print(f"\nBenchmarking {name}...")

            try:
                results, total_time = benchmark(test_images)

            except Exception as error:
                print(f"{name} failed, skipped: {type(error).__name__}: {error}")
                continue

            evaluateResults(results, name)
            table.append(summary(name, results, total_time))

    lines = [
        f"Test images: {len(test_images)}",
        f"{'Method':20} | {'Accuracy':>8} | {'Text recall':>11} | {'No-text rejection':>17} | {'ms/image':>8}",
        "-" * 77
    ]

    for row in table:
        lines.append(f"{row['name']:20} | {row['accuracy']:7.2f}% | {row['recall']:10.2f}% | "
                     f"{row['rejection']:16.2f}% | {row['ms']:8.1f}")

    print("\n" + "\n".join(lines))

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text("\n".join(lines) + "\n")

    print(f"\nSaved {RESULTS_FILE}")

if __name__ == "__main__":
    main()