import argparse
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from hog.hogFeatures import hogFeatures
from hog.hogTextGate import DEFAULT_MODEL, HogTextGate
from hog.makeDataset import OUTPUT_FOLDER



C_VALUES = [0.001, 0.01, 0.1, 1.0]

PROBABILITY_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
REGION_SIZES = range(1, 61)



def trainPatchClassifier(data_folder):
    train = np.load(Path(data_folder) / "train.npz")

    patches, labels, groups = train["patches"], train["labels"], train["image_ids"]

    print(f"Training on {len(labels)} patches from {len(np.unique(groups))} images "
          f"({int(labels.sum())} text / {int((labels == 0).sum())} non-text)")

    features = hogFeatures(patches)

    print(f"HOG: {features.shape[1]} features per patch")

    pipeline = make_pipeline(
        StandardScaler(),
        LinearSVC(max_iter=10000)
    )

    search = GridSearchCV(
        pipeline,
        {"linearsvc__C": C_VALUES},
        cv=GroupKFold(n_splits=min(5, len(np.unique(groups)))),
        scoring="f1"
    )

    search.fit(features, labels, groups=groups)

    for c, score in zip(C_VALUES, search.cv_results_["mean_test_score"]):
        print(f"  C={c:<6} cross-validated patch F1 = {score:.3f}")

    best_c = search.best_params_["linearsvc__C"]

    model = CalibratedClassifierCV(search.best_estimator_, method="sigmoid", cv=3)
    model.fit(features, labels)

    return model, best_c



def tuneImageDecision(model_path, validation_images):
    """Pick the threshold and region size with the best validation accuracy."""
    gate = HogTextGate(model_path)
    gate.load()

    probability_maps, truth = [], []

    for index, (image_path, label) in enumerate(validation_images):
        frame = cv2.imread(str(image_path))

        if frame is None:
            continue

        probabilities, _ = gate.textMap(frame)

        probability_maps.append(probabilities)
        truth.append(label)

        if (index + 1) % 250 == 0:
            print(f"  validation {index + 1}/{len(validation_images)}")

    truth = np.array(truth)

    best = (0.0, 0.5, 12)

    for threshold in PROBABILITY_THRESHOLDS:
        gate.probability_threshold = threshold

        largest = np.array([gate.largestRegion(probabilities) for probabilities in probability_maps])

        for size in REGION_SIZES:
            accuracy = float(np.mean((largest >= size) == truth))

            if accuracy > best[0]:
                best = (accuracy, threshold, size)

    print(f"Best on validation: accuracy {best[0] * 100:.2f} % "
          f"with probability threshold {best[1]} and min region {best[2]} windows")

    return best[1], best[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(OUTPUT_FOLDER))
    parser.add_argument("--out", default=str(DEFAULT_MODEL))
    parser.add_argument("--validation-limit", type=int, default=None, help="tune on fewer images (quick test)")
    args = parser.parse_args()

    start = time.perf_counter()

    model, best_c = trainPatchClassifier(args.data)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump({"model": model, "C": best_c}, out)

    print(f"Patch classifier trained in {time.perf_counter() - start:.1f} s (C = {best_c})")

    from cnn.trainTextCNN import loadDataset

    _, validation_images, _ = loadDataset()

    if args.validation_limit:
        validation_images = validation_images[:args.validation_limit]

    print(f"Tuning the image decision on {len(validation_images)} validation images...")

    threshold, region_size = tuneImageDecision(out, validation_images)

    joblib.dump({
        "model": model,
        "C": best_c,
        "probability_threshold": threshold,
        "min_region_windows": region_size
    }, out)

    print(f"Total training time {time.perf_counter() - start:.1f} s")
    print(f"Saved {out} ({out.stat().st_size / 1024:.0f} KB)")



if __name__ == "__main__":
    main()