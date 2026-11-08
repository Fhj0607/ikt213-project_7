from pathlib import Path

import cv2
import joblib
import numpy as np

from hog.hogFeatures import (
    PATCH_SIZE,
    WINDOW_STRIDE,
    hogFeatures,
    prepareFrame,
    slidingWindows
)

DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "hog_svm.joblib"

class HogTextGate:

    def __init__(self, model_path=DEFAULT_MODEL, probability_threshold=None, min_region_windows=None):
        self.model_path = Path(model_path)

        # None = use the values tuned on the validation set during training
        self.probability_threshold = probability_threshold
        self.min_region_windows = min_region_windows

        self.model = None

    def load(self):
        if self.model is not None:
            return

        if not self.model_path.exists():
            raise RuntimeError(f"HOG model not found at {self.model_path}. Run python -m hog.trainHog first.")

        saved = joblib.load(self.model_path)

        self.model = saved["model"]

        if self.probability_threshold is None:
            self.probability_threshold = saved.get("probability_threshold", 0.5)

        if self.min_region_windows is None:
            self.min_region_windows = saved.get("min_region_windows", 12)


    def predictPatches(self, patches):
        """Probability of text for each (PATCH_SIZE x PATCH_SIZE) grayscale patch."""
        self.load()

        return self.model.predict_proba(hogFeatures(patches))[:, 1]


    def textMap(self, frame):
        """Text probability for every sliding window, as a (rows, cols) grid."""
        gray = prepareFrame(frame)

        windows = slidingWindows(gray)

        rows, cols = windows.shape[:2]

        probabilities = self.predictPatches(windows.reshape(-1, PATCH_SIZE, PATCH_SIZE))

        return probabilities.reshape(rows, cols), gray

    def textWindows(self, probabilities):
        """Threshold the grid, then drop isolated hits.

        Text runs horizontally, so a morphological opening with a 1x2 kernel
        keeps only windows that have a text neighbour beside them.
        """
        binary = (probabilities >= self.probability_threshold).astype(np.uint8)

        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((1, 2), np.uint8))

    def regionSizes(self, probabilities):
        """Connected groups of text windows: label image and size of each group."""
        windows = self.textWindows(probabilities)

        count, labels, stats, _ = cv2.connectedComponentsWithStats(windows, connectivity=8)
        return labels, stats[1:, cv2.CC_STAT_AREA]

    def largestRegion(self, probabilities):
        _, sizes = self.regionSizes(probabilities)

        return int(sizes.max()) if len(sizes) else 0



    def textRegions(self, probabilities):
        """Keep only connected groups of text windows that are big enough.

        Real text forms connected blocks in the window grid, while false hits
        on clutter are scattered small groups.
        """
        labels, sizes = self.regionSizes(probabilities)

        keep_labels = np.flatnonzero(sizes >= self.min_region_windows) + 1

        return np.isin(labels, keep_labels).astype(np.uint8)

    def detect(self, frame):
        self.load()

        probabilities, _ = self.textMap(frame)

        return self.largestRegion(probabilities) >= self.min_region_windows


    def drawDetections(self, frame):
        """Frame (scaled like the detector sees it) with text regions shaded green."""
        self.load()

        probabilities, gray = self.textMap(frame)

        windows = self.textRegions(probabilities)

        output = cv2.resize(frame, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_AREA)

        mask = np.zeros(gray.shape, np.uint8)

        for row, col in np.argwhere(windows):
            y, x = row * WINDOW_STRIDE, col * WINDOW_STRIDE
            mask[y:y + PATCH_SIZE, x:x + PATCH_SIZE] = 255

        overlay = output.copy()
        overlay[mask > 0] = (0, 200, 0)

        output = cv2.addWeighted(overlay, 0.35, output, 0.65, 0)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, (0, 160, 0), 2)

        verdict = "TEXT" if windows.any() else "NO TEXT"

        cv2.putText(output, f"HOG+SVM: {verdict}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return output

    def close(self):
        self.model = None