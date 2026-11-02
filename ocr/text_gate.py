"""Lazy CPU text detection for live capture; never performs recognition."""
from pathlib import Path
import os

import cv2


class TextGate:
    def __init__(self, score_threshold=0.55, min_regions=1, max_side=960):
        self.score_threshold = score_threshold
        self.min_regions = min_regions
        self.max_side = max_side
        self.model = None

    def load(self):
        if self.model is None:
            root = Path(__file__).resolve().parents[1]
            os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(root / ".models" / "paddlex"))
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
            from paddleocr import TextDetection
            self.model = TextDetection(model_name="PP-OCRv5_mobile_det", device="cpu",
                                       enable_mkldnn=False)

    def detect(self, frame):
        self.load()
        height, width = frame.shape[:2]
        scale = min(1.0, self.max_side / max(height, width))
        image = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
        regions = 0
        for result in self.model.predict(image, batch_size=1, limit_side_len=self.max_side,
                                         limit_type="max", box_thresh=self.score_threshold):
            for polygon, score in zip(result["dt_polys"], result["dt_scores"]):
                if float(score) >= self.score_threshold and abs(cv2.contourArea(polygon.astype("float32"))) >= 16:
                    regions += 1
        return regions >= self.min_regions

    def close(self):
        if self.model is not None:
            self.model.close()
            self.model = None
