import cv2
import numpy as np

from processing.processingLibrary import (toGrayscale, resizeImage)

MIN_SHARPNESS = 50.0
MIN_BRIGHTNESS = 50.0
MAX_BRIGHTNESS = 210.0
MIN_CONTRAST = 25.0

STABILITY_THRESHOLD = 6.0
REQUIRED_STABLE_FRAMES = 3


def assessImage(image):
    """Measure one image without changing the live camera's stability history."""
    gray_image = image if image.ndim == 2 else toGrayscale(image)
    sharpness_score = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    brightness_score = np.mean(gray_image)
    contrast_score = np.std(gray_image)
    return {
        "issues": {
            "blurry": bool(sharpness_score < MIN_SHARPNESS),
            "too_dark": bool(brightness_score < MIN_BRIGHTNESS),
            "too_bright": bool(brightness_score > MAX_BRIGHTNESS),
            "low_contrast": bool(contrast_score < MIN_CONTRAST),
        },
        "scores": {
            "sharpness_score": float(sharpness_score),
            "brightness_score": float(brightness_score),
            "contrast_score": float(contrast_score),
        },
    }


class QualityAssessment:

    def __init__(self):
        self.previous_frame = None
        self.stable_count = 0

    def reset(self):
        self.previous_frame = None
        self.stable_count = 0

    def checkStability(self, gray_image):

        current_frame = resizeImage(
            gray_image,
            320,
            240
        )

        if self.previous_frame is None:
            self.previous_frame = current_frame
            return False, 0.0

        difference = cv2.absdiff(
            current_frame,
            self.previous_frame
        )

        mean_difference = np.mean(difference)

        self.previous_frame = current_frame

        if mean_difference < STABILITY_THRESHOLD:
            self.stable_count += 1
        else:
            self.stable_count = 0

        stable = (
                self.stable_count
                >= REQUIRED_STABLE_FRAMES
        )

        return stable, float(mean_difference)

    def assessFrame(self, frame):
        gray_image = toGrayscale(frame)
        assessment = assessImage(gray_image)
        stable, difference_score = self.checkStability(gray_image)
        assessment.update(acceptable=stable, stable=stable, stable_count=self.stable_count)
        assessment["scores"]["difference_score"] = difference_score
        return assessment
