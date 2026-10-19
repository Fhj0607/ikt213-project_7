import cv2
import numpy as np

from processing.processingLibrary import (toGrayscale, resizeImage)



MIN_SHARPNESS = 50.0
MIN_BRIGHTNESS = 50.0
MAX_BRIGHTNESS = 210.0
MIN_CONTRAST = 25.0

STABILITY_THRESHOLD = 6.0
REQUIRED_STABLE_FRAMES = 3



class QualityAssessment:

    def __init__(self):
        self.previous_frame = None
        self.stable_count = 0



    def reset(self):
        self.previous_frame = None
        self.stable_count = 0



    def checkStability(self, gray_image):

        current_frame = resizeImage(gray_image, 320, 240)

        if self.previous_frame is None:
            self.previous_frame = current_frame
            return False

        difference = cv2.absdiff(current_frame, self.previous_frame)

        mean_difference = np.mean(difference)

        self.previous_frame = current_frame

        if mean_difference < STABILITY_THRESHOLD:
            self.stable_count += 1
        else:
            self.stable_count = 0

        return self.stable_count >= REQUIRED_STABLE_FRAMES



    def assessFrame(self, frame):

        gray_image = toGrayscale(frame)

        stable = self.checkStability(gray_image)

        if not stable:
            return {
                "acceptable": False,
                "stable": False,
                "stable_count": self.stable_count,
                "reason": "Image is not stable"
            }

        sharpness_score = cv2.Laplacian(gray_image, cv2.CV_64F).var()
        brightness_score = np.mean(gray_image)
        contrast_score = np.std(gray_image)

        blurry = sharpness_score < MIN_SHARPNESS
        too_dark = brightness_score < MIN_BRIGHTNESS
        too_bright = brightness_score > MAX_BRIGHTNESS
        low_contrast = contrast_score < MIN_CONTRAST

        acceptable = (
                not blurry
                and not too_dark
                and not too_bright
        )

        return {
            "acceptable": acceptable,

            "stable": True,
            "stable_count": self.stable_count,

            "issues": {
                "blurry": blurry,
                "too_dark": too_dark,
                "too_bright": too_bright,
                "low_contrast": low_contrast,
            },

            "scores": {
                "sharpness_score": float(sharpness_score),
                "brightness_score": float(brightness_score),
                "contrast_score": float(contrast_score)
            }
        }