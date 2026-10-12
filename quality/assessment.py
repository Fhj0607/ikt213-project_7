import cv2
import numpy as np

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

    def check_stability(self, gray):

        # resize for faster comparison
        current = cv2.resize(gray, (320, 240))

        if self.previous_frame is None:
            self.previous_frame = current
            return False

        # calculates difference between consecutive frames
        difference = cv2.absdiff(current, self.previous_frame)
        mean_difference = np.mean(difference)

        self.previous_frame = current

        if mean_difference < STABILITY_THRESHOLD:
            self.stable_count += 1
        else:
            self.stable_count = 0

        return self.stable_count >= REQUIRED_STABLE_FRAMES

    def assess(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # sharpness
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()

        # brightness
        brightness = np.mean(gray)

        # contrast
        contrast = np.std(gray)

        passed = True
        reason = "Image quality acceptable"

        if sharpness < MIN_SHARPNESS:
            passed = False
            reason = "Image too blurry"

        elif brightness < MIN_BRIGHTNESS:
            passed = False
            reason = "Image too dark"

        elif brightness > MAX_BRIGHTNESS:
            passed = False
            reason = "Image too bright"

        elif contrast < MIN_CONTRAST:
            passed = False
            reason = "Insufficient contrast"

        # check stability only if other quality checks pass
        if passed:
            stable = self.check_stability(gray)

            if not stable:
                passed = False
                reason = "Waiting for stable image"

        else:
            self.reset()
            stable = False

        return {
            "passed": passed,
            "sharpness": round(float(sharpness), 2),
            "brightness": round(float(brightness), 2),
            "contrast": round(float(contrast), 2),
            "stable": stable,
            "stable_count": self.stable_count,
            "reason": reason
        }