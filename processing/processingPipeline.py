import cv2

from processing.qualityAssessment import assessImage
from processing.processingLibrary import (
    toGrayscale,
    sharpen,
    applyClahe,
    brighten,
    darken,
    alignImage
)


def processImage(image, issues=None, enhance=True):

    processed_image = image.copy()
    report = []

    processed_image, page_found = alignImage(processed_image)

    if page_found:
        report.append("Perspective-corrected and deskewed page")
    else:
        report.append("No page border found, only deskewed")

    if not enhance:
        return processed_image, report

    if issues is None:
        issues = assessImage(processed_image)["issues"]

    if issues["too_dark"]:
        processed_image = brighten(processed_image)
        report.append("Brightened image")

    elif issues["too_bright"]:
        processed_image = darken(processed_image)
        report.append("Darkened image")

    if issues["blurry"] or issues["low_contrast"]:
        processed_image = toGrayscale(processed_image)

    if issues["blurry"]:
        processed_image = sharpen(processed_image)
        report.append("Sharpened image")

    if issues["low_contrast"]:
        processed_image = applyClahe(processed_image)
        report.append("Applied CLAHE")

    if processed_image.ndim == 2:
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)

    return processed_image, report
