from processing.processingLibrary import (
    toGrayscale,
    sharpen,
    applyClahe,
    brighten,
    darken,
    alignImage
)



def processImage(image, issues):

    processed_image = image.copy()
    report = []

    processed_image, page_found = alignImage(processed_image)

    if page_found:
        report.append("Perspective-corrected and deskewed page")
    else:
        report.append("No page border found, only deskewed")

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

    return processed_image, report