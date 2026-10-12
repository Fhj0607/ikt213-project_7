import cv2
from difflib import SequenceMatcher

from processing.preprocessing import ToGrayscale
from processing.analysis import ContrastScore
from ocr.paddle_ocr import ExtractText


image_path = "images/demo.png"

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {image_path}"
    )

gray = ToGrayscale(image)

baseline_text = ExtractText(gray)

contrast_levels = [1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1]

print(
    f"{'Alpha':<8}"
    f"{'Score':<12}"
    f"{'OCR similarity':<16}"
)

print("-" * 36)

for alpha in contrast_levels:
    beta = 128 * (1 - alpha)

    test_image = cv2.convertScaleAbs(
        gray,
        alpha=alpha,
        beta=beta
    )

    score = ContrastScore(test_image)

    text = ExtractText(test_image)

    similarity = SequenceMatcher(None, baseline_text, text).ratio()

    print(
        f"{alpha:<8.2f}"
        f"{score:<12.2f}"
        f"{similarity:<16.3f}"
    )

    filename = f"results/contrast_{alpha:.1f}.png"

    cv2.imwrite(filename, test_image)