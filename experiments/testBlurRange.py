import cv2
from difflib import SequenceMatcher

from processing.preprocessing import ToGrayscale
from processing.analysis import BlurScore
from ocr.paddle_ocr import ExtractText


image_path = "images/demo.png"

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {image_path}"
    )

gray = ToGrayscale(image)

baseline_text = ExtractText(gray)

blur_levels = [
    (1, 1),
    (3, 3),
    (5, 5),
    (7, 7),
    (9, 9),
    (15, 15),
    (21, 21)
]

print(
    f"{'Kernel':<12}"
    f"{'Blur score':<14}"
    f"{'OCR similarity':<16}"
)

print("-" * 42)

for kernel in blur_levels:

    if kernel == (1, 1):
        test_image = gray.copy()
    else:
        test_image = cv2.GaussianBlur(
            gray,
            kernel,
            0
        )

    score = BlurScore(test_image)

    text = ExtractText(test_image)

    similarity = SequenceMatcher(
        None,
        baseline_text,
        text
    ).ratio()

    kernel_name = f"{kernel[0]}x{kernel[1]}"

    print(
        f"{kernel_name:<12}"
        f"{score:<14.2f}"
        f"{similarity:<16.3f}"
    )

    print("\nOCR OUTPUT:")
    print(text)
    print("\n" + "=" * 80 + "\n")

    filename = (
        f"results/blur_{kernel[0]}x{kernel[1]}.png"
    )

    cv2.imwrite(
        filename,
        test_image
    )