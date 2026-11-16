from pathlib import Path

import cv2
import numpy as np

from processing.processingPipeline import processImage


OUTPUT_DIR = Path(__file__).resolve().parent / "results"


def generate_document():
    rng = np.random.default_rng(7)
    background = np.full((900, 1200, 3), (73, 83, 91), dtype=np.uint8)
    noise = rng.integers(-8, 9, (900, 1200, 1), dtype=np.int16)
    background = np.clip(background.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    page = np.full((900, 700, 3), (244, 247, 249), dtype=np.uint8)
    cv2.putText(page, "PAGE ALIGNMENT TEST", (55, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (35, 38, 42), 2, cv2.LINE_AA)
    cv2.line(page, (55, 135), (645, 135), (95, 99, 104), 2)
    lines = [
        "This page is tilted in the image.",
        "The four corners should be found.",
        "Perspective correction should make",
        "the text lines horizontal again.",
        "Check the before and after files.",
    ]
    for index, line in enumerate(lines):
        cv2.putText(page, line, (55, 220 + 95 * index),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (43, 46, 50), 2, cv2.LINE_AA)
    cv2.rectangle(page, (55, 745), (645, 825), (90, 95, 100), 2)
    cv2.putText(page, "Sample document", (85, 795),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (43, 46, 50), 2, cv2.LINE_AA)

    source = np.float32([[0, 0], [699, 0], [699, 899], [0, 899]])
    corners = np.float32([[215, 120], [935, 70], [1020, 820], [145, 785]])
    transform = cv2.getPerspectiveTransform(source, corners)
    warped = cv2.warpPerspective(page, transform, (1200, 900))
    mask = cv2.warpPerspective(np.full((900, 700), 255, dtype=np.uint8),
                               transform, (1200, 900))
    background[mask > 0] = warped[mask > 0]
    return background


def main():
    original = generate_document()
    aligned, steps = processImage(original, enhance=False)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    before = OUTPUT_DIR / "before.png"
    after = OUTPUT_DIR / "after.png"
    if not cv2.imwrite(str(before), original) or not cv2.imwrite(str(after), aligned):
        raise OSError("Could not save before/after images")
    print(f"Before: {before.resolve()} ({original.shape[1]}x{original.shape[0]})")
    print(f"After:  {after.resolve()} ({aligned.shape[1]}x{aligned.shape[0]})")
    print(f"Result: {'; '.join(steps)}")


if __name__ == "__main__":
    main()
