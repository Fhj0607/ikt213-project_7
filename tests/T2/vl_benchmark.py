import argparse
import time
from pathlib import Path

import cv2
import torch

from ocr.vl_reader import VLReader, readable_text
from processing.processingPipeline import processImage


TEST_DIR = Path(__file__).resolve().parent
DATA_DIR = TEST_DIR / "data"
OUTPUT_DIR = TEST_DIR / "results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--handwritten", type=Path,
        default=DATA_DIR / "iam_000_a01-000u_handwritten.png",
        help="local IAM handwriting image for this test"
    )
    parser.add_argument("--device", choices=("gpu:0", "cpu"))
    args = parser.parse_args()

    inputs = (
        ("printed", DATA_DIR / "printed_five_sections.png"),
        ("handwritten", args.handwritten),
        ("spread", DATA_DIR / "printed_spread.png"),
    )
    images = []
    for name, path in inputs:
        image = cv2.imread(str(path))
        if image is None:
            parser.error(f"Cannot read {name} image: {path}")
        images.append((name, image))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reader = VLReader(args.device)
    times = []
    try:
        reader.load()
        for name, image in images:
            if name == "spread":
                middle = image.shape[1] // 2
                pages = (image[:, :middle].copy(), image[:, middle:].copy())
            else:
                pages = (image,)
            pages = [processImage(page, enhance=False)[0] for page in pages]

            seconds = 0.0
            page_texts = []
            for page in pages:
                if reader.device.startswith("gpu"):
                    torch.cuda.synchronize()
                started = time.perf_counter()
                results = list(reader.pipeline.predict(
                    page,
                    max_pixels=reader.max_pixels,
                    max_new_tokens=reader.max_tokens,
                    use_queues=False,
                    use_ocr_for_image_block=True,
                ))
                if reader.device.startswith("gpu"):
                    torch.cuda.synchronize()
                seconds += time.perf_counter() - started
                blocks = [
                    readable_text(block.content)
                    for result in results
                    for block in result["parsing_res_list"]
                ]
                page_texts.append("\n\n".join(text for text in blocks if text))

            recognized = "\n\n".join(page_texts).strip()
            transcript = OUTPUT_DIR / f"{name}.txt"
            transcript.write_text(recognized + "\n", encoding="utf-8")
            times.append(f"{name}: {seconds:.3f} s")
            print(f"{name}: {seconds:.3f} s -> {transcript}")
    finally:
        reader.close()

    (OUTPUT_DIR / "times.txt").write_text("\n".join(times) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
