"""Read a saved image locally without starting the camera or speech UI."""

import argparse
from pathlib import Path
import time

import cv2

from ocr.vl_reader import VLReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--device', default='gpu:0', choices=['gpu:0', 'cpu'])
    parser.add_argument('--two-pages', action='store_true')
    parser.add_argument('--max-pixels', type=int, default=1003520)
    parser.add_argument('--max-tokens', type=int, default=2048)
    parser.add_argument('--output', type=Path, default=Path('results/vl-output.md'))
    args = parser.parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        parser.error(f'Cannot read image: {args.image}')
    reader = VLReader(args.device, args.max_pixels, args.max_tokens)
    try:
        start = time.perf_counter()
        print('Loading local PaddleOCR-VL; first run downloads model weights.', flush=True)
        reader.load()
        print(f'Model load: {time.perf_counter() - start:.1f} seconds', flush=True)
        text, elapsed = reader.read(image, args.two_pages)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
        print(text)
        print(f'Recognition: {elapsed:.1f} seconds. Saved to {args.output}')
    finally:
        reader.close()


if __name__ == '__main__':
    main()
