import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from hog.hogFeatures import PATCH_SIZE, WINDOW_STRIDE, prepareFrame, slidingWindows



COCO_TEXT_JSON = Path("cnn/datasets/raw/cocotext.v2.json")
OUTPUT_FOLDER = Path("cnn/datasets/hogPatches")

TEXT_COVERAGE = 0.5
NON_TEXT_COVERAGE = 0.05

MAX_TEXT_PATCHES = 40
NON_TEXT_PATCHES = 20

PREVIEW_COUNT = 20
SEED = 42



def loadTextBoxes(json_path):
    """File name -> list of COCO-Text boxes (x, y, width, height)."""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))

    boxes = {}

    for image_id, image_info in data["imgs"].items():
        annotation_ids = data["imgToAnns"].get(image_id, [])

        boxes[image_info["file_name"]] = [
            data["anns"][str(annotation_id)]["bbox"]
            for annotation_id in annotation_ids
        ]

    return boxes



def textMask(frame_shape, gray_shape, boxes):
    """1 inside text boxes, 0 elsewhere, at the same scale as the HOG windows."""
    mask = np.zeros(frame_shape[:2], np.uint8)

    for x, y, width, height in boxes:
        x1, y1 = int(x), int(y)
        x2, y2 = int(np.ceil(x + width)), int(np.ceil(y + height))

        mask[y1:y2, x1:x2] = 1

    return cv2.resize(mask, (gray_shape[1], gray_shape[0]), interpolation=cv2.INTER_NEAREST)



def patchesFromImage(image_path, boxes, rng):

    frame = cv2.imread(str(image_path))

    if frame is None:
        print(f"Skipping unreadable file {image_path}")
        return None

    gray = prepareFrame(frame)

    mask = textMask(frame.shape, gray.shape, boxes)

    windows = slidingWindows(gray)

    coverage = slidingWindows(mask).mean(axis=(2, 3))

    text_positions = np.argwhere(coverage >= TEXT_COVERAGE)
    non_text_positions = np.argwhere(coverage <= NON_TEXT_COVERAGE)

    text_positions = rng.permutation(text_positions)[:MAX_TEXT_PATCHES].reshape(-1, 2)
    non_text_positions = rng.permutation(non_text_positions)[:NON_TEXT_PATCHES].reshape(-1, 2)

    positions = np.concatenate([text_positions, non_text_positions])

    patches = np.array([windows[row, col] for row, col in positions], np.uint8)

    labels = np.array([1] * len(text_positions) + [0] * len(non_text_positions), np.uint8)

    return {
        "patches": patches.reshape(-1, PATCH_SIZE, PATCH_SIZE),
        "labels": labels,
        "preview": drawLabels(gray, text_positions, non_text_positions)
    }



def drawLabels(gray, text_positions, non_text_positions):
    """Green = text patch, red = non-text patch. Check these before training."""
    preview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for positions, colour in ((text_positions, (0, 200, 0)), (non_text_positions, (0, 0, 220))):
        for row, col in positions:
            y, x = row * WINDOW_STRIDE, col * WINDOW_STRIDE
            cv2.rectangle(preview, (x, y), (x + PATCH_SIZE - 1, y + PATCH_SIZE - 1), colour, 1)

    return preview



def balanceClasses(patches, labels, image_ids, rng):
    """Drop random samples from the bigger class so text and non-text are equal."""
    text_index = np.flatnonzero(labels == 1)
    non_text_index = np.flatnonzero(labels == 0)

    count = min(len(text_index), len(non_text_index))

    keep = np.concatenate([
        rng.choice(text_index, count, replace=False),
        rng.choice(non_text_index, count, replace=False)
    ])

    keep.sort()

    return patches[keep], labels[keep], image_ids[keep]



def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="only use the first N training images (quick test)")
    args = parser.parse_args()

    from cnn.trainTextCNN import loadDataset

    train_images, _, _ = loadDataset()

    if args.limit:
        train_images = train_images[:args.limit]

    boxes = loadTextBoxes(COCO_TEXT_JSON)

    (OUTPUT_FOLDER / "previews").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)

    all_patches, all_labels, all_ids = [], [], []

    for image_id, (image_path, _) in enumerate(train_images):
        sample = patchesFromImage(image_path, boxes.get(image_path.name, []), rng)

        if sample is None:
            continue

        all_patches.append(sample["patches"])
        all_labels.append(sample["labels"])
        all_ids.append(np.full(len(sample["labels"]), image_id, np.int32))

        if image_id < PREVIEW_COUNT:
            cv2.imwrite(str(OUTPUT_FOLDER / "previews" / image_path.name), sample["preview"])

        if (image_id + 1) % 500 == 0:
            print(f"{image_id + 1}/{len(train_images)} images")

    patches = np.concatenate(all_patches)
    labels = np.concatenate(all_labels)
    image_ids = np.concatenate(all_ids)

    print(f"Collected {int(labels.sum())} text / {int((labels == 0).sum())} non-text patches")

    patches, labels, image_ids = balanceClasses(patches, labels, image_ids, rng)

    np.savez_compressed(OUTPUT_FOLDER / "train.npz", patches=patches, labels=labels, image_ids=image_ids)

    print(f"Saved {len(labels)} balanced patches from {len(np.unique(image_ids))} images "
          f"to {OUTPUT_FOLDER / 'train.npz'}")
    print(f"Check the label previews in {OUTPUT_FOLDER / 'previews'}")



if __name__ == "__main__":
    main()