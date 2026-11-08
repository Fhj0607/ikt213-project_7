
import json
import shutil
import tempfile
from pathlib import Path

from datasets import load_dataset, Image


def prepareDataset():
    dataset_folder = Path("cnn/datasets")
    json_path = Path("cnn/datasets/raw/cocotext.v2.json")

    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    dataset_folder.mkdir(parents=True, exist_ok=True)

    # Stream images instead of downloading the entire dataset
    dataset = load_dataset(
        "AbdoTW/COCO_2014",
        split="train",
        streaming=True
    )

    # Keep original image bytes without decoding every image
    dataset = dataset.cast_column("image", Image(decode=False))

    text_count = 0
    no_text_count = 0
    target = 5000

    # Prepare a new dataset before deleting the existing one
    with tempfile.TemporaryDirectory(dir="cnn") as temp:
        temp_folder = Path(temp)

        text_folder = temp_folder / "text"
        no_text_folder = temp_folder / "noText"

        text_folder.mkdir()
        no_text_folder.mkdir()

        for example in dataset:
            image_id = str(example["id"])

            # Only use images covered by COCO-Text annotations
            image_info = data["imgs"].get(image_id)

            if image_info is None:
                continue

            if image_info["set"] != "train":
                continue

            annotation_ids = data["imgToAnns"].get(image_id, [])

            has_text = len(annotation_ids) > 0

            if has_text:
                if text_count >= target:
                    continue

                destination_folder = text_folder
            else:
                if no_text_count >= target:
                    continue

                destination_folder = no_text_folder

            filename = image_info["file_name"]
            destination = destination_folder / filename

            # Get original JPEG data from Hugging Face
            image_data = example["image"]

            image_bytes = image_data["bytes"]

            if image_bytes is None:
                image_bytes = Path(image_data["path"]).read_bytes()

            destination.write_bytes(image_bytes)

            if has_text:
                text_count += 1
            else:
                no_text_count += 1

            if (text_count + no_text_count) % 100 == 0:
                print(f"Text: {text_count}, No text: {no_text_count}")

            if text_count == target and no_text_count == target:
                break

        if text_count != target or no_text_count != target:
            raise RuntimeError(
                f"Not enough images found. "
                f"Text: {text_count}, No text: {no_text_count}"
            )

        # Replace old dataset after successful collection
        old_text = dataset_folder / "text"
        old_no_text = dataset_folder / "noText"

        if old_text.exists():
            shutil.rmtree(old_text)

        if old_no_text.exists():
            shutil.rmtree(old_no_text)

        shutil.move(str(text_folder), str(old_text))
        shutil.move(str(no_text_folder), str(old_no_text))

    print("\nDataset rebuilt!")
    print("Text images:", text_count)
    print("No-text images:", no_text_count)

if __name__ == "__main__":
    prepareDataset()
