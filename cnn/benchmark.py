
from pathlib import Path
from time import perf_counter

import cv2
import torch
import random
import shutil

from cnn.trainTextCNN import createModel, loadDataset, prepareImage
from ocr.text_gate import TextGate


CNN_THRESHOLD = 0.2
MODEL_PATH = Path("cnn/textCNN.pth")


def evaluateResults(results, name):
    tp = tn = fp = fn = 0

    for actual, predicted in results:
        if actual == 1 and predicted == 1:
            tp += 1
        elif actual == 0 and predicted == 0:
            tn += 1
        elif actual == 0 and predicted == 1:
            fp += 1
        else:
            fn += 1

    total_text = tp + fn
    total_no_text = tn + fp
    total = len(results)

    accuracy = (tp + tn) / total * 100
    recall = tp / total_text * 100
    rejection = tn / total_no_text * 100
    skipped = (tn + fn) / total * 100

    print(f"\n--- {name} ---")
    print("Accuracy:", round(accuracy, 2), "%")
    print("Text recall:", round(recall, 2), "%")
    print("No-text rejection:", round(rejection, 2), "%")
    print("Total images skipped:", round(skipped, 2), "%")

    print("True positives:", tp)
    print("True negatives:", tn)
    print("False positives:", fp)
    print("False negatives:", fn)


def benchmarkCNN(model, test_images):
    results = []
    total_time = 0.0

    model.eval()

    with torch.no_grad():
        for index, (image_path, label) in enumerate(test_images):
            start = perf_counter()

            image = prepareImage(image_path)
            image = image.unsqueeze(0)

            logits = model(image)
            probability = torch.sigmoid(logits).item()

            elapsed = perf_counter() - start
            total_time += elapsed

            predicted = int(probability >= CNN_THRESHOLD)
            results.append((label, predicted))

            if (index + 1) % 100 == 0:
                print(f"CNN: {index + 1}/{len(test_images)}")

    return results, total_time


def benchmarkDetector(detector, test_images):
    results = []
    total_time = 0.0

    for index, (image_path, label) in enumerate(test_images):
        start = perf_counter()

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        predicted = int(detector.detect(image))

        elapsed = perf_counter() - start
        total_time += elapsed

        results.append((label, predicted))

        if (index + 1) % 100 == 0:
            print(f"Detector: {index + 1}/{len(test_images)}")

    return results, total_time



def saveExamples(test_images, cnn_results, detector_results):
    output_folder = Path("cnn/benchmark_images")

    categories = {
        "cnn_correct_detector_wrong": [],
        "detector_correct_cnn_wrong": [],
        "both_wrong": []
    }

    for index, (image_path, label) in enumerate(test_images):
        cnn_prediction = cnn_results[index][1]
        detector_prediction = detector_results[index][1]

        # Only investigate images COCO labels as containing text.
        if label != 1:
            continue

        if cnn_prediction == 1 and detector_prediction == 0:
            categories["cnn_correct_detector_wrong"].append(image_path)

        elif cnn_prediction == 0 and detector_prediction == 1:
            categories["detector_correct_cnn_wrong"].append(image_path)

        elif cnn_prediction == 0 and detector_prediction == 0:
            categories["both_wrong"].append(image_path)

    random.seed(42)

    for category, images in categories.items():
        folder = output_folder / category
        folder.mkdir(parents=True, exist_ok=True)

        # Keep only 30 randomly selected examples.
        selected_images = random.sample(
            images, min(30, len(images))
        )

        for image_path in selected_images:
            shutil.copy2(image_path, folder / image_path.name)

        print(f"{category}: {len(images)} total, "
              f"{len(selected_images)} saved")



def main():
    print("Loading dataset...")

    train_images, validation_images, test_images = loadDataset()

    print("Training images:", len(train_images))
    print("Validation images:", len(validation_images))
    print("Test images:", len(test_images))

    print("\nLoading CNN...")

    model = createModel()
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    )
    model.eval()

    print("Loading PaddleOCR text detector...")

    detector = TextGate()
    detector.load()

    # Warm up both models before timing.
    first_path, _ = test_images[0]

    with torch.no_grad():
        image = prepareImage(first_path).unsqueeze(0)
        model(image)

    image = cv2.imread(str(first_path))
    detector.detect(image)

    print("\nBenchmarking CNN...")

    cnn_results, cnn_time = benchmarkCNN(model, test_images)

    print("\nBenchmarking PaddleOCR detector...")

    try:
        detector_results, detector_time = benchmarkDetector(
            detector, test_images
        )
    finally:
        detector.close()

    evaluateResults(cnn_results, "CNN")
    evaluateResults(detector_results, "PaddleOCR detector")

    saveExamples(test_images, cnn_results, detector_results)

    print("\n--- Processing time ---")

    print("CNN total time:", round(cnn_time, 2), "seconds")
    print("CNN average time:",
          round(cnn_time / len(test_images) * 1000, 2), "ms/image")

    print("Detector total time:", round(detector_time, 2), "seconds")
    print("Detector average time:",
          round(detector_time / len(test_images) * 1000, 2), "ms/image")


if __name__ == "__main__":
    main()
