import torch
from torch.utils.data import DataLoader

from cnn.textClassifier import CNNTextGate
from cnn.trainTextCNN import TextDataset, loadDataset


THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def main():
    _, _, test = loadDataset()
    if not test:
        raise ValueError("No held-out test images found in cnn/datasets/text and noText")
    labels = [int(label) for _, label in test]
    if 0 not in labels or 1 not in labels:
        raise ValueError("The held-out test set must contain both text and no-text images")

    gate = CNNTextGate()
    gate.load()
    scores = []
    loader = DataLoader(TextDataset(test), batch_size=16, shuffle=False, num_workers=0)
    try:
        with torch.inference_mode():
            for images, _ in loader:
                scores.extend(torch.sigmoid(gate.model(images)).flatten().tolist())
    finally:
        gate.close()

    print(f"Held-out test set: {len(test)} images ({sum(labels)} text, {labels.count(0)} no-text)")
    print("Threshold | Accuracy | Text recall | No-text rejection | False negatives")
    for threshold in THRESHOLDS:
        predictions = [score >= threshold for score in scores]
        tp = sum(actual == 1 and predicted for actual, predicted in zip(labels, predictions))
        tn = sum(actual == 0 and not predicted for actual, predicted in zip(labels, predictions))
        fp = sum(actual == 0 and predicted for actual, predicted in zip(labels, predictions))
        fn = sum(actual == 1 and not predicted for actual, predicted in zip(labels, predictions))
        accuracy = (tp + tn) / len(labels)
        recall = tp / (tp + fn)
        rejection = tn / (tn + fp)
        print(f"{threshold:.2f} | {accuracy:.2%} | {recall:.2%} | {rejection:.2%} | {fn}")


if __name__ == "__main__":
    main()
