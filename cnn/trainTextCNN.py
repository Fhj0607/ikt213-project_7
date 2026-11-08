from pathlib import Path
import random
import cv2
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torch.utils.data import Dataset, DataLoader

from processing.processingLibrary import resizeImage



class TextDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        image = prepareImage(image_path)

        return image, torch.tensor(label, dtype=torch.float32)



def loadDataset():
    text_folder = Path("cnn/datasets/text")
    no_text_folder = Path("cnn/datasets/noText")

    text_images = []
    no_text_images = []

    for image_path in sorted(text_folder.glob("*.jpg")):
        text_images.append((image_path, 1))

    for image_path in sorted(no_text_folder.glob("*.jpg")):
        no_text_images.append((image_path, 0))

    random.seed(42)
    random.shuffle(text_images)
    random.shuffle(no_text_images)

    text_train, text_validation, text_test = splitImages(text_images)
    no_train, no_validation, no_test = splitImages(no_text_images)

    train = text_train + no_train
    validation = text_validation + no_validation
    test = text_test + no_test

    random.shuffle(train)
    random.shuffle(validation)
    random.shuffle(test)

    return train, validation, test



def splitImages(images):
    train_end = int(len(images) * 0.70)
    validation_end = int(len(images) * 0.85)

    train = images[:train_end]
    validation = images[train_end:validation_end]
    test = images[validation_end:]

    return train, validation, test




def prepareImage(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = resizeImage(image, 224, 224)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = torch.from_numpy(image)

    image = image.permute(2, 0, 1)

    image = image.float() / 255.0

    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    for channel in range(3):
        image[channel] = (image[channel] - mean[channel]) / std[channel]

    return image



def createModel():
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)

    for parameter in model.features.parameters():
        parameter.requires_grad = False

    model.classifier[3] = nn.Linear(1024, 1)

    return model



def trainModel(model, train_loader, validation_loader, epochs=5):
    loss_function = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)

    best_accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        model.features.eval()

        total_loss = 0.0

        for images, labels in train_loader:
            predictions = model(images)
            labels = labels.unsqueeze(1)

            loss = loss_function(predictions, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("Average training loss:", round(average_loss, 4))

        accuracy = evaluateModel(model, validation_loader)

        if accuracy > best_accuracy:
            best_accuracy = accuracy

            torch.save(model.state_dict(), "cnn/textCNN.pth")

            print("Saved new best model!")



def evaluateModel(model, validation_loader, threshold=0.5):
    model.eval()

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    with torch.no_grad():
        for images, labels in validation_loader:
            predictions = model(images)

            probabilities = torch.sigmoid(predictions)

            predicted_labels = (probabilities >= threshold).float()
            labels = labels.unsqueeze(1)

            true_positive += ((predicted_labels == 1) & (labels == 1)).sum().item()
            true_negative += ((predicted_labels == 0) & (labels == 0)).sum().item()
            false_positive += ((predicted_labels == 1) & (labels == 0)).sum().item()
            false_negative += ((predicted_labels == 0) & (labels == 1)).sum().item()

    total = true_positive + true_negative + false_positive + false_negative

    accuracy = (true_positive + true_negative) / total

    recall = true_positive / (true_positive + false_negative)

    print("Threshold:", threshold)

    print("Validation accuracy:", round(accuracy * 100, 2), "%")
    print("Text recall:", round(recall * 100, 2), "%")

    print("True positives:", true_positive)
    print("True negatives:", true_negative)
    print("False positives:", false_positive)
    print("False negatives:", false_negative)

    return accuracy



def testThresholds(model, validation_loader):
    model.load_state_dict(
        torch.load("cnn/textCNN.pth", weights_only=True)
    )

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

    for threshold in thresholds:
        print()
        evaluateModel(model, validation_loader, threshold)



if __name__ == "__main__":
    train, validation, test = loadDataset()

    print("Training images:", len(train))
    print("Validation images:", len(validation))
    print("Test images:", len(test))

    train_loader = DataLoader(
        TextDataset(train),
        batch_size=16,
        shuffle=True,
        num_workers=0
    )

    validation_loader = DataLoader(
        TextDataset(validation),
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    model = createModel()

    trainModel(model, train_loader, validation_loader, epochs=5)

    print("\n--- Threshold comparison ---")
    testThresholds(model, validation_loader)