from pathlib import Path

import cv2
import numpy as np
import torch

from torch import nn
from torchvision.models import mobilenet_v3_small

from processing.processingLibrary import resizeImage



class CNNTextGate:
    def __init__(self, threshold=0.2):
        self.threshold = threshold
        self.model = None

    def load(self):
        if self.model is not None:
            return

        model = mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)

        path = Path(__file__).with_name("textCNN.pth")

        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))

        model.eval()
        self.model = model

    def detect(self, frame):
        self.load()

        image = resizeImage(frame, 224, 224)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = image.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        image = (image - mean) / std

        image = torch.from_numpy(image.transpose(2, 0, 1).copy()).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(image)
            probability = torch.sigmoid(logits).item()

        return probability >= self.threshold

    def close(self):
        self.model = None