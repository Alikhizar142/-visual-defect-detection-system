from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from .modeling import build_model


@dataclass(slots=True)
class PredictionResult:
    predicted_label: str
    confidence: float
    probabilities: dict[str, float]


class Predictor:
    def __init__(self, model_path: str | Path, class_names: list[str], image_size: int = 224) -> None:
        self.class_names = class_names
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(num_classes=len(class_names), transfer_learning=False)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image: Image.Image) -> PredictionResult:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu()

        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())
        probabilities = {self.class_names[i]: float(probs[i].item()) for i in range(len(self.class_names))}
        return PredictionResult(
            predicted_label=self.class_names[pred_idx],
            confidence=confidence,
            probabilities=probabilities,
        )
