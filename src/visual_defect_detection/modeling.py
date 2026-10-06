from __future__ import annotations

import torch
from torch import nn
from torchvision import models


def build_model(num_classes: int, transfer_learning: bool = True) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)

    if transfer_learning:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def unfreeze_for_finetune(model: nn.Module) -> None:
    for param in model.features.parameters():
        param.requires_grad = True
