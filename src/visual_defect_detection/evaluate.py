from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from torchvision import transforms

from .config import AppConfig
from .data import ImageDataset
from .modeling import build_model
from .utils import write_json

logger = logging.getLogger(__name__)


def _build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def run_evaluation(cfg: AppConfig, model_path: str | None = None) -> dict:
    output_dir = Path(cfg.project.output_dir)
    split_file = output_dir / "splits" / "test.csv"
    if not split_file.exists():
        raise FileNotFoundError(f"Test split file not found: {split_file}")

    test_df = pd.read_csv(split_file)
    class_names = sorted(test_df["label"].unique().tolist())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    model_file = Path(model_path or cfg.training.model_path)
    checkpoint = torch.load(model_file, map_location="cpu")
    if "class_names" in checkpoint:
        class_names = checkpoint["class_names"]
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    test_ds = ImageDataset(test_df, class_to_idx, transform=_build_eval_transform(cfg.training.image_size))
    test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.training.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(class_names), transfer_learning=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    y_probs: list[np.ndarray] = []
    filepaths: list[str] = []

    with torch.no_grad():
        for images, labels, paths in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_probs.extend(probs.cpu().numpy())
            filepaths.extend(paths)

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    metrics = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision_macro": float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "num_test_samples": int(len(test_df)),
    }

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()

    eval_dir = output_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    cm_path = eval_dir / "confusion_matrix.png"
    fig.savefig(cm_path)
    plt.close(fig)

    report = classification_report(y_true_arr, y_pred_arr, target_names=class_names, output_dict=True, zero_division=0)

    fp_fn: dict[str, dict[str, list[dict]]] = {
        class_name: {"false_positives": [], "false_negatives": []} for class_name in class_names
    }
    for idx, (true_idx, pred_idx) in enumerate(zip(y_true_arr, y_pred_arr)):
        if true_idx == pred_idx:
            continue
        true_label = class_names[int(true_idx)]
        pred_label = class_names[int(pred_idx)]
        confidence = float(y_probs[idx][pred_idx])
        sample = {
            "filepath": filepaths[idx],
            "true_label": true_label,
            "predicted_label": pred_label,
            "confidence": confidence,
        }
        fp_fn[pred_label]["false_positives"].append(sample)
        fp_fn[true_label]["false_negatives"].append(sample)

    write_json(eval_dir / "metrics.json", metrics)
    write_json(eval_dir / "classification_report.json", report)
    write_json(eval_dir / "fp_fn_analysis.json", fp_fn)

    logger.info("Evaluation completed", extra={"metrics": metrics, "confusion_matrix_path": str(cm_path)})
    return {"metrics": metrics, "confusion_matrix": str(cm_path), "analysis_path": str(eval_dir / "fp_fn_analysis.json")}
