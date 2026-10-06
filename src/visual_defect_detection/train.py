from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from .config import AppConfig
from .data import ImageDataset, stratified_split, validate_and_collect_dataset
from .modeling import build_model, unfreeze_for_finetune
from .utils import ensure_parent, set_seed, write_json

logger = logging.getLogger(__name__)


def _build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, eval_tf


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    all_true: list[int] = []
    all_pred: list[int] = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels, _paths in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += float(loss.item())
            preds = torch.argmax(logits, dim=1)
            all_true.extend(labels.cpu().numpy().tolist())
            all_pred.extend(preds.cpu().numpy().tolist())

    avg_loss = total_loss / max(len(loader), 1)
    return avg_loss, np.array(all_true), np.array(all_pred)


def _train_phase(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    running_loss = 0.0
    for images, labels, _paths in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += float(loss.item())
    return running_loss / max(len(loader), 1)


def _create_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    class_to_idx: dict[str, int],
    cfg: AppConfig,
) -> tuple[DataLoader, DataLoader, bool, torch.Tensor | None]:
    train_tf, eval_tf = _build_transforms(cfg.training.image_size)
    train_ds = ImageDataset(train_df, class_to_idx, transform=train_tf)
    val_ds = ImageDataset(val_df, class_to_idx, transform=eval_tf)

    class_counts = train_df["label"].value_counts().reindex(class_to_idx.keys())
    imbalance_ratio = float(class_counts.max() / class_counts.min())
    use_imbalance_strategy = imbalance_ratio > cfg.training.imbalance_ratio_threshold

    sampler = None
    class_weights = None
    if use_imbalance_strategy:
        inv_counts = 1.0 / class_counts.to_numpy(dtype=np.float32)
        class_weights = torch.tensor(inv_counts / inv_counts.sum() * len(class_to_idx), dtype=torch.float32)
        sample_weights = train_df["label"].map(lambda x: float(inv_counts[class_to_idx[x]])).to_numpy(dtype=np.float32)
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        logger.info("Class imbalance detected; enabled weighted sampling and weighted loss", extra={"imbalance_ratio": imbalance_ratio})
    else:
        logger.info("Class distribution acceptable; no imbalance strategy applied", extra={"imbalance_ratio": imbalance_ratio})

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=cfg.training.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )
    return train_loader, val_loader, use_imbalance_strategy, class_weights


def run_training(cfg: AppConfig) -> dict:
    set_seed(cfg.project.seed)

    validation = validate_and_collect_dataset(
        raw_dir=Path(cfg.data.raw_dir),
        image_extensions=cfg.data.image_extensions or [".jpg", ".jpeg", ".png", ".bmp"],
        min_images_per_class=cfg.data.min_images_per_class,
        verify_corruption=cfg.data.verify_corruption,
        check_duplicates=cfg.data.check_duplicates,
    )

    train_df, val_df, test_df = stratified_split(
        validation.dataframe,
        test_size=cfg.data.test_size,
        val_size=cfg.data.val_size,
        random_state=cfg.project.seed,
    )

    output_dir = Path(cfg.project.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(split_dir / "train.csv", index=False)
    val_df.to_csv(split_dir / "val.csv", index=False)
    test_df.to_csv(split_dir / "test.csv", index=False)

    class_names = sorted(train_df["label"].unique().tolist())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    write_json(output_dir / "class_names.json", {"class_names": class_names})

    train_loader, val_loader, imbalance_applied, class_weights = _create_loaders(train_df, val_df, class_to_idx, cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(class_names), transfer_learning=True).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    optimizer = AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)

    best_f1 = -1.0
    history: list[dict] = []
    model_path = Path(cfg.training.model_path)
    ensure_parent(model_path)

    total_epochs = cfg.training.epochs + (cfg.finetune.epochs if cfg.finetune.enabled else 0)
    finetune_start = cfg.training.epochs

    for epoch in range(total_epochs):
        if cfg.finetune.enabled and epoch == finetune_start:
            unfreeze_for_finetune(model)
            optimizer = AdamW(model.parameters(), lr=cfg.finetune.learning_rate, weight_decay=cfg.training.weight_decay)

        train_loss = _train_phase(model, train_loader, device, criterion, optimizer)
        val_loss, y_true, y_pred = _evaluate(model, val_loader, device)
        val_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        epoch_record = {
            "epoch": epoch + 1,
            "phase": "finetune" if cfg.finetune.enabled and epoch >= finetune_start else "transfer_learning",
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_f1": val_f1,
        }
        history.append(epoch_record)

        logger.info("Epoch completed", extra=epoch_record)

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "config": asdict(cfg),
                },
                model_path,
            )

    summary = {
        "best_val_f1": best_f1,
        "model_path": str(model_path),
        "dataset": {
            "total_valid_images": int(len(validation.dataframe)),
            "removed_duplicates": validation.removed_duplicates,
            "corrupted_files": validation.corrupted_files,
            "split_counts": {
                "train": int(len(train_df)),
                "val": int(len(val_df)),
                "test": int(len(test_df)),
            },
            "class_counts": validation.dataframe["label"].value_counts().to_dict(),
        },
        "imbalance_strategy_applied": imbalance_applied,
        "history": history,
    }
    write_json(output_dir / "training_summary.json", summary)
    return summary
