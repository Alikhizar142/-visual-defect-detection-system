from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


@dataclass(slots=True)
class DatasetValidationResult:
    dataframe: pd.DataFrame
    removed_duplicates: int
    corrupted_files: list[str]


class ImageDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, class_to_idx: dict[str, int], transform=None) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        image = Image.open(row["filepath"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = self.class_to_idx[row["label"]]
        return image, label, row["filepath"]


def _compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_image(path: Path, extensions: list[str]) -> bool:
    return path.suffix.lower() in set(extensions)


def _infer_label(raw_dir: Path, file_path: Path) -> str:
    relative = file_path.relative_to(raw_dir)
    if len(relative.parts) < 2:
        raise ValueError(f"Image {file_path} is not inside a class subdirectory.")
    return relative.parts[0]


def validate_and_collect_dataset(
    raw_dir: Path,
    image_extensions: list[str],
    min_images_per_class: int,
    verify_corruption: bool,
    check_duplicates: bool,
) -> DatasetValidationResult:
    if not raw_dir.exists() or not raw_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {raw_dir}")

    rows: list[dict[str, str]] = []
    hashes_seen: dict[str, str] = {}
    duplicates = 0
    corrupted: list[str] = []

    for file_path in sorted(raw_dir.rglob("*")):
        if not file_path.is_file() or not _is_image(file_path, image_extensions):
            continue
        try:
            if verify_corruption:
                with Image.open(file_path) as img:
                    img.verify()
        except (UnidentifiedImageError, OSError):
            corrupted.append(str(file_path))
            continue

        if check_duplicates:
            file_hash = _compute_sha256(file_path)
            if file_hash in hashes_seen:
                duplicates += 1
                continue
            hashes_seen[file_hash] = str(file_path)

        rows.append({"filepath": str(file_path), "label": _infer_label(raw_dir, file_path)})

    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        raise ValueError("No valid images found after validation.")

    class_counts = dataframe["label"].value_counts()
    if len(class_counts) < 2:
        raise ValueError("Need at least two classes for classification.")
    if int(class_counts.min()) < min_images_per_class:
        raise ValueError(
            f"Each class must have at least {min_images_per_class} images. Found minimum: {int(class_counts.min())}."
        )

    return DatasetValidationResult(
        dataframe=dataframe,
        removed_duplicates=duplicates,
        corrupted_files=corrupted,
    )


def stratified_split(
    dataframe: pd.DataFrame,
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < test_size < 1 or not 0 < val_size < 1:
        raise ValueError("test_size and val_size must be in (0, 1).")
    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be < 1.")

    train_val, test = train_test_split(
        dataframe,
        test_size=test_size,
        stratify=dataframe["label"],
        random_state=random_state,
    )

    adjusted_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=adjusted_val,
        stratify=train_val["label"],
        random_state=random_state,
    )

    leakage = set(train["filepath"]).intersection(set(val["filepath"]) | set(test["filepath"]))
    if leakage:
        raise RuntimeError(f"Data leakage detected across splits: {len(leakage)} overlapping files")

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
