from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ProjectConfig:
    seed: int = 42
    output_dir: str = "artifacts"


@dataclass(slots=True)
class DataConfig:
    raw_dir: str = "data/raw"
    image_extensions: list[str] | None = None
    test_size: float = 0.2
    val_size: float = 0.2
    min_images_per_class: int = 2
    verify_corruption: bool = True
    check_duplicates: bool = True


@dataclass(slots=True)
class TrainingConfig:
    image_size: int = 224
    batch_size: int = 16
    num_workers: int = 0
    epochs: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    imbalance_ratio_threshold: float = 1.5
    model_path: str = "checkpoints/best_model.pt"


@dataclass(slots=True)
class FinetuneConfig:
    enabled: bool = True
    epochs: int = 1
    learning_rate: float = 1e-4


@dataclass(slots=True)
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    max_file_size_mb: int = 5


@dataclass(slots=True)
class AppConfig:
    project: ProjectConfig
    data: DataConfig
    training: TrainingConfig
    finetune: FinetuneConfig
    api: APIConfig


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    default = AppConfig(
        project=ProjectConfig(),
        data=DataConfig(image_extensions=[".jpg", ".jpeg", ".png", ".bmp", ".webp"]),
        training=TrainingConfig(),
        finetune=FinetuneConfig(),
        api=APIConfig(),
    )
    merged = _deep_merge(asdict(default), yaml.safe_load(config_path.read_text()) or {})

    return AppConfig(
        project=ProjectConfig(**merged["project"]),
        data=DataConfig(**merged["data"]),
        training=TrainingConfig(**merged["training"]),
        finetune=FinetuneConfig(**merged["finetune"]),
        api=APIConfig(**merged["api"]),
    )
