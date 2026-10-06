from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from visual_defect_detection.data import stratified_split, validate_and_collect_dataset


def _create_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=color).save(path)


def test_validation_detects_duplicate_and_corrupt(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _create_image(raw / "ok" / "a.png", (255, 0, 0))
    content = (raw / "ok" / "a.png").read_bytes()
    (raw / "ok" / "dup.png").write_bytes(content)
    _create_image(raw / "ng" / "b.png", (0, 255, 0))
    _create_image(raw / "ng" / "c.png", (0, 128, 0))
    _create_image(raw / "ok" / "d.png", (128, 0, 0))
    (raw / "ng" / "bad.png").write_bytes(b"not-an-image")

    result = validate_and_collect_dataset(
        raw_dir=raw,
        image_extensions=[".png"],
        min_images_per_class=2,
        verify_corruption=True,
        check_duplicates=True,
    )

    assert result.removed_duplicates == 1
    assert len(result.corrupted_files) == 1
    assert len(result.dataframe) == 4


def test_stratified_split_no_leakage() -> None:
    df = pd.DataFrame(
        {
            "filepath": [f"/tmp/{i}.png" for i in range(12)],
            "label": ["a"] * 6 + ["b"] * 6,
        }
    )

    train, val, test = stratified_split(df, test_size=0.25, val_size=0.25, random_state=1)
    assert len(set(train["filepath"]) & set(val["filepath"])) == 0
    assert len(set(train["filepath"]) & set(test["filepath"])) == 0
    assert len(set(val["filepath"]) & set(test["filepath"])) == 0
    assert set(train["label"].unique()) == {"a", "b"}
