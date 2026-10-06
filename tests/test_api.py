from __future__ import annotations

from pathlib import Path

import torch
from fastapi.testclient import TestClient

from visual_defect_detection.api import create_app
from visual_defect_detection.modeling import build_model


def test_health_without_model(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        """
project:
  output_dir: artifacts
api:
  max_file_size_mb: 1
""".strip()
    )

    app = create_app(config_path=str(cfg), model_path=str(tmp_path / "missing.pt"))
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is False


def test_predict_rejects_non_image(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        """
project:
  output_dir: artifacts
api:
  max_file_size_mb: 1
""".strip()
    )

    app = create_app(config_path=str(cfg), model_path=str(tmp_path / "missing.pt"))
    client = TestClient(app)

    response = client.post(
        "/predict",
        files={"file": ("x.txt", b"abc", "text/plain")},
    )
    assert response.status_code == 503


def test_predict_rejects_corrupt_image_with_loaded_model(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "class_names.json").write_text('{"class_names": ["ok", "ng"]}')

    model = build_model(num_classes=2, transfer_learning=False)
    model_path = tmp_path / "dummy.pt"
    torch.save({"model_state_dict": model.state_dict(), "class_names": ["ok", "ng"]}, model_path)

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"""
project:
  output_dir: {output_dir}
api:
  max_file_size_mb: 1
training:
  image_size: 64
""".strip()
    )

    app = create_app(config_path=str(cfg), model_path=str(model_path))
    client = TestClient(app)
    response = client.post(
        "/predict",
        files={"file": ("bad.png", b"not-an-image", "image/png")},
    )
    assert response.status_code == 400
    assert "corrupted image" in response.json()["detail"]
