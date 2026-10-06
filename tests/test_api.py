from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from visual_defect_detection.api import create_app


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
