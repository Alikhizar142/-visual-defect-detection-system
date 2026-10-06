from __future__ import annotations

import io
import logging
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from .config import load_config
from .inference import Predictor

logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class PredictResponse(BaseModel):
    predicted_label: str
    confidence: float
    probabilities: dict[str, float]
    latency_ms: float


def create_app(config_path: str, model_path: str) -> FastAPI:
    cfg = load_config(config_path)

    app = FastAPI(title="Visual Defect Detection API", version="0.1.0")
    class_names_file = Path(cfg.project.output_dir) / "class_names.json"
    predictor: Predictor | None = None

    if Path(model_path).exists() and class_names_file.exists():
        import json

        class_names = json.loads(class_names_file.read_text())["class_names"]
        predictor = Predictor(model_path=model_path, class_names=class_names, image_size=cfg.training.image_size)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", model_loaded=predictor is not None)

    @app.post("/predict", response_model=PredictResponse)
    async def predict(file: UploadFile = File(...)) -> PredictResponse:
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model is not loaded")
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload an image.")

        content = await file.read()
        max_bytes = cfg.api.max_file_size_mb * 1024 * 1024
        if len(content) > max_bytes:
            raise HTTPException(status_code=400, detail=f"File exceeds maximum size of {cfg.api.max_file_size_mb}MB")

        try:
            image = Image.open(io.BytesIO(content))
            image.verify()
            image = Image.open(io.BytesIO(content)).convert("RGB")
        except (UnidentifiedImageError, OSError):
            raise HTTPException(status_code=400, detail="Invalid or corrupted image file")

        start = time.perf_counter()
        result = predictor.predict(image)
        latency_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "Inference completed",
            extra={
                "filename": file.filename,
                "predicted_label": result.predicted_label,
                "confidence": result.confidence,
                "latency_ms": latency_ms,
            },
        )
        return PredictResponse(
            predicted_label=result.predicted_label,
            confidence=result.confidence,
            probabilities=result.probabilities,
            latency_ms=latency_ms,
        )

    return app
