from __future__ import annotations

import argparse
import logging

import uvicorn

from .api import create_app
from .config import load_config
from .evaluate import run_evaluation
from .logging_utils import configure_logging
from .train import run_training



def main() -> None:
    parser = argparse.ArgumentParser(description="Visual Defect Detection System")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--config", required=True, help="Path to YAML config")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    eval_parser.add_argument("--config", required=True, help="Path to YAML config")
    eval_parser.add_argument("--model-path", required=False, help="Optional model path override")

    serve_parser = subparsers.add_parser("serve", help="Run FastAPI service")
    serve_parser.add_argument("--config", required=True, help="Path to YAML config")
    serve_parser.add_argument("--model-path", required=True, help="Model checkpoint path")

    args = parser.parse_args()
    configure_logging()

    if args.command == "train":
        cfg = load_config(args.config)
        summary = run_training(cfg)
        logging.getLogger(__name__).info("Training complete", extra={"summary": summary})
    elif args.command == "evaluate":
        cfg = load_config(args.config)
        result = run_evaluation(cfg, model_path=args.model_path)
        logging.getLogger(__name__).info("Evaluation complete", extra={"result": result})
    elif args.command == "serve":
        cfg = load_config(args.config)
        app = create_app(config_path=args.config, model_path=args.model_path)
        uvicorn.run(app, host=cfg.api.host, port=cfg.api.port)


if __name__ == "__main__":
    main()
