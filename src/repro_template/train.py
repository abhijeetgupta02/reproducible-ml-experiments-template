from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
from omegaconf import OmegaConf

from repro_template.data import DatasetConfig, load_synthetic_classification
from repro_template.models import ModelConfig, build_mlp_classifier
from repro_template.utils import prepare_run_dir, save_json


@dataclass
class ExperimentConfig:
    seed: int = 0
    data: DatasetConfig = DatasetConfig()
    model: ModelConfig = ModelConfig()
    logging_dir: str = "logs/runs"


def load_experiment_config(path: str) -> ExperimentConfig:
    cfg_dict = OmegaConf.to_object(OmegaConf.load(path))
    data_cfg = DatasetConfig(**cfg_dict.get("data", {}))
    model_cfg = ModelConfig(**cfg_dict.get("model", {}))
    return ExperimentConfig(
        seed=cfg_dict.get("seed", 0),
        data=data_cfg,
        model=model_cfg,
        logging_dir=cfg_dict.get("logging_dir", "logs/runs"),
    )


def run_training(cfg: ExperimentConfig) -> Path:
    np.random.seed(cfg.seed)

    X_train, X_test, y_train, y_test = load_synthetic_classification(cfg.data)
    model = build_mlp_classifier(cfg.model)

    run_dir = prepare_run_dir(Path(cfg.logging_dir))
    model.fit(X_train, y_train)

    train_acc = float(model.score(X_train, y_train))
    test_acc = float(model.score(X_test, y_test))

    metrics: Dict[str, Any] = {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    save_json(run_dir / "metrics.json", metrics)

    # Persist model via joblib for evaluation scripts if needed later.
    try:
        import joblib

        joblib.dump(model, run_dir / "model.joblib")
    except Exception:
        # Model persistence is optional; template should not fail if joblib is missing.
        pass

    return run_dir


if __name__ == "__main__":  # pragma: no cover
    cfg = load_experiment_config("configs/experiment.yml")
    run_dir = run_training(cfg)
    print(f"Training complete. Run directory: {run_dir}")
