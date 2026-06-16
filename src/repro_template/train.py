"""Training entry point.

Run as a module::

    python -m repro_template.train --config configs/experiment.yml

Each run produces a self-contained directory under ``logging_dir`` holding the
exact config used, the environment fingerprint, per-epoch metrics, and the
trained model -- everything needed to understand or reproduce the run.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import joblib
from omegaconf import OmegaConf

from repro_template.data import DatasetConfig, load_dataset
from repro_template.models import ModelConfig, build_mlp_classifier, train_with_history
from repro_template.utils import (
    capture_environment,
    prepare_run_dir,
    save_json,
    set_seed,
)

DEFAULT_CONFIG_PATH = "configs/experiment.yml"


@dataclass
class ExperimentConfig:
    seed: int = 0
    data: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    logging_dir: str = "logs/runs"


def load_experiment_config(path: str) -> ExperimentConfig:
    """Load and structure the YAML config via OmegaConf."""
    cfg_dict = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    return ExperimentConfig(
        seed=cfg_dict.get("seed", 0),
        data=DatasetConfig(**cfg_dict.get("data", {})),
        model=ModelConfig(**cfg_dict.get("model", {})),
        logging_dir=cfg_dict.get("logging_dir", "logs/runs"),
    )


def run_training(cfg: ExperimentConfig) -> Path:
    """Execute one training run and return its run directory."""
    set_seed(cfg.seed)

    splits = load_dataset(cfg.data)
    model = build_mlp_classifier(cfg.model)

    run_dir = prepare_run_dir(Path(cfg.logging_dir))

    # Snapshot the config and environment *before* training, so a crashed run
    # still leaves behind a record of how it was launched.
    OmegaConf.save(OmegaConf.structured(cfg), run_dir / "config.yaml")
    save_json(run_dir / "environment.json", capture_environment(cfg.seed))

    history = train_with_history(
        model,
        splits.X_train,
        splits.y_train,
        splits.X_val,
        splits.y_val,
        epochs=cfg.model.epochs,
    )

    metrics: Dict[str, Any] = {
        "final_train_accuracy": history[-1]["train_accuracy"],
        "final_val_accuracy": history[-1]["val_accuracy"],
        "final_train_loss": history[-1]["train_loss"],
        "epochs": cfg.model.epochs,
        "n_train": int(len(splits.X_train)),
        "n_val": int(len(splits.X_val)),
        "n_test": int(len(splits.X_test)),
        "history": history,
    }
    save_json(run_dir / "metrics.json", metrics)

    joblib.dump(model, run_dir / "model.joblib")

    return run_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the example experiment.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to the experiment config YAML (default: {DEFAULT_CONFIG_PATH}).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_experiment_config(args.config)
    run_dir = run_training(cfg)
    print(f"Training complete. Run directory: {run_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
