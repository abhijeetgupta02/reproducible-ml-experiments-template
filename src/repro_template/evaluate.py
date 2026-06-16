"""Evaluation entry point.

Run as a module::

    python -m repro_template.evaluate --config configs/experiment.yml
    python -m repro_template.evaluate --run run_20260616_120000_000000

Loads a trained model from a run directory and scores it on the held-out test
set. To guarantee the *same* split the model was trained against, the data
config is read from the run's own ``config.yaml`` rather than re-derived.
Results are written to ``eval_metrics.json`` inside that run directory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, confusion_matrix

from repro_template.data import DatasetConfig, load_dataset
from repro_template.utils import save_json, set_seed

DEFAULT_CONFIG_PATH = "configs/experiment.yml"


@dataclass
class EvalConfig:
    seed: int = 0
    logging_dir: str = "logs/runs"
    run_subdir: Optional[str] = None  # which run to evaluate; None = latest


def load_eval_config(path: str) -> EvalConfig:
    cfg_dict = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    return EvalConfig(
        seed=cfg_dict.get("seed", 0),
        logging_dir=cfg_dict.get("logging_dir", "logs/runs"),
        run_subdir=cfg_dict.get("run_subdir"),
    )


def _resolve_run_dir(cfg: EvalConfig) -> Path:
    root = Path(cfg.logging_dir)
    if cfg.run_subdir is not None:
        return root / cfg.run_subdir
    run_dirs = sorted(p for p in root.glob("run_*") if p.is_dir())
    if not run_dirs:
        raise RuntimeError(f"No run directories found under {root!s} to evaluate.")
    return run_dirs[-1]  # names are timestamped, so last == latest


def _data_config_for_run(run_dir: Path, fallback_seed: int) -> tuple[DatasetConfig, int]:
    """Reconstruct the dataset config from the run's saved snapshot.

    This is what makes evaluation honest: we rebuild the exact split that was
    used at train time. If the snapshot is missing we fall back to defaults.
    """
    snapshot = run_dir / "config.yaml"
    if snapshot.exists():
        saved = OmegaConf.to_container(OmegaConf.load(snapshot), resolve=True)
        return DatasetConfig(**saved.get("data", {})), int(saved.get("seed", fallback_seed))
    return DatasetConfig(), fallback_seed


def run_evaluation(cfg: EvalConfig) -> Path:
    """Evaluate the selected run on its held-out test set."""
    run_dir = _resolve_run_dir(cfg)

    data_cfg, seed = _data_config_for_run(run_dir, cfg.seed)
    set_seed(seed)

    model_path = run_dir / "model.joblib"
    if not model_path.exists():
        raise RuntimeError(f"No trained model found at {model_path!s}.")
    model = joblib.load(model_path)

    splits = load_dataset(data_cfg)
    y_pred = model.predict(splits.X_test)

    metrics: Dict[str, Any] = {
        "test_accuracy": float(accuracy_score(splits.y_test, y_pred)),
        "confusion_matrix": confusion_matrix(splits.y_test, y_pred).tolist(),
        "n_test": int(len(splits.X_test)),
    }
    save_json(run_dir / "eval_metrics.json", metrics)
    return run_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained run.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to the experiment config YAML (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--run",
        default=None,
        help="Run subdirectory to evaluate (default: most recent run).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_eval_config(args.config)
    if args.run is not None:
        cfg.run_subdir = args.run
    run_dir = run_evaluation(cfg)
    print(f"Evaluation complete. Run directory: {run_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
