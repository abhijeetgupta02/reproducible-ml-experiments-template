from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
from omegaconf import OmegaConf

from repro_template.data import DatasetConfig, load_synthetic_classification
from repro_template.utils import save_json


@dataclass
class EvalConfig:
    seed: int = 0
    data: DatasetConfig = DatasetConfig()
    logging_dir: str = "logs/runs"
    run_subdir: str | None = None  # which run to evaluate


def load_eval_config(path: str) -> EvalConfig:
    cfg_dict = OmegaConf.to_object(OmegaConf.load(path))
    data_cfg = DatasetConfig(**cfg_dict.get("data", {}))
    return EvalConfig(
        seed=cfg_dict.get("seed", 0),
        data=data_cfg,
        logging_dir=cfg_dict.get("logging_dir", "logs/runs"),
        run_subdir=cfg_dict.get("run_subdir"),
    )


def run_evaluation(cfg: EvalConfig) -> Path:
    np.random.seed(cfg.seed)

    root = Path(cfg.logging_dir)
    if cfg.run_subdir is None:
        # Pick latest run by modified time.
        run_dirs = sorted(
            [p for p in root.glob("run_*") if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
        )
        if not run_dirs:
            raise RuntimeError("No run directories found to evaluate.")
        run_dir = run_dirs[-1]
    else:
        run_dir = root / cfg.run_subdir

    try:
        import joblib

        model = joblib.load(run_dir / "model.joblib")
    except Exception as exc:
        raise RuntimeError("Could not load model for evaluation.") from exc

    _, X_test, _, y_test = load_synthetic_classification(cfg.data)
    test_acc = float(model.score(X_test, y_test))

    metrics: Dict[str, Any] = {
        "test_accuracy": test_acc,
        "n_test": int(len(X_test)),
    }

    save_json(run_dir / "eval_metrics.json", metrics)
    return run_dir


if __name__ == "__main__":  # pragma: no cover
    cfg = load_eval_config("configs/experiment.yml")
    run_dir = run_evaluation(cfg)
    print(f"Evaluation complete. Run directory: {run_dir}")
