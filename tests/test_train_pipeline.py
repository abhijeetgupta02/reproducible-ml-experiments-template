"""End-to-end smoke test for the train -> evaluate pipeline.

Deliberately tiny (few samples, few epochs) so it runs in well under a second
while still exercising the real code paths and the on-disk run artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

from repro_template.data import DatasetConfig
from repro_template.evaluate import EvalConfig, run_evaluation
from repro_template.models import ModelConfig
from repro_template.train import ExperimentConfig, run_training


def _tiny_config(logging_dir: Path) -> ExperimentConfig:
    return ExperimentConfig(
        seed=123,
        data=DatasetConfig(n_samples=120, n_features=8, n_informative=5),
        model=ModelConfig(hidden_layer_sizes=[8], epochs=3),
        logging_dir=str(logging_dir),
    )


def test_training_writes_expected_artifacts(tmp_path: Path) -> None:
    run_dir = run_training(_tiny_config(tmp_path))

    assert run_dir.exists()
    for artifact in ("config.yaml", "environment.json", "metrics.json", "model.joblib"):
        assert (run_dir / artifact).exists(), f"missing {artifact}"

    metrics = json.loads((run_dir / "metrics.json").read_text())
    # Per-epoch history should have one entry per configured epoch.
    assert len(metrics["history"]) == 3
    assert 0.0 <= metrics["final_train_accuracy"] <= 1.0


def test_evaluation_writes_metrics(tmp_path: Path) -> None:
    run_training(_tiny_config(tmp_path))

    eval_dir = run_evaluation(EvalConfig(logging_dir=str(tmp_path)))
    eval_metrics_path = eval_dir / "eval_metrics.json"
    assert eval_metrics_path.exists()

    eval_metrics = json.loads(eval_metrics_path.read_text())
    assert 0.0 <= eval_metrics["test_accuracy"] <= 1.0
    assert "confusion_matrix" in eval_metrics
