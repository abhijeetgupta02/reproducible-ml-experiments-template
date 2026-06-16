from pathlib import Path

from repro_template.train import ExperimentConfig, run_training


def test_train_pipeline_creates_run_dir(tmp_path: Path, monkeypatch):
    # Override logging dir to temporary path
    cfg = ExperimentConfig(logging_dir=str(tmp_path))
    run_dir = run_training(cfg)

    assert run_dir.exists()
    assert (run_dir / "metrics.json").exists()
