# Reproducible ML Experiments Template

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-stable-brightgreen)](https://github.com/abhijeetgupta02/reproducible-ml-experiments-template)
[![Maintainer](https://img.shields.io/badge/maintainer-Abhijeet%20Gupta-0e75b6)](https://github.com/abhijeetgupta02)

A small, **working** template for reproducible ML experiments in Python: config
management, per-run logging, a training loop, an evaluation loop, and a tiny
example experiment that actually runs.

> Status: ✅ Working template. The included example trains a small classifier on
> a synthetic dataset and logs everything needed to reproduce the run. Fork it,
> swap in your own data and model, and keep the structure.

## Why you might care

- You are tired of **mystery results** with missing configs and one-off scripts.
- You want a **drop-in experiment skeleton** for a paper, thesis, or benchmark.
- You care that every run leaves behind a **self-contained, reproducible trace**.

## Why this exists

Most "results" are hard to trust because the config, the code, and the numbers
drift apart. This template keeps them together by making three things cheap:

- **Config-first design** — every value that affects a run lives in a
  version-controlled YAML, never hard-coded.
- **Self-contained run directories** — each run writes its config, environment
  fingerprint, metrics, and trained model into one timestamped folder.
- **Deterministic by default** — one seed drives Python, NumPy (and PyTorch if
  you add it), and the data split, so the same config reproduces the same run.

## Repository layout

```text
reproducible-ml-experiments-template/
  configs/
    experiment.yml          # seed, data, model, logging — the single source of truth
  src/
    repro_template/
      __init__.py
      data.py               # synthetic data + deterministic train/val/test split
      models.py             # small MLP + epoch-by-epoch training loop
      train.py              # training entry point (writes a run directory)
      evaluate.py           # evaluation entry point (scores a run on the test set)
      utils.py              # seeding, run dirs, JSON I/O, environment capture
  scripts/
    train.sh
    evaluate.sh
  logs/
    runs/                   # one timestamped subdirectory per run (git-ignored)
  tests/
    test_train_pipeline.py  # fast end-to-end smoke test
  pyproject.toml
  README.md
```

## Quickstart

```bash
# 1. Install (editable, with test extras). A virtualenv is recommended.
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"

# 2. Train using the default config.
python -m repro_template.train --config configs/experiment.yml
# (equivalently: scripts/train.sh)

# 3. Evaluate the most recent run on the held-out test set.
python -m repro_template.evaluate --config configs/experiment.yml
# (or evaluate a specific run: scripts/evaluate.sh run_YYYYMMDD_HHMMSS_ffffff)

# 4. Inspect the artifacts.
ls logs/runs/                       # one folder per run
cat logs/runs/run_*/metrics.json    # training metrics + per-epoch history
```

Two console scripts are also installed: `repro-train` and `repro-evaluate`.

## What the example experiment does

The example is intentionally minimal — it demonstrates the *template*, not a
state-of-the-art result:

- **Data:** scikit-learn's `make_classification` generates a synthetic binary
  classification dataset (no downloads), split deterministically into
  train / validation / test.
- **Model:** a small `MLPClassifier`, trained one epoch at a time so the
  training loop can record genuine per-epoch loss and accuracy.
- **Evaluation:** the saved model is scored on the held-out test set, rebuilding
  the *exact* split from the run's own saved config.

Adapt it by editing `configs/experiment.yml` and replacing `load_dataset` /
`build_mlp_classifier` with your own data and model. The training, logging, and
evaluation plumbing stays the same.

## What a run directory contains

Each run writes a timestamped directory under `logs/runs/`:

| File | Contents |
|---|---|
| `config.yaml` | Exact, fully-resolved config used for the run |
| `environment.json` | Seed, Python version, platform, key package versions |
| `metrics.json` | Final train/val metrics plus full per-epoch `history` |
| `eval_metrics.json` | Test accuracy and confusion matrix (written by `evaluate`) |
| `model.joblib` | The trained model |

### Schematic example (not real numbers)

The structure of `metrics.json` looks like the following. **These values are
illustrative placeholders to show the schema — they are not measured results.**
Run the code to get real numbers for your config.

```json
{
  "epochs": 50,
  "final_train_accuracy": 0.00,
  "final_val_accuracy": 0.00,
  "final_train_loss": 0.00,
  "n_train": 0,
  "n_val": 0,
  "n_test": 0,
  "history": [
    {"epoch": 1, "train_loss": 0.00, "train_accuracy": 0.00, "val_accuracy": 0.00}
  ]
}
```

## Reproducibility notes

- A single `seed` in the config drives all randomness (`utils.set_seed`).
- The train/val/test split is stratified and seeded, so it is stable across runs.
- `evaluate.py` reconstructs the data split from the run's saved `config.yaml`,
  guaranteeing the test set matches what training held out.
- The environment fingerprint records the versions that actually influence
  results, so a run can be explained later.

## Running the tests

```bash
pip install -e ".[test]"
pytest
```

The suite runs a tiny train→evaluate pipeline (few samples, few epochs) and
asserts that the expected artifacts are written. It is fast and uses a
temporary directory, so it never pollutes `logs/`.

## Using this as a template

This is a starting point, not a framework — fork it and make it yours:

- Replace the synthetic dataset in `data.py` with your loader (keep the
  `DataSplits` contract and the rest of the pipeline is unchanged).
- Swap the model in `models.py`; if you adopt PyTorch, `set_seed` already seeds
  it when available.
- Extend `configs/experiment.yml` with whatever your experiment needs — it is
  the single place runs are parameterised.

Dependencies are kept minimal on purpose: `numpy`, `scikit-learn`, `omegaconf`,
and `joblib` (plus `pytest` for tests). Python 3.10+.
