# Reproducible ML Experiments Template

Template for **reproducible ML experiments in Python** with config management, logging, evaluation scripts, and a sane directory structure.

> Status: ✨ Planning / scaffolding — base template and example experiment to be added.

## Goals

- Provide a **cookiecutter-style** starting point for ML research projects.
- Make it easy to:
  - Separate **config** from code.
  - Log experiments in a **consistent, machine-readable** way.
  - Run **reliable evaluations** and re-runs.
- Encourage good practices (seed control, environment capture, etc.).

## Repository Layout (planned)

```text
reproducible-ml-experiments-template/
  configs/
    experiment.yml      # Main experiment configuration
    model.yml
    data.yml
  src/
    project_name/
      __init__.py
      data.py          # Data loading / preprocessing
      models.py        # Model definitions / wrappers
      train.py         # Training loop
      evaluate.py      # Evaluation loop
      utils.py         # Shared utilities (seeding, device setup, etc.)
  scripts/
    train.sh
    evaluate.sh
  logs/
    runs/              # Per-run configs + metrics
  tests/
  pyproject.toml
  README.md
```

## Tech Choices (planned)

- **Config management:** Hydra or OmegaConf
- **Logging:** Structured JSON logs + optional TensorBoard / Weights & Biases hooks
- **Experiment tracking:** Simple run directories with configs + metrics

## Example Experiment

The template will include a minimal example experiment (e.g., small CV or RL task) to demonstrate:

- How to structure configs
- How to run `train` and `evaluate`
- How logs and outputs are organized

## Getting Started (planned)

Once the template is in place, usage will roughly look like:

```bash
# Copy or cookiecutter this repo
cp -r reproducible-ml-experiments-template my-experiment
cd my-experiment

# Edit configs
vim configs/experiment.yml

# Run training
python -m project_name.train config=configs/experiment.yml

# Run evaluation
python -m project_name.evaluate config=configs/experiment.yml
```

If you want to suggest specific tools (e.g., your preferred logger, config library, or directory conventions), please open an issue.
