"""Shared utilities for reproducible experiments.

Everything here is deliberately small and dependency-light: seeding,
run-directory management, JSON persistence, and environment capture. These
are the pieces that make a run *re-runnable* and *auditable*, which is the
whole point of the template.
"""

from __future__ import annotations

import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Packages whose versions we record per run for reproducibility auditing.
_TRACKED_PACKAGES = ("numpy", "scikit-learn", "omegaconf", "joblib")


def set_seed(seed: int) -> None:
    """Seed every RNG we might touch.

    Seeds Python's ``random``, NumPy, and ``PYTHONHASHSEED``. If PyTorch is
    installed it is seeded too (CPU and CUDA), but it is never imported as a
    hard dependency -- the template runs on scikit-learn alone.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:  # optional: only if the user adds torch to their project
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:
        pass


def prepare_run_dir(root: Path) -> Path:
    """Create and return a fresh, uniquely named run directory under ``root``.

    The name is a UTC timestamp down to microseconds, which keeps runs sorted
    chronologically and avoids collisions when two runs start in the same
    second (e.g. inside the test suite).
    """
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    run_dir = root / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write ``payload`` to ``path`` as pretty-printed, sorted JSON."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:  # pragma: no cover - defensive
        return "not-installed"


def capture_environment(seed: int) -> Dict[str, Any]:
    """Capture enough of the environment to explain how a run was produced.

    This is intentionally lightweight (no pip-freeze of the world); it records
    the interpreter, platform, the seed in effect, and the versions of the
    libraries that actually influence results.
    """
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {name: _package_version(name) for name in _TRACKED_PACKAGES},
    }
