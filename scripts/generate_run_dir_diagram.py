#!/usr/bin/env python3
"""Generate a small diagram of a typical run directory.

Renders the run-directory layout to ``docs/run_directory.png`` using matplotlib
(no system Graphviz needed). It mirrors what ``repro_template.train`` and
``repro_template.evaluate`` actually write into ``logs/runs/<run>/`` -- it is a
structural illustration only and contains no metric values.

Usage:
    python scripts/generate_run_dir_diagram.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write a file, never open a window
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "docs" / "run_directory.png"

ROOT_LABEL = "logs/runs/run_<UTC %Y%m%d_%H%M%S_%f>/"
# (filename, what it holds) -- written by train.py / evaluate.py.
FILES = [
    ("config.yaml", "exact, fully-resolved config for this run"),
    ("environment.json", "seed, Python version, platform, package versions"),
    ("metrics.json", "final train/val metrics + per-epoch history"),
    ("eval_metrics.json", "test accuracy + confusion matrix (from evaluate)"),
    ("model.joblib", "the trained model"),
]


def main() -> None:
    rows = len(FILES)
    fig, ax = plt.subplots(figsize=(9.5, 0.7 * rows + 1.4))
    ax.axis("off")
    ax.set_xlim(0, 10)
    top = 0.7 * rows + 1.0
    ax.set_ylim(0, top + 0.4)

    # Root folder box.
    ax.add_patch(
        FancyBboxPatch(
            (0.3, top - 0.55), 9.4, 0.7,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.8, edgecolor="#b06000", facecolor="#fff4e5",
        )
    )
    ax.text(0.6, top - 0.2, ROOT_LABEL, ha="left", va="center",
            fontsize=12, fontweight="bold", family="monospace", color="#7a4500")

    for i, (fname, desc) in enumerate(FILES):
        y = top - 1.25 - i * 0.7
        ax.add_patch(
            FancyBboxPatch(
                (1.2, y - 0.27), 8.3, 0.54,
                boxstyle="round,pad=0.02,rounding_size=0.05",
                linewidth=1.3, edgecolor="#3367d6", facecolor="#e8f0fe",
            )
        )
        # connector from root to file row
        ax.plot([0.7, 1.2], [y, y], color="#9aa0a6", linewidth=1.0, zorder=0)
        ax.text(1.45, y, fname, ha="left", va="center",
                fontsize=10.5, fontweight="bold", family="monospace",
                color="#1a237e")
        ax.text(4.0, y, desc, ha="left", va="center",
                fontsize=9, color="#5f6368")

    ax.set_title("What a run directory contains", fontsize=13, pad=10)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
