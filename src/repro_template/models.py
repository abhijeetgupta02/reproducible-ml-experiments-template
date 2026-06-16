"""Model definition and an epoch-by-epoch training loop.

The example model is a small scikit-learn MLP. We train it with
``partial_fit`` rather than a single ``fit`` so that we can record genuine
per-epoch metrics (training loss plus train/val accuracy) -- the kind of
history a reproducible experiment should log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
from sklearn.neural_network import MLPClassifier


@dataclass
class ModelConfig:
    hidden_layer_sizes: List[int] = field(default_factory=lambda: [64, 64])
    learning_rate_init: float = 1e-3
    epochs: int = 50
    random_state: int = 0


def build_mlp_classifier(cfg: ModelConfig) -> MLPClassifier:
    """Construct the estimator.

    ``max_iter=1`` because we drive the optimisation ourselves, one epoch per
    ``partial_fit`` call, in :func:`train_with_history`.
    """
    return MLPClassifier(
        hidden_layer_sizes=tuple(cfg.hidden_layer_sizes),
        learning_rate_init=cfg.learning_rate_init,
        max_iter=1,
        random_state=cfg.random_state,
    )


def train_with_history(
    model: MLPClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
) -> List[Dict[str, Any]]:
    """Train ``model`` for ``epochs`` passes, returning per-epoch metrics.

    Each entry holds the epoch index, the training loss reported by the
    optimiser, and train/val accuracy measured after that epoch. All numbers
    are computed from the model -- nothing here is fabricated.
    """
    classes = np.unique(y_train)
    history: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        # The first call must declare the full set of classes.
        if epoch == 1:
            model.partial_fit(X_train, y_train, classes=classes)
        else:
            model.partial_fit(X_train, y_train)

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(model.loss_),
                "train_accuracy": float(model.score(X_train, y_train)),
                "val_accuracy": float(model.score(X_val, y_val)),
            }
        )

    return history
