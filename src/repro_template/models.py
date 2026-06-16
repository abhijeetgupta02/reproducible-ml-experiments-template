from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neural_network import MLPClassifier


@dataclass
class ModelConfig:
    hidden_layer_sizes: tuple[int, ...] = (64, 64)
    learning_rate_init: float = 1e-3
    max_iter: int = 50
    random_state: int = 0


def build_mlp_classifier(cfg: ModelConfig) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=cfg.hidden_layer_sizes,
        learning_rate_init=cfg.learning_rate_init,
        max_iter=cfg.max_iter,
        random_state=cfg.random_state,
    )
