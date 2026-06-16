from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


@dataclass
class DatasetConfig:
    n_samples: int = 1000
    n_features: int = 20
    n_informative: int = 10
    n_classes: int = 2
    test_size: float = 0.2
    random_state: int = 0


def load_synthetic_classification(cfg: DatasetConfig) -> Tuple[np.ndarray, ...]:
    X, y = make_classification(
        n_samples=cfg.n_samples,
        n_features=cfg.n_features,
        n_informative=cfg.n_informative,
        n_classes=cfg.n_classes,
        random_state=cfg.random_state,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test
