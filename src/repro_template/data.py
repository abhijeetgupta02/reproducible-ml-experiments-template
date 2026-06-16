"""Synthetic dataset generation and deterministic train/val/test splitting.

The example uses scikit-learn's ``make_classification`` so the template has
zero data-download dependencies and is fully deterministic given a seed.
Swap :func:`load_dataset` for your own loader when adapting the template -- as
long as it returns a :class:`DataSplits`, the rest of the pipeline is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


@dataclass
class DatasetConfig:
    n_samples: int = 1000
    n_features: int = 20
    n_informative: int = 10
    n_classes: int = 2
    # Fractions of the full dataset held out for validation and test.
    val_size: float = 0.2
    test_size: float = 0.2
    random_state: int = 0


@dataclass
class DataSplits:
    """A standard three-way split. Validation is used during training for
    model selection / monitoring; test is touched only at evaluation time."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def load_dataset(cfg: DatasetConfig) -> DataSplits:
    """Generate the synthetic dataset and split it into train/val/test.

    The split is stratified and seeded, so the same ``cfg`` always yields the
    same partition -- a precondition for the reproducibility story.
    """
    X, y = make_classification(
        n_samples=cfg.n_samples,
        n_features=cfg.n_features,
        n_informative=cfg.n_informative,
        n_classes=cfg.n_classes,
        random_state=cfg.random_state,
    )

    # First carve off the test set, then split the remainder into train/val.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )
    # Express val_size as a fraction of the remaining trainval pool.
    val_fraction = cfg.val_size / (1.0 - cfg.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_fraction,
        random_state=cfg.random_state,
        stratify=y_trainval,
    )

    return DataSplits(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )
