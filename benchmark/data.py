"""Data-starvation splitting with leakage-free [0, pi] scaling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Reuse the user's dataset loader; never reimplement loading logic.
from models.datasets import load_dataset


@dataclass
class DataSplit:
    """A single seed-reproducible, scaled train/test split."""

    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    target_names: list[str]

    @property
    def input_dim(self) -> int:
        return int(self.x_train.shape[1])

    @property
    def num_classes(self) -> int:
        return int(np.unique(self.y_train).size)


class DataStarvationModule:
    """Load a dataset and produce a stratified, train-only-scaled split.

    The large ``test_split`` (default 0.9) deliberately starves the models of
    training data, the regime where quantum advantage is hypothesised.
    """

    def __init__(
        self,
        dataset: str,
        test_split: float = 0.9,
        num_qubits: int = 4,
    ) -> None:
        self.dataset = dataset
        self.test_split = test_split
        self.num_qubits = num_qubits

    def split(self, seed: int) -> DataSplit:
        """Return a reproducible split for ``seed``.

        Features are min-max scaled to ``[0, pi]`` (matching the thesis
        ``collect_tables`` pipeline so results are comparable), with the scaler
        fit on the training partition only — no test statistics leak into
        training.
        """
        x, y, target_names = load_dataset(self.dataset, num_qubits=self.num_qubits)
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y)

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=self.test_split,
            stratify=y,
            random_state=seed,
            shuffle=True,
        )

        scaler = MinMaxScaler(feature_range=(0.0, np.pi))
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        return DataSplit(
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            target_names=list(target_names),
        )
