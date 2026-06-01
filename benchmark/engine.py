"""Multi-seed execution engine for matched quantum/classical pairs."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from qiskit import transpile
from qiskit.primitives import BackendEstimatorV2
from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient
from qiskit_machine_learning.neural_networks import EstimatorQNN

# Reuse the user's circuit builder and seeding helper; do not redefine them.
from fragments.quantum import get_quantum_circuit, seed_everything

from .ablation import AblationValidator, count_trainable_parameters
from .config import BenchmarkConfig, ModelPair
from .data import DataSplit, DataStarvationModule
from .noise import IBMNoiseModel

# Estimator shot budgets, mirroring quantum.get_quantum_circuit.
_DEFAULT_SHOTS = 256
_GRADIENT_SHOTS = 64

# Per-seed CSV schema, matching fragments/collect_tables.py so the output is
# consumable by aggregate_seed_results.py and run_stat_tests.py unchanged.
CSV_HEADERS = [
    "timestamp",
    "dataset",
    "model",
    "accuracy",
    "training_time_sec",
    "inference_time_sec",
    "epochs",
    "test_size",
    "learning_rate",
    "hidden_dim",
    "n_chunks",
    "num_qubits",
    "input_dim",
    "num_classes",
    "train_size",
    "test_count",
    "total_params",
    "seed",
]


def build_qnn(
    num_qubits: int,
    entanglement: str,
    noise_mode: str,
    seed: int,
) -> EstimatorQNN:
    """Build an ``EstimatorQNN`` reusing the user's circuit, swapping only noise.

    The feature map, ansatz and observables come from
    ``quantum.get_quantum_circuit`` unchanged; this function rebinds them to an
    Aer backend selected by :class:`IBMNoiseModel` so the same circuit can run
    noiseless or under an audited hardware profile.
    """
    qc, _discard_qnn, feature_map, ansatz, observables = get_quantum_circuit(
        num_qubits=num_qubits, entanglement=entanglement
    )

    backend = IBMNoiseModel(noise_mode).build_backend(seed=seed)
    qc_transpiled = transpile(
        qc, backend=backend, optimization_level=1, seed_transpiler=seed
    )

    estimator = BackendEstimatorV2(backend=backend)
    estimator.options.default_shots = _DEFAULT_SHOTS
    gradient = ParamShiftEstimatorGradient(
        estimator, options={"default_shots": _GRADIENT_SHOTS}
    )

    return EstimatorQNN(
        circuit=qc_transpiled,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        observables=observables,
        estimator=estimator,
        gradient=gradient,
    )


def _train_torch_model(
    model: nn.Module,
    split: DataSplit,
    epochs: int,
    learning_rate: float,
) -> tuple[float, float, float]:
    """Train a torch model and return ``(accuracy, train_time, inference_time)``."""
    x_train = torch.tensor(split.x_train, dtype=torch.float32)
    y_train = torch.tensor(split.y_train, dtype=torch.long)
    x_test = torch.tensor(split.x_test, dtype=torch.float32)
    y_test = torch.tensor(split.y_test, dtype=torch.long)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(x_train), y_train)
        loss.backward()
        optimizer.step()
    train_time = time.time() - start

    start = time.time()
    model.eval()
    with torch.no_grad():
        preds = model(x_test).argmax(dim=1)
        accuracy = (preds == y_test).float().mean().item()
    inference_time = time.time() - start
    return accuracy, train_time, inference_time


def _train_svm(model, split: DataSplit) -> tuple[float, float, float]:
    """Fit/predict an SVM and return ``(accuracy, fit_time, inference_time)``."""
    start = time.time()
    model.fit(split.x_train, split.y_train)
    fit_time = time.time() - start

    start = time.time()
    preds = model.predict(split.x_test)
    inference_time = time.time() - start

    accuracy = float(np.mean(preds == split.y_test))
    return accuracy, fit_time, inference_time


@dataclass
class ModelSummary:
    """Aggregated result for one model across all seeds."""

    dataset: str
    model: str
    seed_count: int
    accuracy_mean: float
    accuracy_std: float
    training_time_mean_sec: float
    total_params: int
    accuracies: np.ndarray = field(repr=False)


class ExecutionEngine:
    """Run a matched pair across N seeds under identical training settings."""

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.data_module = DataStarvationModule(
            dataset=config.dataset,
            test_split=config.test_split,
            num_qubits=config.num_qubits,
        )

    # -- model construction ------------------------------------------------- #
    def _build_quantum_model(self, pair: ModelPair, split: DataSplit, seed: int):
        if pair.is_svm:
            return pair.quantum_cls(
                num_qubits=self.config.num_qubits, random_state=seed
            )
        if pair.quantum_builder is None:
            raise ValueError(f"Pair {pair.name!r} has no quantum builder.")
        qnn = build_qnn(
            num_qubits=self.config.num_qubits,
            entanglement=self.config.entanglement,
            noise_mode=self.config.noise_mode,
            seed=seed,
        )
        return pair.quantum_builder(
            qnn, split.input_dim, split.num_classes, self.config
        )

    def _build_classical_model(
        self, pair: ModelPair, q_model, split: DataSplit, seed: int
    ):
        if pair.is_svm:
            return pair.classical_cls(random_state=seed)
        if pair.classical_builder is None:
            raise ValueError(f"Pair {pair.name!r} has no classical builder.")
        # Counterpart dims come from the run config (and the quantum model's
        # non-configurable structural attributes where applicable).
        return pair.classical_builder(
            q_model, split.input_dim, split.num_classes, self.config
        )

    # -- training dispatch -------------------------------------------------- #
    @staticmethod
    def _train(model, split: DataSplit, config: BenchmarkConfig, is_svm: bool):
        if is_svm:
            return _train_svm(model, split)
        return _train_torch_model(model, split, config.epochs, config.learning_rate)

    @staticmethod
    def _param_count(model, is_svm: bool) -> int:
        if is_svm:
            return 0
        return count_trainable_parameters(model)

    def _build_row(
        self,
        *,
        model_name: str,
        is_quantum: bool,
        is_svm: bool,
        seed: int,
        split: DataSplit,
        accuracy: float,
        training_time: float,
        inference_time: float,
        total_params: int,
    ) -> dict[str, object]:
        """Assemble one per-seed CSV row in the collect_tables schema.

        Mirrors fragments/collect_tables.py: classical rows record ``hidden_dim``
        and leave ``num_qubits`` blank; quantum rows record ``num_qubits``.
        SVM rows leave the gradient-training hyperparameters blank.
        """
        cfg = self.config
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset": cfg.dataset,
            "model": model_name,
            "accuracy": round(accuracy, 6),
            "training_time_sec": round(training_time, 6),
            "inference_time_sec": round(inference_time, 6),
            "epochs": "" if is_svm else cfg.epochs,
            "test_size": cfg.test_split,
            "learning_rate": "" if is_svm else cfg.learning_rate,
            "hidden_dim": "" if (is_svm or is_quantum) else cfg.num_qubits,
            "n_chunks": "" if is_svm else cfg.n_chunks,
            "num_qubits": cfg.num_qubits if is_quantum else "",
            "input_dim": split.input_dim,
            "num_classes": split.num_classes,
            "train_size": int(split.x_train.shape[0]),
            "test_count": int(split.x_test.shape[0]),
            "total_params": total_params,
            "seed": seed,
        }

    # -- main loop ---------------------------------------------------------- #
    def run(self, pair: ModelPair, validate_ablation: bool = True):
        """Run the N-seed loop, returning ``(quantum_summary, classical_summary)``.

        Per-seed rows (full collect_tables schema) are accumulated in
        ``self.classical_rows`` / ``self.quantum_rows`` for :meth:`write_results`.
        """
        self.classical_rows: list[dict[str, object]] = []
        self.quantum_rows: list[dict[str, object]] = []
        q_metrics: list[tuple[float, float]] = []  # (accuracy, training_time)
        c_metrics: list[tuple[float, float]] = []
        q_params = c_params = 0

        for offset in range(self.config.n_seeds):
            seed = self.config.base_seed + offset
            # Seed torch + numpy + python RNGs identically for both models.
            seed_everything(seed)
            split = self.data_module.split(seed)

            q_model = self._build_quantum_model(pair, split, seed)
            c_model = self._build_classical_model(pair, q_model, split, seed)

            if validate_ablation and not pair.is_svm:
                AblationValidator(q_model, c_model, pair.expected_delta).validate()

            q_params = self._param_count(q_model, pair.is_svm)
            c_params = self._param_count(c_model, pair.is_svm)

            q_acc, q_time, q_infer = self._train(q_model, split, self.config, pair.is_svm)
            c_acc, c_time, c_infer = self._train(c_model, split, self.config, pair.is_svm)

            self.quantum_rows.append(
                self._build_row(
                    model_name=pair.quantum_cls.__name__,
                    is_quantum=True,
                    is_svm=pair.is_svm,
                    seed=seed,
                    split=split,
                    accuracy=q_acc,
                    training_time=q_time,
                    inference_time=q_infer,
                    total_params=q_params,
                )
            )
            self.classical_rows.append(
                self._build_row(
                    model_name=pair.classical_cls.__name__,
                    is_quantum=False,
                    is_svm=pair.is_svm,
                    seed=seed,
                    split=split,
                    accuracy=c_acc,
                    training_time=c_time,
                    inference_time=c_infer,
                    total_params=c_params,
                )
            )
            q_metrics.append((q_acc, q_time))
            c_metrics.append((c_acc, c_time))

        q_summary = self._summarise(pair.quantum_cls.__name__, q_metrics, q_params)
        c_summary = self._summarise(pair.classical_cls.__name__, c_metrics, c_params)
        return q_summary, c_summary

    def _summarise(
        self,
        model_name: str,
        metrics: list[tuple[float, float]],
        total_params: int,
    ) -> ModelSummary:
        accuracies = np.array([m[0] for m in metrics], dtype=float)
        times = np.array([m[1] for m in metrics], dtype=float)
        return ModelSummary(
            dataset=self.config.dataset,
            model=model_name,
            seed_count=len(metrics),
            accuracy_mean=float(np.mean(accuracies)),
            accuracy_std=float(np.std(accuracies, ddof=1)) if len(metrics) > 1 else 0.0,
            training_time_mean_sec=float(np.mean(times)),
            total_params=total_params,
            accuracies=accuracies,
        )

    # -- persistence -------------------------------------------------------- #
    def _resolve_out_dir(self) -> Path:
        """Resolve and create the output directory, with a clear failure path.

        Relative dirs resolve against the cwd and ``~`` is expanded, so
        ``results`` lands in the repo root rather than the filesystem root.
        """
        out_dir = Path(self.config.output_dir).expanduser()
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise OSError(
                f"Cannot create output directory {out_dir!r}: {exc.strerror}. "
                "Choose a writable location, e.g. --output-dir results "
                "(relative to the repo root) or --output-dir ~/benchmark_results."
            ) from exc
        return out_dir

    @staticmethod
    def _append_rows(csv_path: Path, rows: list[dict[str, object]]) -> None:
        """Append rows to a CSV, writing the header only when the file is new."""
        write_header = not csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_HEADERS)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    def write_results(self) -> list[Path]:
        """Write per-seed CSVs in the collect_tables schema; return the paths.

        Files are named ``classical_results_<seed>.csv`` and
        ``quantum_results_<seed>.csv`` and appended to (header written once), so
        the output is directly consumable by ``aggregate_seed_results.py`` and
        ``run_stat_tests.py`` and composes across multiple pair runs into one dir.
        """
        out_dir = self._resolve_out_dir()
        written: list[Path] = []

        for prefix, rows in (
            ("classical", getattr(self, "classical_rows", [])),
            ("quantum", getattr(self, "quantum_rows", [])),
        ):
            for row in rows:
                csv_path = out_dir / f"{prefix}_results_{row['seed']}.csv"
                self._append_rows(csv_path, [row])
                if csv_path not in written:
                    written.append(csv_path)

        return written
