from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np


def calculate_est_qpu_time(
    train_size: int,
    quantum_params: int = 12,
    circuit_depth: int = 10,
    clops: int = 290000,
) -> float:
    """
    Calculate estimated native QPU execution time on IBM Torino (290K CLOPS).

    **CRITICAL: Uses only quantum ansatz parameters, NOT total hybrid network parameters.**

    Reason: The parameter-shift rule for quantum backprop only applies to trainable gates
    in the quantum circuit (ansatz). Classical PyTorch layers (compressors, classifiers)
    use standard CPU backpropagation—no additional circuits needed.

    Using parameter-shift rule gradient estimation:
    - Forward pass: 1 circuit per sample
    - Backward pass: 2 circuits per parameter per sample (parameter shift rule)
    - Total circuits per epoch = train_size × (1 + 2 × quantum_params_only)
    - Total layer operations = circuits × circuit_depth
    - Execution time = operations / 290,000 CLOPS

    Args:
        train_size: Number of training samples
        quantum_params: Trainable parameters in quantum ansatz only (default 12 for RealAmplitudes(4,reps=2))
        circuit_depth: Average circuit depth in layers (default 10)
        clops: IBM Torino speed in Circuit Layer Operations Per Second (default 290,000)

    Returns:
        Estimated execution time in seconds per epoch
    """
    circuits = train_size * (1 + 2 * quantum_params)
    operations = circuits * circuit_depth
    time_sec = operations / clops
    return time_sec


def load_seed_files(results_dir: Path, prefix: str) -> pd.DataFrame:
    pattern = f"{prefix}_results_*.csv"
    seed_files = sorted(results_dir.glob(pattern))

    if not seed_files:
        raise FileNotFoundError(f"No files found for pattern: {results_dir / pattern}")

    frames: list[pd.DataFrame] = []
    for file_path in seed_files:
        df = pd.read_csv(file_path)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "dataset",
        "model",
        "epochs",
        "test_size",
        "learning_rate",
        "hidden_dim",
        "n_chunks",
        "num_qubits",
        "input_dim",
        "num_classes",
        "total_params",
    ]

    agg = (
        df.groupby(group_cols, dropna=False)
        .agg(
            seed_count=("seed", "nunique"),
            train_size=("train_size", "first"),
            test_count=("test_count", "first"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            training_time_mean_sec=("training_time_sec", "mean"),
            training_time_std_sec=("training_time_sec", "std"),
            inference_time_mean_sec=("inference_time_sec", "mean"),
            inference_time_std_sec=("inference_time_sec", "std"),
        )
        .reset_index()
    )

    agg["accuracy_std"] = agg["accuracy_std"].fillna(0.0)
    agg["training_time_std_sec"] = agg["training_time_std_sec"].fillna(0.0)
    agg["inference_time_std_sec"] = agg["inference_time_std_sec"].fillna(0.0)

    agg["accuracy_mean_pct"] = agg["accuracy_mean"] * 100.0
    agg["accuracy_std_pct"] = agg["accuracy_std"] * 100.0
    agg["accuracy_summary"] = agg.apply(
        lambda row: (
            f"{row['accuracy_mean_pct']:.2f}% +/- {row['accuracy_std_pct']:.1f}%"
        ),
        axis=1,
    )

    # Identify quantum models (classical-only models get NaN for QPU timing)
    QUANTUM_MODELS = {
        "ClassicalQuantumMLP",
        "IterativeQNN",
        "SplitAttentionQNN",
        "ResQNet",
        "QKAResQNet",
    }

    # Calculate estimated native QPU execution time on IBM Torino
    agg["est_qpu_time_torino_sec"] = agg.apply(
        lambda row: (
            calculate_est_qpu_time(
                train_size=row["train_size"],
                quantum_params=12,
                circuit_depth=10,
                clops=290000,
            )
            if row["model"] in QUANTUM_MODELS
            else np.nan
        ),
        axis=1,
    )

    ordered_cols = [
        "dataset",
        "model",
        "seed_count",
        "accuracy_summary",
        "accuracy_mean",
        "accuracy_std",
        "accuracy_mean_pct",
        "accuracy_std_pct",
        "training_time_mean_sec",
        "training_time_std_sec",
        "inference_time_mean_sec",
        "inference_time_std_sec",
        "est_qpu_time_torino_sec",
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
    ]

    return agg[ordered_cols].sort_values(["dataset", "model"]).reset_index(drop=True)


def save_aggregated(results_dir: Path, prefix: str) -> Path:
    source_df = load_seed_files(results_dir, prefix)
    aggregated_df = aggregate_results(source_df)

    output_path = results_dir / f"{prefix}_results_seed_summary.csv"
    aggregated_df.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    results_dir = Path("results-fingertips-resn")

    classical_out = save_aggregated(results_dir, "classical")
    quantum_out = save_aggregated(results_dir, "quantum")

    print(f"Saved: {classical_out}")
    print(f"Saved: {quantum_out}")


if __name__ == "__main__":
    main()
