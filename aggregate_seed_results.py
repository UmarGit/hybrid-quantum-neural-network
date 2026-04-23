from __future__ import annotations

from pathlib import Path

import pandas as pd


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
        lambda row: f"{row['accuracy_mean_pct']:.2f}% +/- {row['accuracy_std_pct']:.1f}%",
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
    results_dir = Path("results")

    classical_out = save_aggregated(results_dir, "classical")
    quantum_out = save_aggregated(results_dir, "quantum")

    print(f"Saved: {classical_out}")
    print(f"Saved: {quantum_out}")


if __name__ == "__main__":
    main()
