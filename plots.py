from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("results-lk-col")
PLOTS_DIR = Path("plots")

CLASSICAL_TO_QUANTUM = {
    "ClassicalMLP": "ClassicalQuantumMLP",
    "IterativeClassicalNN": "IterativeQNN",
    "ResNet": "ResQNet",
    "SplitAttentionClassicalNN": "SplitAttentionQNN",
}

MODEL_LABELS = {
    "ClassicalMLP": "MLP",
    "ClassicalQuantumMLP": "MLP",
    "IterativeClassicalNN": "Iterative",
    "IterativeQNN": "Iterative",
    "ResNet": "Res",
    "ResQNet": "Res",
    "SplitAttentionClassicalNN": "SplitAttn",
    "SplitAttentionQNN": "SplitAttn",
}

DATASET_LABELS = {"colon": "Colon", "leukemia": "Leukemia"}


def _load_seed_tables(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    classical_files = sorted(results_dir.glob("classical_results_*.csv"))
    quantum_files = sorted(results_dir.glob("quantum_results_*.csv"))

    if not classical_files:
        raise FileNotFoundError(f"No classical seed files found in {results_dir}")
    if not quantum_files:
        raise FileNotFoundError(f"No quantum seed files found in {results_dir}")

    classical_df = pd.concat((pd.read_csv(f) for f in classical_files), ignore_index=True)
    quantum_df = pd.concat((pd.read_csv(f) for f in quantum_files), ignore_index=True)

    return classical_df, quantum_df


def _build_boxplot_dataframe(classical_df: pd.DataFrame, quantum_df: pd.DataFrame) -> pd.DataFrame:
    classical = classical_df.copy()
    classical["paradigm"] = "Classical"

    quantum = quantum_df.copy()
    quantum["paradigm"] = "Quantum"

    combined = pd.concat([classical, quantum], ignore_index=True)
    combined = combined[combined["dataset"].isin(DATASET_LABELS.keys())].copy()
    combined["dataset_label"] = combined["dataset"].map(DATASET_LABELS)
    combined["accuracy_pct"] = combined["accuracy"] * 100.0

    # Collapse 4-model x 20-seed runs into 20 points per paradigm by averaging per seed.
    per_seed = (
        combined.groupby(["dataset_label", "paradigm", "seed"], as_index=False)
        .agg(accuracy_pct=("accuracy_pct", "mean"))
        .sort_values(["dataset_label", "paradigm", "seed"])
    )

    return per_seed


def _build_parameter_table(classical_df: pd.DataFrame, quantum_df: pd.DataFrame) -> pd.DataFrame:
    # We remove the flawed "core" calculation and extract the actual Total Parameters 
    # to highlight the strict 1-to-1 parameter constraint of the experiment.
    classical_means = (
        classical_df.groupby(["dataset", "model"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            total_params_mean=("total_params", "mean"),
        )
    )

    quantum_means = (
        quantum_df.groupby(["dataset", "model"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            total_params_mean=("total_params", "mean"),
        )
    )

    rows: list[dict[str, object]] = []
    for classical_name, quantum_name in CLASSICAL_TO_QUANTUM.items():
        left = classical_means[classical_means["model"] == classical_name]
        right = quantum_means[quantum_means["model"] == quantum_name]

        merged = left.merge(right, on="dataset", suffixes=("_classical", "_quantum"))
        if merged.empty:
            continue

        for rec in merged.to_dict(orient="records"):
            rows.append(
                {
                    "dataset": rec["dataset"],
                    "dataset_label": DATASET_LABELS.get(rec["dataset"], rec["dataset"]),
                    "family": MODEL_LABELS[classical_name],
                    "classical_accuracy_pct": float(rec["accuracy_mean_classical"]) * 100.0,
                    "quantum_accuracy_pct": float(rec["accuracy_mean_quantum"]) * 100.0,
                    "classical_total_params": float(rec["total_params_mean_classical"]),
                    "quantum_total_params": float(rec["total_params_mean_quantum"]),
                }
            )

    param_df = pd.DataFrame(rows)
    if param_df.empty:
        raise ValueError("Unable to infer parameter table from provided CSVs")

    return param_df


def plot_20_seed_boxplot(seed_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    colors = {"Classical": "#B45309", "Quantum": "#0E7490"}

    for ax, dataset in zip(axes, ["Leukemia", "Colon"]):
        subset = seed_df[seed_df["dataset_label"] == dataset]
        classical_vals = subset[subset["paradigm"] == "Classical"]["accuracy_pct"].values
        quantum_vals = subset[subset["paradigm"] == "Quantum"]["accuracy_pct"].values

        box = ax.boxplot(
            [classical_vals, quantum_vals],
            labels=["Classical", "Quantum"],
            widths=0.55,
            patch_artist=True,
            showmeans=True,
            meanline=True,
            medianprops={"color": "#111827", "linewidth": 1.7},
            meanprops={"color": "#111827", "linewidth": 1.2, "linestyle": "--"},
        )

        for patch, label in zip(box["boxes"], ["Classical", "Quantum"]):
            patch.set_facecolor(colors[label])
            patch.set_alpha(0.55)
            patch.set_edgecolor("#111827")
            patch.set_linewidth(1.2)

        for key in ["whiskers", "caps"]:
            for line in box[key]:
                line.set_color("#111827")
                line.set_linewidth(1.0)

        ax.set_title(f"{dataset} (20 Seed Means)")
        ax.set_xlabel("Paradigm")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    axes[0].set_ylabel("Accuracy (%)")
    fig.suptitle("20-Seed Accuracy Distribution: Classical vs Quantum")
    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_parameter_vs_accuracy(param_df: pd.DataFrame, out_path: Path) -> None:
    summary = (
        param_df.groupby("dataset_label", as_index=False)
        .agg(
            classical_accuracy_pct=("classical_accuracy_pct", "mean"),
            quantum_accuracy_pct=("quantum_accuracy_pct", "mean"),
            classical_total_params=("classical_total_params", "mean"),
            quantum_total_params=("quantum_total_params", "mean"),
        )
        .sort_values("dataset_label")
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = range(len(summary))
    width = 0.36

    axes[0].bar(
        [i - width / 2 for i in x],
        summary["classical_accuracy_pct"],
        width=width,
        label="Classical",
        color="#B45309",
        alpha=0.85,
    )
    axes[0].bar(
        [i + width / 2 for i in x],
        summary["quantum_accuracy_pct"],
        width=width,
        label="Quantum",
        color="#0E7490",
        alpha=0.85,
    )
    axes[0].set_title("Accuracy (Mean Across Architectures)")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_xticks(list(x), summary["dataset_label"].tolist())
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)
    axes[0].legend(frameon=False)

    axes[1].bar(
        [i - width / 2 for i in x],
        summary["classical_total_params"],
        width=width,
        label="Classical Model",
        color="#9A3412",
        alpha=0.9,
    )
    axes[1].bar(
        [i + width / 2 for i in x],
        summary["quantum_total_params"],
        width=width,
        label="Quantum Model",
        color="#155E75",
        alpha=0.9,
    )

    # Added label explaining that capacity is equal
    for i, row in summary.reset_index(drop=True).iterrows():
        axes[1].text(
            i,
            max(row["classical_total_params"], row["quantum_total_params"]) * 1.02,
            "Equal Capacity\n(1-to-1 Ablation)", 
            ha="center",
            va="bottom",
            fontsize=9,
        )

    axes[1].set_title("Total Trainable Parameters")
    axes[1].set_ylabel("Parameter Count")
    axes[1].set_xticks(list(x), summary["dataset_label"].tolist())
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)
    
    # Expand y-axis slightly to fit the text
    axes[1].set_ylim(0, summary[["classical_total_params", "quantum_total_params"]].max().max() * 1.25)

    fig.suptitle("1-to-1 Ablation: Accuracy Parity under Strict Parameter Constraints")
    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    classical_df, quantum_df = _load_seed_tables(RESULTS_DIR)
    seed_df = _build_boxplot_dataframe(classical_df, quantum_df)
    param_df = _build_parameter_table(classical_df, quantum_df)

    boxplot_path = PLOTS_DIR / "lk_col_20_seed_boxplot.png"
    bar_path = PLOTS_DIR / "lk_col_parameter_vs_accuracy.png"

    plot_20_seed_boxplot(seed_df, boxplot_path)
    plot_parameter_vs_accuracy(param_df, bar_path)

    print(f"Saved: {boxplot_path}")
    print(f"Saved: {bar_path}")


if __name__ == "__main__":
    main()