from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


plt.style.use("seaborn-v0_8-whitegrid")
sns.set_theme(
    context="paper",
    style="whitegrid",
    font="DejaVu Sans",
    font_scale=1.05,
    rc={
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 240,
    },
)


CLASSICAL_COLOR = "#B45309"
QUANTUM_COLOR = "#0E7490"
EDGE_COLOR = "#111827"
GRID_ALPHA = 0.3
TITLE_SIZE = 13
LABEL_SIZE = 11
TICK_SIZE = 10
LEGEND_SIZE = 10
FIGURE_SIZE = (10, 6)
WIDE_FIGURE_SIZE = (11, 6)


MODEL_PAIRS: list[tuple[str, str, str]] = [
    ("ClassicalMLP", "ClassicalQuantumMLP", "MLP"),
    ("IterativeClassicalNN", "IterativeQNN", "Iterative"),
    ("SplitAttentionClassicalNN", "SplitAttentionQNN", "SplitAttn"),
    ("ResNet", "ResQNet", "Res"),
    ("CKAResCNet", "QKAResQNet", "CKARes"),
]

MODEL_LABELS: dict[str, str] = {
    "ClassicalMLP": "MLP",
    "ClassicalQuantumMLP": "MLP",
    "IterativeClassicalNN": "Iterative",
    "IterativeQNN": "Iterative",
    "SplitAttentionClassicalNN": "SplitAttn",
    "SplitAttentionQNN": "SplitAttn",
    "ResNet": "Res",
    "ResQNet": "Res",
    "CKAResCNet": "CKARes",
    "QKAResQNet": "CKARes",
}

DATASET_LABELS: dict[str, str] = {
    "swiss_roll": "Swiss Roll",
    "leukemia": "Leukemia",
    "colon": "Colon",
    "breast": "Breast",
    "ntangled": "NTangled",
}

FAMILY_CONFIGS: dict[str, tuple[str, Path, list[str]]] = {
    "swiss-roll": ("swiss_roll", Path("results-swiss-roll"), ["swiss_roll"]),
    "leukemia-colon-new": (
        "leukemia_colon_new",
        Path("results-leukemia-colon-new"),
        ["leukemia", "colon"],
    ),
    "all-datasets": (
        "all_datasets",
        Path("results-all-datasets"),
        ["swiss_roll", "breast", "qsar", "colon", "leukemia", "ntangled"],
    ),
}

DEFAULT_OUTPUT_DIR = Path("plots/thesis")

CORE_FAMILIES = {"swiss-roll", "leukemia-colon-new", "all-datasets"}
APPENDIX_FAMILIES = {"all-datasets"}


@dataclass(frozen=True)
class FamilyData:
    key: str
    results_dir: Path
    datasets: list[str]
    classical_summary: pd.DataFrame
    quantum_summary: pd.DataFrame
    seed_runs: pd.DataFrame


def dataset_title(dataset: str) -> str:
    return DATASET_LABELS.get(dataset, dataset.replace("_", " ").title())


def load_summary_table(results_dir: Path, prefix: str) -> pd.DataFrame:
    summary_path = results_dir / f"{prefix}_results_seed_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary CSV: {summary_path}")
    return pd.read_csv(summary_path)


def load_seed_runs(results_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for paradigm, pattern in (("Classical", "classical_results_*.csv"), ("Quantum", "quantum_results_*.csv")):
        for file_path in sorted(results_dir.glob(pattern)):
            if file_path.name.endswith("seed_summary.csv"):
                continue
            frame = pd.read_csv(file_path)
            frame = frame.copy()
            frame["paradigm"] = paradigm
            frame["source_file"] = file_path.name
            frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No seed CSVs found in {results_dir}")

    combined = pd.concat(frames, ignore_index=True)
    combined["accuracy_pct"] = combined["accuracy"] * 100.0
    return combined


def load_family_data(key: str) -> FamilyData:
    prefix, results_dir, datasets = FAMILY_CONFIGS[key]
    classical_summary = load_summary_table(results_dir, "classical")
    quantum_summary = load_summary_table(results_dir, "quantum")
    seed_runs = load_seed_runs(results_dir)
    return FamilyData(
        key=key,
        results_dir=results_dir,
        datasets=datasets,
        classical_summary=classical_summary,
        quantum_summary=quantum_summary,
        seed_runs=seed_runs,
    )


def ensure_output_dir(base_dir: Path, family_key: str) -> Path:
    out_dir = base_dir / family_key
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_figure(fig: plt.Figure, out_stem: Path) -> None:
    fig.savefig(out_stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def _style_axes(ax: plt.Axes, title: str, xlabel: str | None = None, ylabel: str | None = None) -> None:
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold")
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=LABEL_SIZE, fontweight="bold")
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=LABEL_SIZE, fontweight="bold")
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.grid(axis="y", linestyle="--", alpha=GRID_ALPHA)


def _paired_rows(classical_summary: pd.DataFrame, quantum_summary: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for classical_name, quantum_name, family in MODEL_PAIRS:
        classical_rows = classical_summary[(classical_summary["dataset"] == dataset) & (classical_summary["model"] == classical_name)]
        quantum_rows = quantum_summary[(quantum_summary["dataset"] == dataset) & (quantum_summary["model"] == quantum_name)]
        if classical_rows.empty or quantum_rows.empty:
            continue
        c_row = classical_rows.iloc[0]
        q_row = quantum_rows.iloc[0]
        rows.append(
            {
                "family": family,
                "classical_model": classical_name,
                "quantum_model": quantum_name,
                "classical_accuracy_pct": float(c_row["accuracy_mean_pct"]),
                "quantum_accuracy_pct": float(q_row["accuracy_mean_pct"]),
                "classical_accuracy_std_pct": float(c_row["accuracy_std_pct"]),
                "quantum_accuracy_std_pct": float(q_row["accuracy_std_pct"]),
                "classical_training_time": float(c_row["training_time_mean_sec"]),
                "quantum_training_time": float(q_row["training_time_mean_sec"]),
                "classical_training_std": float(c_row["training_time_std_sec"]),
                "quantum_training_std": float(q_row["training_time_std_sec"]),
                "classical_inference_time": float(c_row["inference_time_mean_sec"]),
                "quantum_inference_time": float(q_row["inference_time_mean_sec"]),
                "classical_inference_std": float(c_row["inference_time_std_sec"]),
                "quantum_inference_std": float(q_row["inference_time_std_sec"]),
                "classical_total_params": float(c_row["total_params"]),
                "quantum_total_params": float(q_row["total_params"]),
            }
        )

    return pd.DataFrame(rows)


def _model_order_for_dataset(seed_runs: pd.DataFrame, dataset: str) -> list[str]:
    subset = seed_runs[seed_runs["dataset"] == dataset].copy()
    if subset.empty:
        return []
    order_df = (
        subset.groupby("model", as_index=False)["accuracy_pct"]
        .median()
        .sort_values("accuracy_pct", ascending=False)
    )
    return order_df["model"].tolist()


def _plot_metric_bars(
    paired: pd.DataFrame,
    dataset: str,
    metric_left: str,
    metric_right: str,
    ylabel: str,
    title: str,
    out_path: Path,
    log_scale: bool = False,
    err_left_col: str | None = None,
    err_right_col: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    x = np.arange(len(paired))
    width = 0.36

    left_values = paired[metric_left].to_list()
    right_values = paired[metric_right].to_list()
    left_err = paired[err_left_col] if err_left_col and err_left_col in paired.columns else None
    right_err = paired[err_right_col] if err_right_col and err_right_col in paired.columns else None

    ax.bar(
        x - width / 2,
        left_values,
        width,
        yerr=left_err,
        capsize=4,
        color=CLASSICAL_COLOR,
        edgecolor=EDGE_COLOR,
        label="Classical",
    )
    ax.bar(
        x + width / 2,
        right_values,
        width,
        yerr=right_err,
        capsize=4,
        color=QUANTUM_COLOR,
        edgecolor=EDGE_COLOR,
        label="Quantum",
    )

    _style_axes(ax, title, ylabel=ylabel)
    ax.set_xticks(x, paired["family"].tolist())
    if log_scale:
        ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=LEGEND_SIZE)

    for idx, value in enumerate(left_values):
        ax.text(idx - width / 2, value * 1.03, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    for idx, value in enumerate(right_values):
        ax.text(idx + width / 2, value * 1.03, f"{value:.1f}", ha="center", va="bottom", fontsize=8)

    save_figure(fig, out_path)


def _plot_accuracy_boxplot(seed_runs: pd.DataFrame, dataset: str, out_path: Path) -> None:
    subset = seed_runs[seed_runs["dataset"] == dataset].copy()
    if subset.empty:
        return

    subset["model_label"] = subset.apply(
        lambda row: f"{row['paradigm']} {MODEL_LABELS.get(row['model'], row['model'])}", axis=1
    )
    order = (
        subset.groupby("model_label", as_index=False)["accuracy_pct"]
        .median()
        .sort_values("accuracy_pct", ascending=False)["model_label"]
        .tolist()
    )
    if not order:
        return

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    palette = {label: (CLASSICAL_COLOR if label.startswith("Classical") else QUANTUM_COLOR) for label in order}

    sns.boxplot(
        data=subset,
        x="model_label",
        y="accuracy_pct",
        order=order,
        palette=palette,
        width=0.62,
        fliersize=2,
        linewidth=1.1,
        ax=ax,
    )

    _style_axes(ax, f"{dataset_title(dataset)} Accuracy Distribution", "Model", "Accuracy (%)")
    ax.tick_params(axis="x", rotation=25)

    save_figure(fig, out_path)


def _seed_boxplot_frame(seed_runs: pd.DataFrame, datasets: list[str]) -> pd.DataFrame:
    subset = seed_runs[seed_runs["dataset"].isin(datasets)].copy()
    if subset.empty:
        return subset

    subset["paradigm"] = subset["paradigm"].astype(str)
    subset["dataset_label"] = subset["dataset"].map(dataset_title)
    subset["accuracy_pct"] = subset["accuracy"] * 100.0
    per_seed = (
        subset.groupby(["dataset_label", "paradigm", "seed"], as_index=False)
        .agg(accuracy_pct=("accuracy_pct", "mean"))
        .sort_values(["dataset_label", "paradigm", "seed"])
    )
    return per_seed


def plot_leukemia_colon_20_seed_boxplot(data: FamilyData, out_path: Path) -> None:
    seed_df = _seed_boxplot_frame(data.seed_runs, ["leukemia", "colon"])
    if seed_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=WIDE_FIGURE_SIZE, sharey=True)
    colors = {"Classical": CLASSICAL_COLOR, "Quantum": QUANTUM_COLOR}

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
            medianprops={"color": EDGE_COLOR, "linewidth": 1.7},
            meanprops={"color": EDGE_COLOR, "linewidth": 1.2, "linestyle": "--"},
        )

        for patch, label in zip(box["boxes"], ["Classical", "Quantum"]):
            patch.set_facecolor(colors[label])
            patch.set_alpha(0.9)
            patch.set_edgecolor(EDGE_COLOR)
            patch.set_linewidth(1.2)

        for key in ["whiskers", "caps"]:
            for line in box[key]:
                line.set_color(EDGE_COLOR)
                line.set_linewidth(1.0)

        _style_axes(ax, f"{dataset} (20 Seed Means)", "Paradigm")

    axes[0].set_ylabel("Accuracy (%)", fontsize=LABEL_SIZE, fontweight="bold")
    fig.suptitle("20-Seed Accuracy Distribution: Classical vs Quantum", fontsize=TITLE_SIZE, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, out_path)


def _parameter_summary_for_datasets(data: FamilyData, datasets: list[str]) -> pd.DataFrame:
    summary_rows: list[dict[str, object]] = []
    for dataset in datasets:
        paired = _paired_rows(data.classical_summary, data.quantum_summary, dataset)
        if paired.empty:
            continue

        summary_rows.append(
            {
                "dataset": dataset,
                "dataset_label": dataset_title(dataset),
                "classical_accuracy_pct": float(paired["classical_accuracy_pct"].mean()),
                "quantum_accuracy_pct": float(paired["quantum_accuracy_pct"].mean()),
                "classical_total_params": float(paired["classical_total_params"].mean()),
                "quantum_total_params": float(paired["quantum_total_params"].mean()),
            }
        )

    return pd.DataFrame(summary_rows)


def plot_leukemia_colon_parameter_vs_accuracy(data: FamilyData, out_path: Path) -> None:
    summary = _parameter_summary_for_datasets(data, ["leukemia", "colon"])
    if summary.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=WIDE_FIGURE_SIZE)
    x = np.arange(len(summary))
    width = 0.36

    axes[0].bar(
        x - width / 2,
        summary["classical_accuracy_pct"],
        width=width,
        label="Classical",
        color=CLASSICAL_COLOR,
        edgecolor=EDGE_COLOR,
    )
    axes[0].bar(
        x + width / 2,
        summary["quantum_accuracy_pct"],
        width=width,
        label="Quantum",
        color=QUANTUM_COLOR,
        edgecolor=EDGE_COLOR,
    )
    axes[0].set_xticks(x, summary["dataset_label"].tolist())
    _style_axes(axes[0], "Accuracy (Mean Across Architectures)", ylabel="Accuracy (%)")
    axes[0].legend(frameon=False, fontsize=LEGEND_SIZE)

    axes[1].bar(
        x - width / 2,
        summary["classical_total_params"],
        width=width,
        label="Classical Model",
        color=CLASSICAL_COLOR,
        edgecolor=EDGE_COLOR,
    )
    axes[1].bar(
        x + width / 2,
        summary["quantum_total_params"],
        width=width,
        label="Quantum Model",
        color=QUANTUM_COLOR,
        edgecolor=EDGE_COLOR,
    )

    for idx, row in summary.reset_index(drop=True).iterrows():
        axes[1].text(
            idx,
            max(row["classical_total_params"], row["quantum_total_params"]) * 1.02,
            "Equal Capacity\n(1-to-1 Ablation)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    axes[1].set_xticks(x, summary["dataset_label"].tolist())
    _style_axes(axes[1], "Total Trainable Parameters", ylabel="Parameter Count")
    axes[1].legend(frameon=False, fontsize=LEGEND_SIZE)
    axes[1].set_ylim(0, summary[["classical_total_params", "quantum_total_params"]].max().max() * 1.25)

    fig.suptitle(
        "1-to-1 Ablation: Accuracy Parity under Strict Parameter Constraints",
        fontsize=TITLE_SIZE,
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, out_path)


def plot_swiss_roll(data: FamilyData, out_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    paired = _paired_rows(data.classical_summary, data.quantum_summary, "swiss_roll")
    if paired.empty:
        return outputs

    outputs.append(out_dir / "swiss_roll_accuracy_pairs")
    _plot_metric_bars(
        paired,
        "swiss_roll",
        "classical_accuracy_pct",
        "quantum_accuracy_pct",
        "Accuracy (%)",
        "Swiss Roll: Classical vs Quantum Accuracy",
        outputs[-1],
        err_left_col="classical_accuracy_std_pct",
        err_right_col="quantum_accuracy_std_pct",
    )

    outputs.append(out_dir / "swiss_roll_parameter_parity")
    _plot_metric_bars(
        paired,
        "swiss_roll",
        "classical_total_params",
        "quantum_total_params",
        "Trainable Parameters",
        "Swiss Roll: Strict 1-to-1 Parameter Parity",
        outputs[-1],
    )

    seed_boxplot = out_dir / "swiss_roll_seed_boxplot"
    _plot_accuracy_boxplot(data.seed_runs, "swiss_roll", seed_boxplot)
    outputs.append(seed_boxplot)

    return outputs


def plot_leukemia_colon_new(data: FamilyData, out_dir: Path) -> list[Path]:
    outputs: list[Path] = []

    boxplot_path = out_dir / "leukemia_colon_20_seed_boxplot"
    plot_leukemia_colon_20_seed_boxplot(data, boxplot_path)
    outputs.append(boxplot_path)

    parameter_plot_path = out_dir / "leukemia_colon_parameter_vs_accuracy"
    plot_leukemia_colon_parameter_vs_accuracy(data, parameter_plot_path)
    outputs.append(parameter_plot_path)

    for dataset in data.datasets:
        paired = _paired_rows(data.classical_summary, data.quantum_summary, dataset)
        if paired.empty:
            continue

        boxplot_path = out_dir / f"{dataset}_seed_boxplot"
        _plot_accuracy_boxplot(data.seed_runs, dataset, boxplot_path)
        outputs.append(boxplot_path)

        if dataset == "leukemia":
            param_path = out_dir / "leukemia_parameter_parity"
            _plot_metric_bars(
                paired,
                dataset,
                "classical_total_params",
                "quantum_total_params",
                "Trainable Parameters",
                f"{dataset_title(dataset)}: Strict 1-to-1 Parameter Parity",
                param_path,
            )
            outputs.append(param_path)

    return outputs


def plot_all_datasets(data: FamilyData, out_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    summary = data.classical_summary.copy()
    summary_q = data.quantum_summary.copy()

    datasets = [dataset for dataset in data.datasets if dataset in set(summary["dataset"]).intersection(set(summary_q["dataset"]))]
    paired_frames = []
    for dataset in datasets:
        paired = _paired_rows(summary, summary_q, dataset)
        if paired.empty:
            continue
        paired["dataset"] = dataset
        paired_frames.append(paired)

    if not paired_frames:
        return outputs

    paired_all = pd.concat(paired_frames, ignore_index=True)

    def _plot_dataset_metric(metric_left: str, metric_right: str, ylabel: str, title: str, out_name: str, log_scale: bool = False) -> None:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        x = np.arange(len(datasets))
        width = 0.34
        dataset_summary = (
            paired_all.groupby("dataset", as_index=False)
            .agg(
                classical_accuracy_pct=("classical_accuracy_pct", "mean"),
                quantum_accuracy_pct=("quantum_accuracy_pct", "mean"),
                classical_training_time=("classical_training_time", "mean"),
                quantum_training_time=("quantum_training_time", "mean"),
                classical_inference_time=("classical_inference_time", "mean"),
                quantum_inference_time=("quantum_inference_time", "mean"),
                classical_total_params=("classical_total_params", "mean"),
                quantum_total_params=("quantum_total_params", "mean"),
            )
            .set_index("dataset")
            .reindex(datasets)
        )

        left = dataset_summary[metric_left].tolist()
        right = dataset_summary[metric_right].tolist()

        ax.bar(x - width / 2, left, width, color=CLASSICAL_COLOR, edgecolor=EDGE_COLOR, label="Classical")
        ax.bar(x + width / 2, right, width, color=QUANTUM_COLOR, edgecolor=EDGE_COLOR, label="Quantum")
        ax.set_xticks(x, [dataset_title(dataset) for dataset in datasets])
        _style_axes(ax, title, ylabel=ylabel)
        if log_scale:
            ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=LEGEND_SIZE)
        save_figure(fig, out_dir / out_name)

    _plot_dataset_metric(
        "classical_training_time",
        "quantum_training_time",
        "Training Time per Epoch (s)",
        "All Datasets: Training Time Comparison",
        "all_datasets_training_time",
        log_scale=True,
    )
    outputs.append(out_dir / "all_datasets_training_time")

    return outputs


def plot_appendix(data: FamilyData, out_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    if data.key != "all-datasets":
        return outputs

    summary = data.classical_summary.copy()
    summary_q = data.quantum_summary.copy()
    datasets = [dataset for dataset in data.datasets if dataset in set(summary["dataset"]).intersection(set(summary_q["dataset"]))]
    paired_frames = []
    for dataset in datasets:
        paired = _paired_rows(summary, summary_q, dataset)
        if paired.empty:
            continue
        paired["dataset"] = dataset
        paired_frames.append(paired)

    if not paired_frames:
        return outputs

    paired_all = pd.concat(paired_frames, ignore_index=True)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    x = np.arange(len(datasets))
    width = 0.34
    dataset_summary = (
        paired_all.groupby("dataset", as_index=False)
        .agg(
            classical_accuracy_pct=("classical_accuracy_pct", "mean"),
            quantum_accuracy_pct=("quantum_accuracy_pct", "mean"),
            classical_total_params=("classical_total_params", "mean"),
            quantum_total_params=("quantum_total_params", "mean"),
        )
        .set_index("dataset")
        .reindex(datasets)
    )
    left = dataset_summary["classical_accuracy_pct"].tolist()
    right = dataset_summary["quantum_accuracy_pct"].tolist()
    ax.bar(x - width / 2, left, width, color=CLASSICAL_COLOR, edgecolor=EDGE_COLOR, label="Classical")
    ax.bar(x + width / 2, right, width, color=QUANTUM_COLOR, edgecolor=EDGE_COLOR, label="Quantum")
    ax.set_xticks(x, [dataset_title(dataset) for dataset in datasets])
    _style_axes(ax, "All Datasets: Accuracy Comparison", ylabel="Accuracy (%)")
    ax.legend(frameon=False, fontsize=LEGEND_SIZE)
    save_figure(fig, out_dir / "all_datasets_accuracy")
    outputs.append(out_dir / "all_datasets_accuracy")

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    for _, row in paired_all.iterrows():
        ax.scatter(row["classical_total_params"], row["classical_accuracy_pct"], s=72, color=CLASSICAL_COLOR, edgecolor=EDGE_COLOR)
        ax.scatter(row["quantum_total_params"], row["quantum_accuracy_pct"], s=72, color=QUANTUM_COLOR, edgecolor=EDGE_COLOR)
        ax.text(row["quantum_total_params"], row["quantum_accuracy_pct"] + 0.6, f"{dataset_title(row['dataset'])}-{row['family']}", ha="center", fontsize=7)
    _style_axes(ax, "All Datasets: Accuracy vs Parameters", "Trainable Parameters", "Accuracy (%)")
    classical_legend = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=CLASSICAL_COLOR, markeredgecolor=EDGE_COLOR, markersize=8, label="Classical")
    quantum_legend = plt.Line2D([0], [0], marker="^", color="w", markerfacecolor=QUANTUM_COLOR, markeredgecolor=EDGE_COLOR, markersize=8, label="Quantum")
    ax.legend(handles=[classical_legend, quantum_legend], frameon=False, fontsize=LEGEND_SIZE)
    save_figure(fig, out_dir / "all_datasets_accuracy_vs_params")
    outputs.append(out_dir / "all_datasets_accuracy_vs_params")

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate thesis-ready plots from seed summaries.")
    parser.add_argument(
        "--family",
        choices=["swiss-roll", "leukemia-colon-new", "all-datasets", "all"],
        default="all",
        help="Which result family to plot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where figure files will be written.",
    )
    parser.add_argument(
        "--section",
        choices=["core", "appendix", "all"],
        default="core",
        help="Which curated figure set to generate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = [args.family] if args.family != "all" else ["swiss-roll", "leukemia-colon-new", "all-datasets"]

    for family_key in targets:
        data = load_family_data(family_key)
        out_dir = ensure_output_dir(args.output_dir, family_key)

        generated: list[Path] = []
        if args.section in {"core", "all"} and family_key in CORE_FAMILIES:
            if family_key == "swiss-roll":
                generated = plot_swiss_roll(data, out_dir)
            elif family_key == "leukemia-colon-new":
                generated = plot_leukemia_colon_new(data, out_dir)
            elif family_key == "all-datasets":
                generated = plot_all_datasets(data, out_dir)
        if args.section in {"appendix", "all"} and family_key in APPENDIX_FAMILIES:
            generated.extend(plot_appendix(data, out_dir))

        if not generated:
            print(f"[{family_key}] no figures selected for section={args.section}")
        else:
            print(f"[{family_key}] saved {len(generated)} plot groups in {out_dir}")


if __name__ == "__main__":
    main()