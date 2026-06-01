from csv import DictReader
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
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
QUANTUM_COLOR = "#BB8BFF"
EDGE_COLOR = "#111827"
TABULAR_COLOR = "#64748B"
SPARSITY_COLOR = "#B45309"
NONLINEAR_COLOR = "#7C3AED"
MANIFOLD_COLOR = "#0E7490"
QUANTUM_NATIVE_COLOR = "#FF8C00"
FIGURE_SIZE = (10, 6)
TITLE_SIZE = 13
LABEL_SIZE = 11
LEGEND_SIZE = 10

ROOT = Path(__file__).resolve().parent
DATA_FILE = (
    ROOT
    / "results-final"
    / "Quantum-Classical Model Ablation Study - full-datasets.csv"
)
OUTPUT_DIR = ROOT / "results-final"


def _parse_float(value: str) -> float:
    value = (value or "").strip()
    if not value:
        return float("nan")
    return float(value)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            row for row in DictReader(handle) if row.get("dataset") and row.get("model")
        ]


def first_row(rows: list[dict[str, str]], dataset: str, model: str) -> dict[str, str]:
    for row in rows:
        if row["dataset"] == dataset and row["model"] == model:
            return row
    raise KeyError(f"Missing row for dataset={dataset!r}, model={model!r}")


def best_row(
    rows: list[dict[str, str]], dataset: str, models: list[str]
) -> dict[str, str]:
    candidates = [
        row for row in rows if row["dataset"] == dataset and row["model"] in models
    ]
    if not candidates:
        raise KeyError(f"No rows found for dataset={dataset!r} in {models!r}")
    return max(candidates, key=lambda row: _parse_float(row["accuracy_mean_pct"]))


def save_figure(fig: plt.Figure, filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def plot_slide_6(rows: list[dict[str, str]]) -> None:
    dataset_order = [
        "circles",
        "moons",
        "swiss_roll",
        "breast",
        "qsar",
        "indian_pines_small",
        "colon",
        "leukemia",
        "brain",
        "ntangled",
    ]
    dataset_labels = {
        "circles": "Circles",
        "moons": "Moons",
        "swiss_roll": "Swiss Roll",
        "breast": "Breast Cancer",
        "qsar": "QSAR",
        "indian_pines_small": "Indian Pines",
        "colon": "Colon",
        "leukemia": "Leukemia",
        "brain": "Brain RNA-seq",
        "ntangled": "NTangled",
    }
    topology_groups = {
        "circles": (1, "Synthetic Geometry", NONLINEAR_COLOR),
        "moons": (1, "Synthetic Geometry", NONLINEAR_COLOR),
        "swiss_roll": (1, "Synthetic Geometry", NONLINEAR_COLOR),
        "breast": (2, "Standard Tabular", TABULAR_COLOR),
        "qsar": (2, "Standard Tabular", TABULAR_COLOR),
        "indian_pines_small": (3, "Real-World High-Dim", CLASSICAL_COLOR),
        "colon": (3, "Real-World High-Dim", CLASSICAL_COLOR),
        "leukemia": (3, "Real-World High-Dim", CLASSICAL_COLOR),
        "brain": (3, "Real-World High-Dim", CLASSICAL_COLOR),
        "ntangled": (4, "Quantum-Native", QUANTUM_NATIVE_COLOR),
    }

    points = []
    for dataset in dataset_order:
        row = next(candidate for candidate in rows if candidate["dataset"] == dataset)
        topology_score, topology_label, color = topology_groups[dataset]
        feature_dim = _parse_float(row["input_dim"])
        sample_count = _parse_float(row["train_size"]) + _parse_float(row["test_count"])
        points.append(
            {
                "dataset": dataset_labels[dataset],
                "x": feature_dim,
                "y": topology_score,
                "size": max(sample_count, 1.0),
                "color": color,
                "group": topology_label,
            }
        )

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for point in points:
        ax.scatter(
            point["x"],
            point["y"],
            s=point["size"] * 1.2,
            c=point["color"],
            alpha=0.75,
            edgecolors=EDGE_COLOR,
            linewidths=0.8,
        )
        ax.annotate(
            point["dataset"],
            (point["x"], point["y"]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xscale("log")
    ax.set_xlabel(
        "Feature Dimensionality $p$ (log scale)", fontsize=LABEL_SIZE, fontweight="bold"
    )
    ax.set_ylabel("Topological Complexity", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_title(
        "Dataset Dimensionality vs. Topology", fontsize=TITLE_SIZE, fontweight="bold"
    )
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(
        [
            "Synthetic Geometry",
            "Standard Tabular",
            "Real-World High-Dim",
            "Quantum-Native",
        ],
        fontsize=10,
    )
    ax.set_ylim(0.5, 4.5)
    ax.grid(True, which="both", axis="x", linestyle="--", alpha=0.3)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Synthetic Geometry",
            markerfacecolor=NONLINEAR_COLOR,
            markeredgecolor=EDGE_COLOR,
            markersize=9,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Standard Tabular",
            markerfacecolor=TABULAR_COLOR,
            markeredgecolor=EDGE_COLOR,
            markersize=9,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Real-World High-Dim",
            markerfacecolor=CLASSICAL_COLOR,
            markeredgecolor=EDGE_COLOR,
            markersize=9,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Quantum-Native",
            markerfacecolor=QUANTUM_NATIVE_COLOR,
            markeredgecolor=EDGE_COLOR,
            markersize=9,
        ),
    ]
    ax.legend(
        handles=legend_handles, fontsize=LEGEND_SIZE, frameon=False, loc="lower right"
    )
    ax.text(
        0.02,
        -0.19,
        "Bubble size reflects total sample count. The x-axis is logarithmic to span 2 to 23,433 features.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    plt.tight_layout()
    save_figure(fig, "slide_6_dimensionality_topology.png")


def plot_slide_8(rows: list[dict[str, str]]) -> None:
    datasets = ["circles", "moons", "swiss_roll"]
    labels = ["Circles", "Moons", "Swiss Roll"]
    class_models = [
        "ClassicalMLP",
        "IterativeClassicalNN",
        "SplitAttentionClassicalNN",
        "ResNet",
        "CKAResCNet",
    ]
    quantum_models = [
        "ClassicalQuantumMLP",
        "IterativeQNN",
        "SplitAttentionQNN",
        "ResQNet",
        "QKAResQNet",
    ]

    classical_means = []
    classical_stds = []
    quantum_means = []
    quantum_stds = []
    svm_means = []
    svm_stds = []

    for dataset in datasets:
        classical_best = best_row(rows, dataset, class_models)
        quantum_best = best_row(rows, dataset, quantum_models)
        svm_row = first_row(rows, dataset, "ClassicalSVM")

        classical_means.append(_parse_float(classical_best["accuracy_mean_pct"]))
        classical_stds.append(_parse_float(classical_best["accuracy_std_pct"]))
        quantum_means.append(_parse_float(quantum_best["accuracy_mean_pct"]))
        quantum_stds.append(_parse_float(quantum_best["accuracy_std_pct"]))
        svm_means.append(_parse_float(svm_row["accuracy_mean_pct"]))
        svm_stds.append(_parse_float(svm_row["accuracy_std_pct"]))

    x = np.arange(len(labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.bar(
        x - width,
        classical_means,
        width,
        yerr=classical_stds,
        capsize=5,
        label="Best Classical NN",
        color=CLASSICAL_COLOR,
        edgecolor=EDGE_COLOR,
    )
    ax.bar(
        x,
        quantum_means,
        width,
        yerr=quantum_stds,
        capsize=5,
        label="Best Hybrid QNN",
        color=QUANTUM_COLOR,
        edgecolor=EDGE_COLOR,
    )
    ax.bar(
        x + width,
        svm_means,
        width,
        yerr=svm_stds,
        capsize=5,
        label="Classical SVM",
        color=EDGE_COLOR,
        edgecolor=EDGE_COLOR,
    )

    ax.set_ylabel("Mean Accuracy (%)", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_title(
        "Geometric Alignment: The Staircase Plot",
        fontsize=TITLE_SIZE,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=LEGEND_SIZE, frameon=False, ncols=3)
    plt.tight_layout()
    save_figure(fig, "slide_8_geometric_alignment.png")


def plot_slide_9(rows: list[dict[str, str]]) -> None:
    datasets = ["brain", "indian_pines_small"]
    labels = ["Brain RNA-seq", "Indian Pines"]
    classical_model = "CKAResCNet"
    quantum_model = "QKAResQNet"

    classical_means = []
    classical_stds = []
    quantum_means = []
    quantum_stds = []

    for dataset in datasets:
        classical_row = first_row(rows, dataset, classical_model)
        quantum_row = first_row(rows, dataset, quantum_model)
        classical_means.append(_parse_float(classical_row["accuracy_mean_pct"]))
        classical_stds.append(_parse_float(classical_row["accuracy_std_pct"]))
        quantum_means.append(_parse_float(quantum_row["accuracy_mean_pct"]))
        quantum_stds.append(_parse_float(quantum_row["accuracy_std_pct"]))

    x = np.arange(len(labels))
    width = 0.32

    fig, ax = plt.subplots(figsize=(9.6, 6))
    ax.bar(
        x - width / 2,
        classical_means,
        width,
        yerr=classical_stds,
        capsize=7,
        label="Classical Twin (CKAResCNet)",
        color=CLASSICAL_COLOR,
        edgecolor=EDGE_COLOR,
        alpha=0.9,
    )
    ax.bar(
        x + width / 2,
        quantum_means,
        width,
        yerr=quantum_stds,
        capsize=7,
        label="Quantum Twin (QKAResQNet)",
        color=QUANTUM_COLOR,
        edgecolor=EDGE_COLOR,
        alpha=0.9,
    )

    ax.set_ylabel("Validation Accuracy (%)", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_title(
        "Representational Parity: Overlapping Seed Statistics",
        fontsize=TITLE_SIZE,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=LEGEND_SIZE, frameon=False)
    ax.text(
        0.5,
        -0.18,
        "The source table stores mean ± std over 20 seeds, so the overlap is shown directly rather than as raw box plots.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )
    plt.tight_layout()
    save_figure(fig, "slide_9_representational_parity.png")


def plot_slide_10(rows: list[dict[str, str]]) -> None:
    datasets = ["swiss_roll", "qsar", "leukemia"]
    labels = ["Swiss Roll", "QSAR", "Leukemia"]
    classical_model = "ResNet"
    quantum_model = "ResQNet"

    classical_times = []
    quantum_times = []
    qpu_times = []

    for dataset in datasets:
        classical_row = first_row(rows, dataset, classical_model)
        quantum_row = first_row(rows, dataset, quantum_model)
        classical_times.append(_parse_float(classical_row["training_time_mean_sec"]))
        quantum_times.append(_parse_float(quantum_row["training_time_mean_sec"]))
        qpu_times.append(_parse_float(quantum_row["est_qpu_time_torino_sec"]))

    x = np.arange(len(labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        x - width,
        classical_times,
        width,
        label="Classical Simulation",
        color=CLASSICAL_COLOR,
        edgecolor=EDGE_COLOR,
    )
    ax.bar(
        x,
        quantum_times,
        width,
        label="QNN Simulation",
        color=QUANTUM_COLOR,
        edgecolor=EDGE_COLOR,
    )
    ax.bar(
        x + width,
        qpu_times,
        width,
        label="Estimated QPU Time",
        color=EDGE_COLOR,
        edgecolor=EDGE_COLOR,
    )

    ax.set_yscale("log")
    ax.set_ylabel(
        "Training / Execution Time (Seconds) [Log Scale]",
        fontsize=LABEL_SIZE,
        fontweight="bold",
    )
    ax.set_title(
        "Computational Scaling: Simulation Penalty vs QPU Reality",
        fontsize=TITLE_SIZE,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(fontsize=LEGEND_SIZE, frameon=False)
    ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    save_figure(fig, "slide_10_log_scale_scaling.png")


def main() -> None:
    rows = load_rows(DATA_FILE)
    plot_slide_6(rows)
    plot_slide_8(rows)
    plot_slide_9(rows)
    plot_slide_10(rows)
    print(f"Saved slide plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
