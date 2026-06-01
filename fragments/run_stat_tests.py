#!/usr/bin/env python3
import os
import glob
import pandas as pd
import numpy as np
from scipy import stats


def load_model_accuracies(folder, dataset, model_name, kind):
    pattern = os.path.join(folder, f"{kind}_results_*.csv")
    all_files = glob.glob(pattern)
    # exclude any summary files
    files = [p for p in all_files if not p.endswith("_seed_summary.csv")]

    def _extract_index(p: str):
        base = os.path.splitext(os.path.basename(p))[0]
        parts = base.split("_")
        try:
            return int(parts[-1])
        except Exception:
            for part in reversed(parts):
                if part.isdigit():
                    return int(part)
            return float("inf")

    files = sorted(files, key=_extract_index)

    accs = []
    seeds = []
    for p in files:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        row = df[(df.dataset == dataset) & (df.model == model_name)]
        if not row.empty:
            acc = float(row.iloc[0].accuracy)
            accs.append(acc)
            seeds.append(int(row.iloc[0].seed))
    return np.array(accs), seeds


def run_tests(classical, quantum, label):
    print(f"\n=== {label} ===")
    print(f"{'Group':<10} {'n':>4} {'Mean':>10} {'Std':>10}")
    print("-" * 38)
    print(
        f"{'Classical':<10} {len(classical):>4} {np.mean(classical):>10.4f} {np.std(classical, ddof=1):>10.4f}"
    )
    print(
        f"{'Quantum':<10} {len(quantum):>4} {np.mean(quantum):>10.4f} {np.std(quantum, ddof=1):>10.4f}"
    )

    # Welch's t-test
    tstat, tp = stats.ttest_ind(classical, quantum, equal_var=False, nan_policy="omit")
    # Mann-Whitney U (two-sided)
    try:
        U, up = stats.mannwhitneyu(classical, quantum, alternative="two-sided")
    except Exception:
        U, up = np.nan, np.nan

    print("\nStatistical tests")
    print(f"{'Welch t-test':<20} t = {tstat:>9.4f}, p = {tp:.4g}")
    print(f"{'Mann-Whitney U':<20} U = {U:>9.4f}, p = {up:.4g}")


def main():
    base = os.path.dirname(__file__)
    dat_folder = os.path.join(base, "results-leukemia-colon-new")
    dat_folder_ind = os.path.join(base, "results-svm-ind-bra")
    dat_folder_syn = os.path.join(base, "results-swiss-roll")

    # leuk_folder = os.path.join(base, 'results-leukemia-colon-new')

    classical_models = [
        "ClassicalMLP",
        "IterativeClassicalNN",
        "SplitAttentionClassicalNN",
        "ResNet",
        "CKAResCNet",
        # "ClassicalSVM"
    ]
    quantum_models = [
        "ClassicalQuantumMLP",
        "IterativeQNN",
        "SplitAttentionQNN",
        "ResQNet",
        "QKAResQNet",
        # "QuantumSVM"
    ]

    datasets = [
        # ("circles", dat_folder_syn, "circles"),
        # ("moons", dat_folder_syn, "moons"),
        # ("swiss_roll", dat_folder_syn, "swiss_roll"),
        # ("Brain RNA Seq", dat_folder_ind, "brain"),
        # ("Indian Spines", dat_folder_ind, "indian_pines_small"),
        # ("Breast", dat_folder, "breast"),
        # ("QSAR", dat_folder, "qsar"),
        ("Colon", dat_folder, "colon"),
        ("Leukemia", dat_folder, "leukemia"),
        # ("Ntangled", dat_folder, "ntangled"),
    ]

    print("\n" + "=" * 80)
    print("Statistical Comparison: Classical vs Quantum Models")
    print("=" * 80)

    for classical_name, quantum_name in zip(classical_models, quantum_models):
        print("\n" + "#" * 80)
        print(f"Model Pair: {classical_name}  vs  {quantum_name}")
        print("#" * 80)

        for pretty_name, folder, dataset_key in datasets:
            c_vals, _ = load_model_accuracies(
                folder, dataset_key, classical_name, "classical"
            )
            q_vals, _ = load_model_accuracies(
                folder, dataset_key, quantum_name, "quantum"
            )

            if len(c_vals) == 0 or len(q_vals) == 0:
                print(
                    f"\n=== {pretty_name}: missing data for one or both models "
                    f"(classical={len(c_vals)}, quantum={len(q_vals)}) ==="
                )
                continue

            run_tests(c_vals, q_vals, pretty_name)


if __name__ == "__main__":
    main()
