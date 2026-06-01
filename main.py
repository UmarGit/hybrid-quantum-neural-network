#!/usr/bin/env python3
"""Run a 20-seed benchmark for one dataset and one matched pair.

Example
-------
    conda activate qnn
    python main.py --dataset breast --pair resnet --seeds 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark.config import PAIR_REGISTRY, BenchmarkConfig
from benchmark.engine import ExecutionEngine
from benchmark.stats import StatisticalAnalyzer

# Reuse the legacy aggregation logic; do not reimplement it.
from fragments.aggregate_seed_results import save_aggregated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="datasets.load_dataset choice")
    parser.add_argument(
        "--pair", default="resnet", choices=sorted(PAIR_REGISTRY), help="matched pair"
    )
    parser.add_argument("--seeds", type=int, default=20, help="number of seeds")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--test-split", type=float, default=0.9)
    parser.add_argument("--num-qubits", type=int, default=4)
    parser.add_argument(
        "--n-chunks", type=int, default=3, help="split-attention chunk count"
    )
    parser.add_argument(
        "--gate-limit",
        type=float,
        default=0.1,
        help="initial residual gate value for the kernel-alignment pair",
    )
    parser.add_argument(
        "--entanglement",
        default="full",
        choices=["linear", "full", "circular"],
        help="QNN ansatz entanglement (collect_tables used 'full')",
    )
    parser.add_argument(
        "--noise", default="noiseless", choices=["noiseless", "torino", "kingston"]
    )
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument(
        "--aggregate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write classical/quantum *_seed_summary.csv after the run",
    )
    return parser.parse_args()


def aggregate(output_dir: str) -> None:
    """Roll the per-seed CSVs up into seed-summary tables (reused fragment)."""
    results_dir = Path(output_dir).expanduser()
    for prefix in ("classical", "quantum"):
        try:
            summary_path = save_aggregated(results_dir, prefix)
        except FileNotFoundError:
            print(f"Skipped {prefix} aggregation: no {prefix}_results_*.csv found.")
            continue
        print(f"Wrote {summary_path}")


def print_stat_matrix(dataset: str, pair_name: str, result) -> None:
    print("\n" + "=" * 72)
    print(f"Statistical comparison — dataset={dataset}  pair={pair_name}")
    print("=" * 72)
    header = f"{'Group':<12}{'Mean':>12}{'Std':>12}"
    print(header)
    print("-" * len(header))
    print(f"{'Classical':<12}{result.classical_mean:>12.4f}{result.classical_std:>12.4f}")
    print(f"{'Quantum':<12}{result.quantum_mean:>12.4f}{result.quantum_std:>12.4f}")
    print("-" * len(header))
    print(f"Welch t-test       p = {result.welch_p:.4g}")
    print(f"Mann-Whitney U     p = {result.mannwhitney_p:.4g}")
    print(f"Significant (a=.05): {result.significant}")
    print(f"VERDICT            : {result.verdict.upper()}")
    print("=" * 72)


def main() -> None:
    args = parse_args()
    pair = PAIR_REGISTRY[args.pair]

    config = BenchmarkConfig(
        dataset=args.dataset,
        n_seeds=args.seeds,
        epochs=args.epochs,
        learning_rate=args.lr,
        test_split=args.test_split,
        num_qubits=args.num_qubits,
        n_chunks=args.n_chunks,
        gate_limit=args.gate_limit,
        entanglement=args.entanglement,
        noise_mode=args.noise,
        output_dir=args.output_dir,
    )

    engine = ExecutionEngine(config)
    q_summary, c_summary = engine.run(pair)

    written = engine.write_results()
    print(f"\nWrote {len(written)} per-seed CSV file(s) to {config.output_dir}/")

    if args.aggregate:
        aggregate(config.output_dir)

    result = StatisticalAnalyzer().compare(
        c_summary.accuracies, q_summary.accuracies
    )
    print_stat_matrix(args.dataset, pair.name, result)


if __name__ == "__main__":
    main()
