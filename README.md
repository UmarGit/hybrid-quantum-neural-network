# Hybrid Quantum Neural Network Benchmarking

Research code for the master's thesis:
**"The Development of an Experimental Benchmark for Hybrid Quantum-Classical Neural Networks Using Kernel Methods"**
ITMO University, 2026 — Ahmed Umar, supervised by Ivan Vladimirovich Khodnenko.

## Core Research Question

Not "is quantum always better?" but precisely:

- **When** do hybrid quantum blocks provide statistically significant utility over their classical twins?
- **When** are they neutral (representational parity)?
- **When** do they actively degrade performance?

The benchmark answers this by mapping results onto a 2×2 empirical matrix of **data topology** (geometric vs. tabular) × **data volume** (sufficient vs. extreme p >> N starvation), producing four quantified operational boundaries validated by Welch's t-test and Mann-Whitney U across 19–20 random seeds.

## Thesis and Preprint

| Document | File |
|---|---|
| Master's Thesis | [thesis.pdf](thesis.pdf) |
| Preprint (compiled) | [preprint.pdf](preprint.pdf) |

## Key Results

All results use strict 1-to-1 parameter ablation — classical and quantum twins differ by exactly 8 parameters for residual pairs and 28 for split-attention pairs, isolating the quantum component. The six core datasets are simulated under the empirically calibrated noise profile of the 133-qubit IBM Torino processor (Heron r1: single-qubit error 0.033%, two-qubit error 0.62%, readout error 3.03%); the gesture experiment additionally uses the 156-qubit IBM Kingston profile (Heron r2: readout error 1.09%) and physical Kingston hardware.

### Boundary 1 — Geometric Utility (Swiss Roll, 20 seeds)

| Architecture Pair | Classical | Quantum | Δ (pp) | Welch p | MW p |
|---|---|---|---|---|---|
| SplitAttn | 67.29% ± 11.8% | **91.77% ± 4.4%** | +24.48 | 6.5×10⁻⁹ | 3.6×10⁻⁷ |
| QKAResQNet | 67.73% ± 11.8% | **82.16% ± 9.2%** | +14.42 | 1.2×10⁻⁴ | 2.5×10⁻⁴ |
| ResQNet | 69.93% ± 14.2% | **82.56% ± 7.8%** | +12.62 | 1.6×10⁻³ | 7.7×10⁻³ |
| QuantumMLP | 75.32% ± 12.1% | **84.44% ± 9.3%** | +9.12 | 0.011 | 0.017 |
| IterativeQNN | 69.56% ± 9.4% | **76.74% ± 10.1%** | +7.18 | 0.025 | 0.035 |

Quantum twins outperform their parameter-matched classical twins across **all 5 pairs** on topologically complex geometry under data starvation. All hybrids nonetheless remain below the exact classical RBF-SVM ceiling (98.89% on Swiss Roll), so they act as scalable O(N) approximations of O(N³) kernel methods rather than surpassing them.

### Boundary 2 — Representational Parity (Leukemia & Colon, 19 seeds)

Residual architectures (ResQNet, QKAResQNet) achieve statistical parity with tighter variance — implicit regularization from unitary constraints.

| Architecture | Dataset | Classical | Quantum | Δ | Welch p | Verdict |
|---|---|---|---|---|---|---|
| ResQNet | Leukemia | 71.42% ± 11.2% | 68.91% ± 10.0% | −2.51 | 0.471 | **Parity** |
| ResQNet | Colon | 63.63% ± 11.4% | 65.13% ± 8.7% | +1.50 | 0.651 | **Parity** |
| QKAResQNet | Leukemia | 70.85% ± 11.4% | 69.55% ± 9.1% | −1.30 | 0.701 | **Parity** |
| QKAResQNet | Colon | 66.26% ± 7.7% | 63.35% ± 11.1% | −2.91 | 0.353 | **Parity** |

### Boundary 3 — Execution Time Penalty

Total training time over the full 20-epoch run (not per epoch):

| Dataset | Classical (s) | Quantum (s) | Slowdown |
|---|---|---|---|
| Swiss Roll | 0.008 | 255.5 | ×31,900 |
| QSAR | 0.008 | 419.0 | ×52,400 |
| Leukemia | 0.015 | 27.3 | ×1,817 |
| Colon | 0.011 | 23.6 | ×2,141 |
| NTangled | 0.007 | 205.8 | ×29,400 |

### Boundary 4 — Classical Decoherence

All hybrid architectures stall at ~50% (random guessing) on NTangled GHZ-state data. The PyTorch linear compression layer irreversibly destroys multipartite entanglement before the quantum circuit processes it.

### Additional Finding — Inductive Bias Misalignment Penalty

SplitAttentionQNN is **significantly worse** than classical on genomic data — confirming that architectural mismatch actively degrades performance, not merely fails to help.

| Dataset | Classical | Quantum | Δ | Welch p | MW p |
|---|---|---|---|---|---|
| Leukemia | 65.51% | 59.11% | −6.40 | 0.0016 | 0.0004 |
| Colon | 59.40% | 49.15% | −10.24 | 0.0098 | 0.0012 |

### Physical Hardware Validation (IBM Kingston, Heron r2)

QKAResQNet was deployed for inference on the physical 156-qubit IBM Kingston processor using real-world ESP32 accelerometer gesture data, evaluated under three conditions:

| Condition | Classical | Quantum |
|---|---|---|
| Noiseless simulation | 87.58% ± 7.1% | **89.39% ± 6.5%** |
| IBM Torino noise (sim) | **89.55% ± 9.9%** | 87.73% ± 10.3% |
| IBM Kingston (hardware) | 85.71% | **92.86%** |

On physical hardware, QKAResQNet inference reached **92.86%**, matching the classical RBF-SVM ceiling and exceeding both its classical twin and its own noisy-simulator inference (85.71%). The batch took 84 s of wall-clock time, dominated by CPU–QPU I/O latency rather than gate execution — identifying classical-quantum communication overhead as the principal deployment bottleneck. As a 5-fold CV result on N=56, this is a directional signal pending larger labeled datasets.

## Repository Layout

```
.
├── LICENSE
├── README.md
├── requirements.txt
├── pyproject.toml                  # Package metadata and pytest config
├── main.py                         # Entry point for the modular benchmark
│
├── models/                         # Model definitions (reused by the benchmark)
│   ├── __init__.py
│   ├── classical_models.py         # All classical architectures
│   ├── quantum_models.py           # All hybrid quantum architectures
│   ├── svm_models.py               # Classical + quantum SVM baselines
│   └── datasets.py                 # Dataset loaders and preprocessing
│
├── benchmark/                      # Modular benchmarking package
│   ├── __init__.py
│   ├── config.py                   # BenchmarkConfig + matched-pair registry
│   ├── data.py                     # DataStarvationModule (stratified, leak-free split)
│   ├── ablation.py                 # AblationValidator (parameter-parity guard)
│   ├── noise.py                    # IBMNoiseModel (Torino/Kingston/noiseless)
│   ├── engine.py                   # ExecutionEngine (N-seed loop, CSV+JSON)
│   └── stats.py                    # StatisticalAnalyzer (Welch + Mann-Whitney + verdict)
│
├── tests/                          # Pytest suite for the benchmark package
│   ├── test_ablation.py
│   ├── test_noise.py
│   └── test_stats.py
│
├── fragments/                      # Legacy thesis scripts (reused by the package)
│   ├── quantum.py                  # Quantum circuit builder + training (get_quantum_circuit)
│   ├── classical.py                # Classical training and evaluation script
│   ├── collect_tables.py           # Multi-seed benchmarking runner
│   ├── aggregate_seed_results.py   # Seed summary CSV generation
│   ├── run_stat_tests.py           # Welch t-test + Mann-Whitney U across model pairs
│   ├── 5-fold-cv-test.py           # 5-fold cross-validation experiment
│   ├── plots.py                    # Figure generation from benchmark CSVs
│   ├── thesis_plots.py             # Thesis-specific plot generation
│   ├── thesis-plots.py             # Thesis plot variant
│   ├── final-plots.py              # Final publication-ready figures
│   └── preprint-old.pdf
│
├── notebooks/                      # Early exploratory experiments (reference only)
│   ├── circle-full-experiment-arch.ipynb
│   ├── gesture-full-experiment-arch-hardware.ipynb
│   ├── iris-full-experiment-arch.ipynb
│   └── moons-full-experiment-arch.ipynb
│
├── plots/
│   ├── resqnet_quantum_bottleneck.png
│   └── thesis/
│       ├── all-datasets/
│       │   └── all_datasets_training_time.png
│       ├── leukemia-colon-new/
│       │   ├── colon_seed_boxplot.png
│       │   ├── leukemia_colon_20_seed_boxplot.png
│       │   ├── leukemia_colon_parameter_vs_accuracy.png
│       │   ├── leukemia_parameter_parity.png
│       │   └── leukemia_seed_boxplot.png
│       └── swiss-roll/
│           ├── swiss_roll_accuracy_pairs.png
│           ├── swiss_roll_parameter_parity.png
│           └── swiss_roll_seed_boxplot.png
│
├── thesis.pdf
└── preprint.pdf
```

> The modular benchmark (`main.py` + `benchmark/`) is the supported entry point.
> The `fragments/` directory holds the original thesis scripts; `benchmark/`
> reuses `fragments/quantum.py` (`get_quantum_circuit`, `seed_everything`) by
> import. Run everything from the repository root so `models` and `fragments`
> resolve.

## Architectures

Five parameter-matched classical/quantum pairs are evaluated:

| Classical | Quantum | Description |
|---|---|---|
| `ClassicalMLP` | `ClassicalQuantumMLP` | Standard MLP with quantum bottleneck |
| `IterativeClassicalNN` | `IterativeQNN` | Iterative feature processing |
| `SplitAttentionClassicalNN` | `SplitAttentionQNN` | Spatial chunking for high-dim inputs |
| `ResNet` | `ResQNet` | Classical residual highway bypassing barren plateaus |
| `CKAResCNet` | `QKAResQNet` | ResQNet with trainable Quantum Kernel Alignment feature map |

The quantum bottleneck in all cases: a **4-qubit ZFeatureMap + RealAmplitudes ansatz** operating in a 16-dimensional Hilbert space. Core datasets are simulated under the IBM Torino noise profile; QKAResQNet is additionally validated on physical IBM Kingston hardware.

## Datasets

| Dataset | n | p | p/N | Role in benchmark |
|---|---|---|---|---|
| Swiss Roll | 500 | 2 | — | Complex geometry — topological utility testbed |
| Circles | 500 | 2 | — | Complex geometry — radial separability |
| Moons | 500 | 2 | — | Complex geometry — interleaved separability |
| Breast Cancer | 569 | 30 | — | Standard tabular — parity sanity check |
| QSAR | 1,055 | 41 | — | Standard tabular — cheminformatics |
| Leukemia | 72 | 7,129 | 99:1 | Extreme p ≫ N — genomic starvation |
| Colon Cancer | 62 | 2,000 | 32:1 | Extreme p ≫ N — genomic starvation |
| Brain RNA-seq | 100 | 23,433 | 234:1 | Extreme p ≫ N — 7-class imbalanced genomics |
| Indian Pines | 100 | 200 | 2:1 | High-dim hyperspectral imaging |
| NTangled | 400 | — | — | Quantum-native GHZ states — decoherence boundary |
| Gesture (ESP32) | 56 | 96 | — | Real-world geometry — physical-hardware validation |

All datasets use a 90/10 train/test starvation split applied uniformly.

## Quickstart

```bash
git clone <repo-url>
cd quantum-neural-network

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Legacy thesis scripts now live under fragments/ — run them as modules
# from the repo root so `models` resolves.

# Run full multi-seed benchmark
python -m fragments.collect_tables

# Aggregate seed-level results into summary CSVs
python -m fragments.aggregate_seed_results

# Run statistical significance tests (Welch t-test + Mann-Whitney U)
python -m fragments.run_stat_tests

# Generate all figures
python -m fragments.plots
python -m fragments.thesis_plots
```

Conda alternative:

```bash
conda activate qnn
pip install -r requirements.txt
python -m fragments.collect_tables
```

## Running the Benchmark (modular package)

The `benchmark/` package wraps the thesis models in a reproducible multi-seed
runner. It is driven by `main.py`, which benchmarks **one dataset** against
**one matched architecture pair** across N seeds and prints a statistical
verdict.

```bash
conda activate qnn

# Breast Cancer, ResQNet vs ResNet, 20 seeds, noiseless simulator
python main.py --dataset breast --pair resnet --seeds 20

# Swiss Roll, kernel-alignment pair, under the IBM Torino noise profile
python main.py --dataset swiss_roll --pair kernel_alignment --noise torino

# Split-attention with custom chunking and a larger residual gate
python main.py --dataset qsar --pair split_attention --n-chunks 5 --gate-limit 0.3
```

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--dataset` | *(required)* | Any `models/datasets.py` choice (`breast`, `swiss_roll`, `qsar`, `leukemia`, `colon`, `circles`, ...) |
| `--pair` | `resnet` | Matched pair: `mlp`, `iterative`, `split_attention`, `resnet`, `kernel_alignment`, `svm` |
| `--seeds` | `20` | Number of random seeds (each trains both models) |
| `--epochs` | `50` | Training epochs (ignored for the `svm` pair) |
| `--lr` | `0.01` | Adam learning rate |
| `--test-split` | `0.9` | Test fraction (the data-starvation split) |
| `--num-qubits` | `4` | Qubits / quantum-bottleneck width |
| `--n-chunks` | `3` | Split-attention chunk count |
| `--gate-limit` | `0.1` | Initial residual-gate value for the kernel-alignment pair |
| `--noise` | `noiseless` | `noiseless`, `torino`, or `kingston` |
| `--output-dir` | `benchmark_results` | Where per-seed CSVs and summaries are written |
| `--aggregate` / `--no-aggregate` | `--aggregate` | Roll per-seed CSVs into `*_seed_summary.csv` after the run |

### What it does per run

1. **Loads + splits** the dataset with `StandardScaler` fit on the train
   partition only (no leakage), stratified and seed-reproducible.
2. **Validates ablation parity** — asserts the quantum/classical pair differs
   by exactly the expected parameter delta (8 for most pairs; 48 for
   split-attention at `n_chunks=3`) so the comparison isolates the quantum layer.
3. **Trains both models** under identical epochs / optimizer / lr / split for
   each of N seeds, seeding torch, numpy, and the Aer simulator.
4. **Writes** per-seed `classical_results_<seed>.csv` and
   `quantum_results_<seed>.csv` to the output dir, in the same 18-column schema
   as `fragments/collect_tables.py` (timestamp, dataset, model, accuracy,
   training_time_sec, inference_time_sec, epochs, test_size, learning_rate,
   hidden_dim, n_chunks, num_qubits, input_dim, num_classes, train_size,
   test_count, total_params, seed) — directly consumable by
   `fragments.aggregate_seed_results` and `fragments.run_stat_tests`.
5. **Aggregates** (unless `--no-aggregate`) the per-seed CSVs into
   `classical_results_seed_summary.csv` and `quantum_results_seed_summary.csv`
   via `fragments.aggregate_seed_results` — adding accuracy/timing means and
   stds plus the estimated IBM Torino QPU time for the quantum models.
6. **Prints** the stat matrix: per-group mean/std, Welch t-test p, Mann-Whitney
   U p, and a `quantum wins` / `classical wins` / `parity` verdict.

> Note: the `svm` pair requires `--num-qubits` to equal the dataset's feature
> count (`QuantumSVM` enforces feature_dim == num_qubits). The `torino` and
> `kingston` noise profiles use a density-matrix simulator and are markedly
> slower than `noiseless`.

### Running the tests

```bash
conda activate qnn
pip install pytest
python -m pytest tests/ -q
```

## Reproducing Specific Results

### Swiss Roll (20-seed, all architecture pairs)

```bash
python -m fragments.collect_tables --dataset swiss_roll --seeds 20
python -m fragments.aggregate_seed_results
python -m fragments.run_stat_tests  # outputs Welch p and Mann-Whitney U per pair
```

### Leukemia / Colon (19-seed, geometric seed sequence)

```bash
python -m fragments.collect_tables --dataset leukemia colon --seeds 20
python -m fragments.aggregate_seed_results
python -m fragments.run_stat_tests
```

Note: Leukemia and Colon use a geometric seed sequence (2, 4, 6, 8, 16, 32, ...) to increase diversity across the extreme starvation regime. Swiss Roll uses a sequential sequence (42, 43, 44, ...). Both protocols satisfy the independence assumption of Welch's t-test.

### Statistical Tests Only

```bash
python -m fragments.run_stat_tests
```

Outputs Welch t-statistic, p-value, Mann-Whitney U and p-value for each classical/quantum pair across Swiss Roll, Leukemia, and Colon.

## Tested Environment

| Dependency | Version |
|---|---|
| Python | 3.11 |
| OS | macOS |
| PyTorch | latest stable |
| Qiskit | latest stable |
| qiskit-machine-learning | latest stable |

## Hardware vs. Simulator

Default: simulator with IBM Torino noise model (no IBM account required).

For physical hardware execution, an IBM Quantum account and valid runtime credentials are required. Configure via the Qiskit IBM Runtime service before running `fragments/quantum.py`.

## Troubleshooting

| Problem | Solution |
|---|---|
| Import errors | Reinstall dependencies in a clean environment: `pip install -r requirements.txt` |
| Qiskit backend / auth failures | Run in simulator mode first; configure IBM Quantum credentials for hardware |
| Missing CSVs | Run `python -m fragments.collect_tables` before `python -m fragments.aggregate_seed_results` |
| Slow runtime | Reduce epoch count for smoke tests before full runs; quantum simulation is ~1,800–52,000× slower than classical over a full run by design |
| `fragments.run_stat_tests` returns empty | Verify result CSV paths in script match your local `results/` and `results-lk-col/` directories |

## Early Exploratory Notebooks

The `notebooks/` directory (`iris`, `moons`, `circles`, `gesture`) holds early-stage experiments predating the main benchmark pipeline. They use older code and are retained for reference only. The canonical results are produced by the modular benchmark (`python main.py`) or the legacy `fragments/` workflow (`fragments.collect_tables` → `fragments.aggregate_seed_results` → `fragments.run_stat_tests` → `fragments.plots`).

## License

MIT. See [LICENSE](LICENSE).

## Citation

If you use this repository, please cite:

```
Umar, A. (2026). The Development of an Experimental Benchmark for Hybrid
Quantum-Classical Neural Networks Using Kernel Methods.
Master's Thesis, ITMO University, Saint Petersburg.
```