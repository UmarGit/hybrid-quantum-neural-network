# Hybrid Quantum Neural Network Benchmarking

Research code for the master's thesis:
**"The Development of an Experimental Benchmark for Hybrid Quantum-Classical Neural Networks Using Kernel Methods"**
ITMO University, 2026 — Ahmed Umar, supervised by Ivan Vladimirovich Khodnenko.

## Core Research Question

Not "is quantum always better?" but precisely:

- **When** do hybrid quantum blocks provide a statistically significant advantage?
- **When** are they neutral (representational parity)?
- **When** do they actively degrade performance?

The benchmark answers this by mapping results onto a 2×2 empirical matrix of **data topology** (geometric vs. tabular) × **data volume** (sufficient vs. extreme p >> N starvation), producing four quantified operational boundaries validated by Welch's t-test and Mann-Whitney U across 19–20 random seeds.

## Thesis and Preprint

| Document | File |
|---|---|
| Master's Thesis | [thesis.pdf](thesis.pdf) |
| Preprint (compiled) | [preprint.pdf](preprint.pdf) |

## Key Results

All results use strict 1-to-1 parameter ablation — classical and quantum twins differ by fewer than 28 parameters (< 0.1%). Quantum simulations incorporate the empirically calibrated noise profile of the 133-qubit IBM Torino processor (single-qubit error: 0.03%, two-qubit error: 0.62%, readout error: 3.03%).

### Boundary 1 — Topological Advantage (Swiss Roll, 20 seeds)

| Architecture Pair | Classical | Quantum | Δ (pp) | Welch p | MW p |
|---|---|---|---|---|---|
| SplitAttn | 67.29% ± 11.8% | **91.77% ± 4.4%** | +24.48 | 6.5×10⁻⁹ | 3.6×10⁻⁷ |
| QKAResQNet | 67.73% ± 11.8% | **82.16% ± 9.2%** | +14.43 | 1.2×10⁻⁴ | 2.5×10⁻⁴ |
| ResQNet | 69.93% ± 14.2% | **82.56% ± 7.8%** | +12.62 | 1.6×10⁻³ | 7.7×10⁻³ |
| QuantumMLP | 75.32% ± 12.1% | **84.44% ± 9.3%** | +9.12 | 0.011 | 0.017 |
| IterativeQNN | 69.56% ± 9.4% | **76.74% ± 10.1%** | +7.18 | 0.025 | 0.035 |

Quantum architectures outperform parameter-matched classical twins across **all 5 pairs** on topologically complex geometry under data starvation.

### Boundary 2 — Representational Parity (Leukemia & Colon, 19 seeds)

Residual architectures (ResQNet, QKAResQNet) achieve statistical parity with tighter variance — implicit regularization from unitary constraints.

| Architecture | Dataset | Classical | Quantum | Δ | Welch p | Verdict |
|---|---|---|---|---|---|---|
| ResQNet | Leukemia | 71.42% ± 11.2% | 68.91% ± 10.0% | −2.51 | 0.471 | **Parity** |
| ResQNet | Colon | 63.63% ± 11.4% | 65.13% ± 8.7% | +1.50 | 0.651 | **Parity** |
| QKAResQNet | Leukemia | 70.85% ± 11.4% | 69.55% ± 9.1% | −1.30 | 0.701 | **Parity** |
| QKAResQNet | Colon | 66.26% ± 7.7% | 63.35% ± 11.1% | −2.91 | 0.353 | **Parity** |

### Boundary 3 — Execution Time Penalty

| Dataset | Classical (s/epoch) | Quantum (s/epoch) | Slowdown |
|---|---|---|---|
| Swiss Roll | 0.008 | 255.5 | ×30,823 |
| QSAR | 0.008 | 419.0 | ×54,375 |
| Leukemia | 0.015 | 27.3 | ×1,874 |
| Colon | 0.011 | 23.6 | ×2,123 |
| NTangled | 0.007 | 205.8 | ×30,063 |

### Boundary 4 — Classical Decoherence

All hybrid architectures stall at ~50% (random guessing) on NTangled GHZ-state data. The PyTorch linear compression layer irreversibly destroys multipartite entanglement before the quantum circuit processes it.

### Additional Finding — Inductive Bias Misalignment Penalty

SplitAttentionQNN is **significantly worse** than classical on genomic data — confirming that architectural mismatch actively degrades performance, not merely fails to help.

| Dataset | Classical | Quantum | Δ | Welch p | MW p |
|---|---|---|---|---|---|
| Leukemia | 65.51% | 59.11% | −6.40 | 0.0016 | 0.0004 |
| Colon | 59.40% | 49.15% | −10.25 | 0.0098 | 0.0012 |

## Repository Layout

```
.
├── LICENSE
├── README.md
│
├── models/                         # Model definitions
│   ├── __init__.py
│   ├── classical_models.py         # All classical architectures
│   ├── quantum_models.py           # All hybrid quantum architectures
│   └── datasets.py                 # Dataset loaders and preprocessing
│
├── classical.py                    # Classical training and evaluation script
├── quantum.py                      # Quantum training and evaluation script
├── circuit.py                      # Quantum circuit definitions (ZFeatureMap + RealAmplitudes)
│
├── collect_tables.py               # Multi-seed benchmarking runner
├── aggregate_seed_results.py       # Seed summary CSV generation
├── run_stat_tests.py               # Welch t-test + Mann-Whitney U across model pairs
├── plots.py                        # Figure generation from benchmark CSVs
├── thesis_plots.py                 # Thesis-specific plot generation
├── final-plots.py                  # Final publication-ready figures
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
├── circles/                        # Early exploratory experiments (circles dataset)
├── moon/                           # Early exploratory experiments (moons dataset)
│
├── circle-full-experiment-arch.ipynb   # Notebook: circles experiment
├── iris-full-experiment-arch.ipynb     # Notebook: iris experiment
├── moons-full-experiment-arch.ipynb    # Notebook: moons experiment
│
├── requirements.txt
├── thesis.pdf
├── preprint.pdf
└── preprint-old.pdf
```

## Architectures

Five parameter-matched classical/quantum pairs are evaluated:

| Classical | Quantum | Description |
|---|---|---|
| `ClassicalMLP` | `ClassicalQuantumMLP` | Standard MLP with quantum bottleneck |
| `IterativeClassicalNN` | `IterativeQNN` | Iterative feature processing |
| `SplitAttentionClassicalNN` | `SplitAttentionQNN` | Spatial chunking for high-dim inputs |
| `ResNet` | `ResQNet` | Classical residual highway bypassing barren plateaus |
| `CKAResCNet` | `QKAResQNet` | ResQNet with trainable Quantum Kernel Alignment feature map |

The quantum bottleneck in all cases: **4-qubit ZFeatureMap + 3-layer RealAmplitudes ansatz** (12 trainable quantum parameters, 16-dimensional Hilbert space), simulated under IBM Torino noise profiles.

## Datasets

| Dataset | n | p | p/N | Role in benchmark |
|---|---|---|---|---|
| Swiss Roll | 400 | 3 | 0.0075 | Complex geometry — topological advantage testbed |
| Breast Cancer | 569 | 30 | 0.053 | Standard tabular — parity sanity check |
| QSAR | 1,055 | 41 | 0.039 | Standard tabular — cheminformatics |
| Leukemia | 72 | 7,129 | 99:1 | Extreme p >> N — genomic starvation |
| Colon Cancer | 62 | 2,000 | 32:1 | Extreme p >> N — genomic starvation |
| NTangled | 400 | — | — | Quantum-native GHZ states — decoherence boundary |

All datasets use a 90/10 train/test starvation split applied uniformly.

## Quickstart

```bash
git clone <repo-url>
cd quantum-neural-network

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run full multi-seed benchmark
python collect_tables.py

# Aggregate seed-level results into summary CSVs
python aggregate_seed_results.py

# Run statistical significance tests (Welch t-test + Mann-Whitney U)
python run_stat_tests.py

# Generate all figures
python plots.py
python thesis_plots.py
```

Conda alternative:

```bash
conda activate qnn
pip install -r requirements.txt
python collect_tables.py
```

## Reproducing Specific Results

### Swiss Roll (20-seed, all architecture pairs)

```bash
python collect_tables.py --dataset swiss_roll --seeds 20
python aggregate_seed_results.py
python run_stat_tests.py  # outputs Welch p and Mann-Whitney U per pair
```

### Leukemia / Colon (19-seed, geometric seed sequence)

```bash
python collect_tables.py --dataset leukemia colon --seeds 20
python aggregate_seed_results.py
python run_stat_tests.py
```

Note: Leukemia and Colon use a geometric seed sequence (2, 4, 6, 8, 16, 32, ...) to increase diversity across the extreme starvation regime. Swiss Roll uses a sequential sequence (42, 43, 44, ...). Both protocols satisfy the independence assumption of Welch's t-test.

### Statistical Tests Only

```bash
python run_stat_tests.py
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

For physical hardware execution, an IBM Quantum account and valid runtime credentials are required. Configure via the Qiskit IBM Runtime service before running `quantum.py`.

## Troubleshooting

| Problem | Solution |
|---|---|
| Import errors | Reinstall dependencies in a clean environment: `pip install -r requirements.txt` |
| Qiskit backend / auth failures | Run in simulator mode first; configure IBM Quantum credentials for hardware |
| Missing CSVs | Run `collect_tables.py` before `aggregate_seed_results.py` |
| Slow runtime | Reduce epoch count for smoke tests before full runs; quantum simulation is 1,900–54,000× slower than classical per epoch by design |
| `run_stat_tests.py` returns empty | Verify result CSV paths in script match your local `results/` and `results-lk-col/` directories |

## Early Exploratory Notebooks

The notebooks (`iris`, `moons`, `circles`) represent early-stage experiments predating the main benchmark pipeline. They use older code and are retained for reference only. The canonical results are produced by the script workflow (`collect_tables.py` → `aggregate_seed_results.py` → `run_stat_tests.py` → `plots.py`).

## License

MIT. See [LICENSE](LICENSE).

## Citation

If you use this repository, please cite:

```
Umar, A. (2026). The Development of an Experimental Benchmark for Hybrid
Quantum-Classical Neural Networks Using Kernel Methods.
Master's Thesis, ITMO University, Saint Petersburg.
```