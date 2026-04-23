# Hybrid Quantum Neural Network Benchmarking

Research code for a systematic comparison of classical and hybrid quantum neural architectures across diverse data regimes.

## Project Status

Active research project. The repository focuses on reproducible experimental workflows and analysis artifacts rather than production deployment.

## Why This Repository Exists

The current work is a systematic comparison of parameter-matched classical and hybrid quantum neural architectures across diverse data regimes. The core question is not "is quantum always better", but:

- when hybrid quantum blocks help,
- when they are neutral,
- and when classical backbones dominate.

The benchmark is organized around a strict 1-to-1 replacement protocol where both models share the same outer architecture and training setup, and only the core processing block changes.

The repository includes architecture definitions, controlled experiment drivers, and aggregated multi-seed results for both baseline and extreme-starvation settings.

## Thesis And Preprint

- Thesis: [thesis.pdf](thesis.pdf)
- Preprint (source): [preprint.tex](preprint.tex)
- Preprint (compiled): [preprint.pdf](preprint.pdf)

How these map to the code:

- Experimental implementation: [classical.py](classical.py), [quantum.py](quantum.py), [models](models)
- Aggregation and reporting pipeline: [collect_tables.py](collect_tables.py), [aggregate_seed_results.py](aggregate_seed_results.py), [plots.py](plots.py)
- Generated artifacts: [results](results), [results-lk-col](results-lk-col), [plots](plots)

## Repository Layout

- [models](models): Classical and quantum model implementations
- [classical.py](classical.py): Classical training and evaluation script
- [quantum.py](quantum.py): Quantum training and evaluation script
- [collect_tables.py](collect_tables.py): Multi-seed benchmarking runner
- [aggregate_seed_results.py](aggregate_seed_results.py): Seed summary generation
- [plots.py](plots.py): Figure generation from benchmark CSV files
- [iris-full-experiment-arch.ipynb](iris-full-experiment-arch.ipynb), [moons-full-experiment-arch.ipynb](moons-full-experiment-arch.ipynb), [circle-full-experiment-arch.ipynb](circle-full-experiment-arch.ipynb): Notebook experiments

## Quickstart

1. Clone repository.
2. Create environment and install dependencies.
3. Run experiments and generate summaries.

Example commands:

```bash
git clone <repo-url>
cd quantum-neural-network

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python collect_tables.py
python aggregate_seed_results.py
python plots.py
```

If you use conda, an existing environment can be used in the same way after activation.

Conda example:

```bash
conda activate qnn
pip install -r requirements.txt
python collect_tables.py
python aggregate_seed_results.py
python plots.py
```

## Tested Environment

- Python: 3.11
- OS: macOS
- Key frameworks: PyTorch, Qiskit, qiskit-machine-learning

## Reproducing Experiments

### Notebook Workflow(Old Code)

Run notebooks in this order:

1. [iris-full-experiment-arch.ipynb](iris-full-experiment-arch.ipynb)
2. [moons-full-experiment-arch.ipynb](moons-full-experiment-arch.ipynb)
3. [circle-full-experiment-arch.ipynb](circle-full-experiment-arch.ipynb)

### Script Workflow(New Code)

- Benchmark tables: [collect_tables.py](collect_tables.py)
- Aggregate seed statistics: [aggregate_seed_results.py](aggregate_seed_results.py)
- Generate plots: [plots.py](plots.py)

The script pipeline supports two key regimes described in the new preprint:

- baseline multi-seed comparisons (general topologies)
- 20-seed extreme-starvation sweeps on Leukemia and Colon

Expected outputs:

- Per-seed CSV files in [results](results) or [results-lk-col](results-lk-col)
- Aggregated summary CSV files with mean and standard deviation per model
- Figure files in [plots](plots)

## Data Sources

- Iris, Moons, Circles: generated or sourced through scikit-learn workflows.
- Colon and Leukemia benchmarks: loaded through project dataset utilities and recorded in the result tables.
- Additional materials are present under [NTangled_Datasets](NTangled_Datasets).

## Hardware Vs Simulator

Default execution is simulator-oriented for reproducibility and speed.

Hardware execution requires an IBM Quantum account and valid runtime credentials. If credentials are unavailable, run in simulator mode.

## Key Results Snapshot

The latest results align with [preprint.tex](preprint.tex):

- No universal quantum advantage across all datasets.
- Hybrid ResQNet is competitive with parameter-matched classical ResNet.
- Under extreme starvation, both models converge to a similar accuracy ceiling.
- A modest variance reduction appears in Leukemia for the hybrid model.

Seed-level benchmark summaries for Leukemia and Colon:

- [results-lk-col/classical_results_seed_summary.csv](results-lk-col/classical_results_seed_summary.csv)
- [results-lk-col/quantum_results_seed_summary.csv](results-lk-col/quantum_results_seed_summary.csv)

Figures used in the preprint discussion:

- [plots/lk_col_20_seed_boxplot.png](plots/lk_col_20_seed_boxplot.png)
- [plots/lk_col_parameter_vs_accuracy.png](plots/lk_col_parameter_vs_accuracy.png)

## Troubleshooting

- Import errors:
    - Reinstall dependencies with requirements file in a clean environment.
- Qiskit backend/authentication failures:
    - Use simulator mode first, then configure IBM Quantum credentials.
- Missing generated artifacts:
    - Ensure benchmark collection runs before aggregation and plotting.
- Slow runtime:
    - Use smaller epoch counts for smoke tests before full runs.

## Contributing

Issues and pull requests are welcome for reproducibility fixes, cleaner experiment abstractions, and additional benchmark scenarios.

## License

This project is licensed under MIT. See [LICENSE](LICENSE).

## Citation

If you use this repository in research, please cite the thesis and/or preprint:

- [thesis.pdf](thesis.pdf)
- [preprint.tex](preprint.tex)
- [preprint.pdf](preprint.pdf)
 