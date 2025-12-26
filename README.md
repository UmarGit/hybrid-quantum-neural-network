# Hybrid Quantum Neural Network Benchmarking

**Research into Accelerating Neural Network training using Quantum Computing**

This repository contains the experimental code and benchmarking framework for a Master's research project investigating the geometric capabilities of Hybrid Quantum Neural Networks (HQNN). The study compares Classical MLPs against Qiskit-based QNNs across three tiers of geometric complexity: Linear, Non-Linear, and Topological.

## 🧪 Project Overview

The primary goal of this research is to isolate the **"Geometric Advantage"** of Quantum Machine Learning. By benchmarking on specific datasets, we demonstrate that while Quantum Hardware (NISQ) currently suffers from latency (`ibm_torino`), the Quantum Models exhibit **acceleration in convergence** for complex topological problems where classical linear models fail.

**Key Findings:**
* **Linear Data (Iris):** Classical models are superior in speed; Quantum models are expressive but overkill.
* **Non-Linear Data (Two Moons):** Quantum Entanglement (`ZZFeatureMap`) enables the model to "bend" decision boundaries, matching classical performance.
* **Topological Data (Concentric Circles):** **Quantum Advantage.** The QNN successfully learns the closed-loop topology (annulus) via high-dimensional mapping, whereas the shallow Classical MLP fails to converge (stalling at ~60% accuracy).

## Repository Structure

The project is structured into modular python scripts for model definitions and Jupyter notebooks for specific experimental drivers.

```text
.
├── README.md                          # Project documentation
├── classical.py                       # PyTorch definitions for Classical MLP baselines
├── quantum.py                         # Qiskit definitions for Quantum Circuits (Ansatz, FeatureMaps) & Hybrid QNN classes
├── iris-full-experiment-arch.ipynb    # Experiment 1: Linear Baseline & Hardware Deployment
├── moons-full-experiment-arch.ipynb   # Experiment 2: Non-Linear Entanglement Test
└── circle-full-experiment-arch.ipynb  # Experiment 3: Topological Stress Test (The "Smoking Gun" Result)

```

## 🛠 Installation & Requirements

This project utilizes **PyTorch** for optimization and **Qiskit** for quantum circuit simulation and hardware execution.

```bash
pip install qiskit qiskit-machine-learning qiskit-ibm-runtime
pip install torch torchvision
pip install scikit-learn matplotlib numpy

```

## 🚀 Usage

### 1. Model Definitions (`quantum.py` & `classical.py`)

These files contain the core classes imported into the notebooks.

* **`quantum.py`**: Defines the `QuantumMLP` class using `qiskit_machine_learning.connectors.TorchConnector`. It handles the `ZZFeatureMap` (Entanglement) and `EfficientSU2` (Ansatz).
* **`classical.py`**: Defines the standard `nn.Module` architectures used for baseline comparisons.

### 2. Running Experiments

To reproduce the topological findings, run the notebooks in the following order of complexity:

1. **`iris-full-experiment-arch.ipynb`**:
* Runs the baseline calibration.
* Contains the code for deploying to **IBM Quantum Hardware** (`ibm_torino`).
* *Note: Requires valid IBM Quantum API Token.*


2. **`moons-full-experiment-arch.ipynb`**:
* Demonstrates the "Interference Island" effect where quantum boundaries curve around non-linear data.


3. **`circle-full-experiment-arch.ipynb`**:
* **Main Result:** Demonstrates the QNN forming a closed loop (bullseye) decision boundary, solving the topology that the classical model slices in half.



## 📊 Visual Results

The experiments generate 2D and 3D Decision Boundary landscapes.

* **Classical Behavior:** Tends to create linear polygonal cuts (Hyperplanes).
* **Quantum Behavior:** Creates periodic, wave-like interference patterns (Partial Fourier Series), allowing for the natural formation of closed loops and islands.

## ⚠️ Hardware vs. Simulator

The code is configured to run on `Qiskit Aer` (Simulator) by default for speed.
To run on real hardware (e.g., `ibm_torino` or `ibm_brisbane`), update the `service` definition:

```python
service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    token="YOUR_TOKEN",
    instance="YOUR_INSTANCE"
)
```

## 👨‍🔬 Author

**Ahmed Umar**

* **University:** ITMO University
* **Group:** J4232
* **Supervisor:** Khodnenko Ivan Vladimirovich, PhD
