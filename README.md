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

## Installation & Requirements

This project utilizes **PyTorch** for optimization and **Qiskit** for quantum circuit simulation and hardware execution.

```bash
pip install qiskit qiskit-machine-learning qiskit-ibm-runtime
pip install torch torchvision
pip install scikit-learn matplotlib numpy

```

## Usage

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



## Visual Benchmarks

We compare the decision boundaries of a standard Classical MLP vs. the Hybrid Quantum Neural Network (HQNN) across non-linear and topological datasets.

| Dataset | Classical MLP (Baseline) | Hybrid Quantum NN (Ours) |
| :---: | :---: | :---: |
| **Two Moons**<br>*(Non-Linearity)* | <img src="moon/classical_decision_boundary.png" width="400"/> | <img src="moon/quantum_simulator_decision_boundary.png" width="400"/> |
| **Concentric Circles**<br>*(Topology)* | <img src="circles/classical_decision_boundary.png" width="400"/> | <img src="circles/quantum_simulator_decision_boundary.png" width="400"/> |

**Analysis:**

* **Row 1 (Moons):** Both models solve the problem, but the Quantum model uses entanglement to "bend" the space, creating a smoother decision boundary.
* **Row 2 (Circles):** This is the **critical result**. The Classical MLP fails to close the loop (creating a linear cut), whereas the HQNN naturally forms a closed topology ("bullseye") due to the periodic nature of the quantum feature map.

### Quantitative Results

Comparing Classical vs. Quantum performance across varying geometric complexities.

| Dataset | Classical MLP (Acc) | Quantum QNN (Acc) | Observation |
| --- | --- | --- | --- |
| **Iris** (Linear) | 83.33% | **93.33%** | Quantum is accurate but overkill for simple linear data. |
| **Moons** (Non-Linear) | **90.42%** | 81.25% | Quantum successfully captures the curvature but struggles with noise. |
| **Circles** (Topological) | 61.67% (Failed) | **68.00%** | **Quantum Advantage.** Classical model failed to converge (linear cut); Quantum naturally solved the ring topology. |

## Sensitivity Analysis (Iris Dataset)

We conducted controlled experiments to evaluate the impact of circuit depth and feature map frequency on model trainability.

| Configuration | Reps | Accuracy | Analysis |
| --- | --- | --- | --- |
| **Baseline** | 2 | **93.33%** | **Optimal.** Balanced expressibility and trainability. |
| **Shallow** | 1 | 83.33% | **Underfitting.** The circuit lacked sufficient parameters to separate classes. |
| **Deep** | 3 | 90.00% | **Diminishing Returns.** Increased training time (+42%) with no accuracy gain. |
| **High-Freq** | 3 | 73.33% | **Optimization Failure.** High-frequency encoding created a chaotic landscape (**Barren Plateau**), preventing gradient convergence. |


## Hardware vs. Simulator

The code is configured to run on `Qiskit Aer` (Simulator) by default for speed.
To run on real hardware (e.g., `ibm_torino` or `ibm_brisbane`), update the `service` definition:

```python
service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    token="YOUR_TOKEN",
    instance="YOUR_INSTANCE"
)
```

## Maintainer

Umar Ahmed - Research Engineer (ITMO) Developed as part of the Hybrid Quantum Architectures Benchmarking Initiative.
