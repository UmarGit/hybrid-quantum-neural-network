import os
import time
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import multiprocessing

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Qiskit Imports
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import BackendEstimatorV2, StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as RuntimeEstimator

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

from models.quantum_models import (
    BaseQNN,
    ClassicalQuantumMLP,
    IterativeQNN,
    SplitAttentionQNN,
    ResQNet,
)
from models.datasets import load_dataset


# Training
def train_quantum(
    model: BaseQNN,
    epochs: int,
    X_train_tensor: torch.FloatTensor,
    y_train_tensor: torch.LongTensor,
):
    start_train = time.time()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    losses = []

    # model.train()
    print(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model.forward(X_train_tensor)
        loss = criterion.forward(output, y_train_tensor)
        loss.backward()
        optimizer.step()

        losses.append(loss)

        # if (epoch + 1) % 2 == 0:
        #     print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss.item():.4f}")

    end_train = time.time()
    training_time = end_train - start_train
    return model, training_time, losses


# Inference & Evaluation
def inference_quantum_simulator(
    model: BaseQNN,
    X_test_tensor: torch.FloatTensor,
    y_test_tensor: torch.LongTensor | None,
):
    accuracy = 0.0
    start_inference = time.time()
    model.eval()

    with torch.no_grad():
        output = model.forward(X_test_tensor)
        _, predictions = torch.max(output, 1)
        if y_test_tensor is not None:
            accuracy = (predictions == y_test_tensor).sum().item() / len(y_test_tensor)

    end_inference = time.time()
    inference_time = end_inference - start_inference

    return accuracy, inference_time, predictions


def inference_quantum_hardware(
    qc: QuantumCircuit,
    model: BaseQNN,
    feature_map,
    ansatz,
    observables,
    X_test_tensor: torch.FloatTensor,
    y_test_tensor: torch.LongTensor,
):
    try:
        print("\n--- Connecting to IBM Quantum Hardware ---")
        service = QiskitRuntimeService(
            channel="ibm_quantum_platform",
        )
        backend = service.least_busy(operational=True, simulator=False)
        print(f"Selected Backend: {backend.name}")

        # 1. Transpile
        print("Transpiling circuit...")
        coupling_map = backend.coupling_map
        basis_gates = backend.operation_names
        isa_circuit = transpile(
            qc, coupling_map=coupling_map, basis_gates=basis_gates, optimization_level=1
        )

        # 2. Map Observables
        layout = isa_circuit.layout
        mapped_observables = [
            obs.apply_layout(layout, num_qubits=isa_circuit.num_qubits)
            for obs in observables
        ]

        # 3. Hardware Estimator
        hardware_estimator = RuntimeEstimator(mode=backend)

        # 4. Re-create QNN for Hardware
        hw_qnn = EstimatorQNN(
            circuit=isa_circuit,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            observables=mapped_observables,
            estimator=hardware_estimator,
        )

        hw_model = BaseQNN(hw_qnn)
        hw_model.load_state_dict(model.state_dict())

        print("Submitting job to QPU... (This may take a few minutes)")
        start_inference = time.time()

        with torch.no_grad():
            output = hw_model(X_test_tensor)
            _, predictions = torch.max(output, 1)

        accuracy = (predictions == y_test_tensor).sum().item() / len(y_test_tensor)
        end_inference = time.time()

        return accuracy, end_inference - start_inference, predictions

    except Exception as e:
        print(f"Hardware Failed: {e}")
        return 0.0, 0.0, None


def plot_pca_decision_boundary_quantum_simulator(
    model: BaseQNN,
    X_scaled: torch.FloatTensor,
    y: torch.LongTensor,
    target_names: list[str],
):
    feat_x, feat_y = 2, 3

    # grid extents in the scaled space
    x_min, x_max = X_scaled[:, feat_x].min() - 0.2, X_scaled[:, feat_x].max() + 0.2
    y_min, y_max = X_scaled[:, feat_y].min() - 0.2, X_scaled[:, feat_y].max() + 0.2
    h = 0.02

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # build full 4D points by filling other dims with the mean (computed in scaled space)
    means = X_scaled.mean(axis=0)
    full_grid = np.tile(means, (grid_points.shape[0], 1))
    full_grid[:, feat_x] = grid_points[:, 0]
    full_grid[:, feat_y] = grid_points[:, 1]

    # model inference
    model.eval()
    with torch.no_grad():
        grid_tensor = torch.FloatTensor(full_grid)  # shape (M, 4)
        logits = model(grid_tensor)  # shape (M, num_classes)
        preds = logits.argmax(dim=1).cpu().numpy()

    Z = preds.reshape(xx.shape)

    plt.figure(figsize=(9, 6))
    plt.contourf(xx, yy, Z, alpha=0.3)
    scatter = plt.scatter(
        X_scaled[:, feat_x], X_scaled[:, feat_y], c=y, edgecolors="k", s=60
    )
    plt.xlabel(f"feature {feat_x}")
    plt.ylabel(f"feature {feat_y}")
    plt.title("Decision boundary in 2D feature slice (others fixed to mean)")
    plt.legend(handles=scatter.legend_elements()[0], labels=target_names)
    plt.savefig("quantum_simulator_decision_boundary.png")
    plt.show()


def z_observable(qubit_index: int, num_qubits: int) -> SparsePauliOp:
    """
    Create Z observable acting on a single qubit in an n-qubit system.

    Args:
        qubit_index: Target qubit (0 = least significant/rightmost)
        num_qubits: Total number of qubits

    Returns:
        SparsePauliOp with Z on target qubit, I elsewhere
    """
    if not 0 <= qubit_index < num_qubits:
        raise ValueError(f"qubit_index must be in [0, {num_qubits - 1}]")

    # Build Pauli string: I...I Z I...I (Z at position num_qubits-1-qubit_index)
    pauli_str = "I" * (num_qubits - qubit_index - 1) + "Z" + "I" * qubit_index
    return SparsePauliOp.from_list([(pauli_str, 1.0)])


def all_z_observables(num_qubits: int) -> list[SparsePauliOp]:
    """Create list of Z observables for all qubits [0, 1, ..., n-1]."""
    return [z_observable(i, num_qubits) for i in range(num_qubits)]


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seeds set to {seed}")


P_1QUBIT = 3.316e-4
P_2QUBIT = 6.20e-3
P_READOUT = 3.03e-2


def create_noisy_backend():
    noise_model = NoiseModel()

    # A. Gate Errors (Depolarizing)
    error_1 = depolarizing_error(P_1QUBIT, 1)
    noise_model.add_all_qubit_quantum_error(
        error_1, ["rx", "ry", "rz", "h", "u", "sx", "x"]
    )

    error_2 = depolarizing_error(P_2QUBIT, 2)
    noise_model.add_all_qubit_quantum_error(error_2, ["cx", "cz", "ecr"])

    # B. Readout Errors (Measurement Flip)
    # Prob of measuring 0 given 1, and 1 given 0
    readout_error = ReadoutError(
        [[1 - P_READOUT, P_READOUT], [P_READOUT, 1 - P_READOUT]]
    )
    noise_model.add_all_qubit_readout_error(readout_error)

    print("* HARDWARE REALITY: IBM Torino Simulation Active")
    print(f"   • Gate Noise: 1q={P_1QUBIT:.2%}, 2q={P_2QUBIT:.2%}")
    print(f"   • Readout Noise: {P_READOUT:.2%} (Dominant Factor)")

    num_cores = multiprocessing.cpu_count() - 1

    # C. Create the Simulator Backend with the Noise Model baked in
    # This prevents 'Options' attribute errors by defining it at instantiation
    backend = AerSimulator(
        noise_model=noise_model,
        method="density_matrix",  # Density matrix is ideal for accurate noise on small # of qubits
        max_parallel_threads=num_cores,
        max_parallel_experiments=num_cores,
        max_parallel_shots=1,
    )
    return backend


def get_quantum_circuit(num_qubits: int = 4):
    feature_map = ZFeatureMap(feature_dimension=num_qubits, reps=2)
    ansatz = RealAmplitudes(num_qubits=num_qubits, reps=2, entanglement="linear")

    qc = QuantumCircuit(num_qubits)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)

    observables = all_z_observables(num_qubits)

    # --- 2. SIMULATOR SETUP (NOISY) ---
    print("Setting up Noisy Simulator...")

    # --- 2. SIMULATOR SETUP (FIXED) ---
    print("Setting up BackendEstimatorV2...")
    # 1. Create the backend
    torino_backend = create_noisy_backend()

    # 2. Transpile the circuit to backend's basis gates
    # This decomposes high-level instructions like ZFeatureMap into basic gates
    print("Transpiling circuit to backend basis gates...")
    qc_transpiled = transpile(qc, backend=torino_backend, optimization_level=1)

    # 3. Wrap it in the Estimator V2 (This handles the primitive logic)
    noisy_estimator = BackendEstimatorV2(backend=torino_backend)
    # Set default shots via run options
    noisy_estimator.options.default_shots = 256

    # --- Simulator Run ---
    print("Setting up Simulator...")
    # estimator = StatevectorEstimator()
    qnn = EstimatorQNN(
        circuit=qc_transpiled,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        observables=observables,
        estimator=noisy_estimator,
    )

    return qnn


if __name__ == "__main__":
    # --- Data Prep ---
    seed_everything(54)

    # Device setup - Use MPS (Apple Silicon GPU) if available, otherwise CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    dataset_name = "leukemia"
    test_size = 0.9

    X, y, target_names = load_dataset(dataset_name)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_tensor = torch.FloatTensor(X_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)

    # --- Circuit Setup ---
    num_qubits = 4
    n_chunks = 10

    qnn = get_quantum_circuit(num_qubits=num_qubits)

    classicalqmlp = ClassicalQuantumMLP(
        qnn=qnn,
        input_dim=X_train_tensor.shape[1],
        output_dim=len(target_names),
        num_qubits=num_qubits,
    )
    iterativeqnn = IterativeQNN(
        qnn=qnn,
        input_dim=X_train_tensor.shape[1],
        num_qubits=num_qubits,
        hidden_dim=8,
        num_iterations=2,
    )
    splitattentionqnn = SplitAttentionQNN(
        qnn=qnn,
        input_dim=X_train_tensor.shape[1],
        num_qubits=num_qubits,
        n_chunks=n_chunks,
    )
    resqnet = ResQNet(
        qnn=qnn,
        input_dim=X_train_tensor.shape[1],
        output_dim=len(target_names),
        num_qubits=num_qubits,
        quantum_gate_limit=0.1,
    )

    models = [splitattentionqnn]
    titles = ["Split Attention"]

    epochs = 20

    for index, model in enumerate(models):
        # Move model to device
        model = model.to(device)

        trained_model, training_time, losses = train_quantum(
            model, epochs, X_train_tensor, y_train_tensor
        )

        # torch.save(trained_model.state_dict(), "quantum_model.pt")

        accuracy_qs, inference_time_qs, predictions_qs = inference_quantum_simulator(
            trained_model, X_test_tensor, y_test_tensor
        )

        total_params = sum(
            p.numel() for p in trained_model.parameters() if p.requires_grad
        )

        print("-" * 30)
        print(f"       Quantum Simulator MODEL RESULTS: {titles[index]}       ")
        print("-" * 30)
        print(f"Accuracy:           {accuracy_qs * 100:.2f}%")
        print(f"Training Time:      {training_time:.5f} sec")
        print(f"Inference Time:     {inference_time_qs:.5f} sec")
        print(f"Qubits:           {num_qubits}")
        print(f"N_Chunks:           {n_chunks}")
        print(f"Test Size:           {test_size}")
        print(f"Total number of parameters: {total_params}")
        print("-" * 30)
