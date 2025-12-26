import time
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# Qiskit Imports
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as RuntimeEstimator


# Classical MLP Model
class QuantumMLP(nn.Module):
    def __init__(self, qnn: EstimatorQNN, input_dim = 2, output_dim = 3):
        super(QuantumMLP, self).__init__()
        self.qnn = TorchConnector(qnn)
        self.network = nn.Sequential(nn.Linear(input_dim, 8), nn.ReLU(), nn.Linear(8, output_dim))

    def forward(self, x):
        x = self.qnn(x)
        x = self.network(x)
        return x


# Training
def train_quantum(
    model: QuantumMLP,
    epochs: int,
    X_train_tensor: torch.FloatTensor,
    y_train_tensor: torch.LongTensor,
):
    start_train = time.time()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    model.train()
    print(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss.item():.4f}")

    end_train = time.time()
    training_time = end_train - start_train
    return model, training_time


# Inference & Evaluation
def inference_quantum_simulator(
    model: QuantumMLP,
    X_test_tensor: torch.FloatTensor,
    y_test_tensor: torch.LongTensor | None,
):
    accuracy = 0.0
    start_inference = time.time()
    model.eval()

    with torch.no_grad():
        output = model(X_test_tensor)
        _, predictions = torch.max(output, 1)
        if y_test_tensor is not None:
            accuracy = (predictions == y_test_tensor).sum().item() / len(y_test_tensor)

    end_inference = time.time()
    inference_time = end_inference - start_inference

    return accuracy, inference_time, predictions


def inference_quantum_hardware(
    qc: QuantumCircuit,
    model: QuantumMLP,
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

        hw_model = QuantumMLP(hw_qnn)
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
    model: QuantumMLP,
    X_scaled: torch.FloatTensor,
    y: torch.LongTensor,
    target_names: list[str],
):
    x_min, x_max = X_scaled[:, 0].min() - 0.2, X_scaled[:, 0].max() + 0.2
    y_min, y_max = X_scaled[:, 1].min() - 0.2, X_scaled[:, 1].max() + 0.2

    h = 0.05

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h, dtype="float"),
        np.arange(y_min, y_max, h, dtype="float"),
    )
    grid_tensor = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])

    _, _, predictions = inference_quantum_simulator(model, grid_tensor, None)

    Z = predictions.reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.viridis)

    scatter = plt.scatter(
        X_scaled[:, 0], X_scaled[:, 1], c=y, edgecolors="k", s=60, cmap=plt.cm.viridis
    )

    plt.title("Hybrid QNN Decision Boundary")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(handles=scatter.legend_elements()[0], labels=target_names)
    plt.savefig("quantum_simulator_decision_boundary.png")
    print("Plot saved.")

def plot_3d_decision_boundary_quantum(model, X_scaled, y, target_names):
    x_min, x_max = X_scaled[:, 0].min() - 0.2, X_scaled[:, 0].max() + 0.2
    y_min, y_max = X_scaled[:, 1].min() - 0.2, X_scaled[:, 1].max() + 0.2
    h = 0.1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h, dtype="float"),
        np.arange(y_min, y_max, h, dtype="float"),
    )
    grid_tensor = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])

    _, _, predictions = inference_quantum_simulator(model, grid_tensor, None)
    
    Z = predictions.reshape(xx.shape)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    _ = ax.plot_surface(xx, yy, Z, cmap=plt.cm.viridis, alpha=0.4, antialiased=False)

    scatter = ax.scatter(
        X_scaled[:, 0], 
        X_scaled[:, 1], 
        y,           
        c=y, 
        cmap=plt.cm.viridis, 
        edgecolors='k', 
        s=40,
        depthshade=False
    )

    ax.set_title("3D Quantum Decision Landscape")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Classes")
    
    legend1 = ax.legend(*scatter.legend_elements(), title="Data", loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.add_artist(legend1)

    ax.view_init(elev=30, azim=-60)
    
    plt.savefig("quantum_3d_decision_boundary.png")
    print("3D Plot saved as 'quantum_3d_decision_boundary.png'")
    plt.show()

if __name__ == "__main__":
    # --- Data Prep ---
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pca = PCA(n_components=2)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_train_tensor = torch.LongTensor(y_train)
    y_test_tensor = torch.LongTensor(y_test)

    # --- Circuit Setup ---
    num_qubits = 2
    feature_map = ZFeatureMap(feature_dimension=num_qubits, reps=2)
    ansatz = RealAmplitudes(num_qubits=num_qubits, reps=2, entanglement="linear")

    qc = QuantumCircuit(num_qubits)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)

    observables = [
        SparsePauliOp.from_list([("IZ", 1)]),
        SparsePauliOp.from_list([("ZI", 1)]),
    ]

    # --- Simulator Run ---
    print("Setting up Simulator...")
    estimator = StatevectorEstimator()
    qnn = EstimatorQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        observables=observables,
        estimator=estimator,
    )

    quantum_model = QuantumMLP(qnn)

    # Train (Increased epochs to 40 for better convergence)
    trained_model, training_time = train_quantum(
        quantum_model, 20, X_train_tensor, y_train_tensor
    )

    accuracy_qs, inference_time_qs, predictions_qs = inference_quantum_simulator(
        trained_model, X_test_tensor, y_test_tensor
    )

    print("-" * 30)
    print("       Quantum Simulator MODEL RESULTS       ")
    print("-" * 30)
    print(f"Accuracy:           {accuracy_qs * 100:.2f}%")
    print(f"Training Time:      {training_time:.5f} sec")
    print(f"Inference Time:     {inference_time_qs:.5f} sec")
    print("-" * 30)

    plot_pca_decision_boundary_quantum_simulator(
        trained_model, X_test_tensor, y_test_tensor, target_names
    )

