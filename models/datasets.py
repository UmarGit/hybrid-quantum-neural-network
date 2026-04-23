import numpy as np
import pandas as pd

from sklearn.datasets import make_swiss_roll, load_breast_cancer, fetch_openml


def load_dataset(choice="swiss_roll", num_qubits=4):
    if choice == "swiss_roll":
        X, y_float = make_swiss_roll(n_samples=500, noise=0.1, random_state=42)
        y = np.array([1 if x > 9 else 0 for x in y_float])
        X = X[:, [0, 2]]
        target_names = ["Inner Roll", "Outer Roll"]
        print("Dataset: 2D Swiss Roll | Complexity: High Non-Linearity (Toy)")

    elif choice == "breast":
        breast_cancer_data = load_breast_cancer()
        X = breast_cancer_data.data
        y = breast_cancer_data.target
        target_names = breast_cancer_data.target_names.tolist()
        print(
            f"Dataset: Breast Cancer | Features: {X.shape[1]} | Complexity: Medium (Real-World)"
        )

    elif choice == "qsar":
        # 41 Features, ~1000 samples.
        print("Fetching QSAR dataset from OpenML... (may take a moment)")
        qsar = fetch_openml(name="qsar-biodeg", version=1, parser="auto")
        X = qsar.data.to_numpy().astype(np.float32)
        # Convert categories to 0 and 1
        y = qsar.target.astype("category").cat.codes.to_numpy()
        target_names = ["Not Ready Biodegradable", "Ready Biodegradable"]
        print(
            f"Dataset: QSAR Cancer Microarray | Features: {X.shape[1]} | Complexity: High (Cheminformatics)"
        )

    elif choice == "colon":
        # 2000 Features, only 62 Samples!
        print("Fetching Colon Cancer dataset from OpenML... (may take a moment)")
        colon = fetch_openml(name="colon-cancer", version=1, parser="auto")
        
        if hasattr(colon.data, 'toarray'):
            X = colon.data.toarray().astype(np.float32)
        else:
            X = np.asarray(colon.data, dtype=np.float32)
        
        y = pd.Series(colon.target).astype("category").cat.codes.to_numpy()
        
        target_names = ["Negative", "Positive"]
        print(
            f"Dataset: Colon Cancer Microarray | Features: {X.shape[1]} | Complexity: Extreme Starvation (Genomics)"
        )

    elif choice == "leukemia":
        # 7129 Features, only 72 Samples!
        print("Fetching Leukemia dataset from OpenML... (may take a moment)")
        leuk = fetch_openml(name="leukemia", version=1, parser="auto")
        X = leuk.data.to_numpy().astype(np.float32)
        y = leuk.target.astype("category").cat.codes.to_numpy()
        target_names = ["ALL", "AML"]
        print(
            f"Dataset: Leukemia Microarray | Features: {X.shape[1]} | Complexity: Extreme Starvation (Genomics)"
        )

    elif choice == "ntangled":
        import os
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import EfficientSU2
        from qiskit.quantum_info import Statevector

        print("Generating NTangled dataset from official generator weights...")

        # Point this to your cloned repository path
        base_path = "NTangled_Datasets/Hardware_Efficient/4_Qubits/Depth_5"

        # Load the low entanglement (CE=0.05) and high entanglement (CE=0.35) weights
        low_entanglement_file = os.path.join(base_path, "hwe_4q_ps_5_5_weights.npy")
        high_entanglement_file = os.path.join(base_path, "hwe_4q_ps_35_5_weights.npy")

        if not os.path.exists(low_entanglement_file):
            raise FileNotFoundError(
                f"Weights not found at {low_entanglement_file}. Check your relative path."
            )

        weights_low = np.load(low_entanglement_file)
        weights_high = np.load(high_entanglement_file)

        # We will generate 250 samples per class to simulate data starvation
        n_samples = 250

        def generate_states(weights, num_qubits, n_samples):
            # 1. Flatten the array to guarantee it's 1D (e.g., changes shape (1, 60) to (60,))
            flat_weights = weights.flatten()

            # 2. Use .size to get the absolute total number of elements
            # For HWE with U3 gates: 3 parameters per qubit per layer
            reps = (flat_weights.size // (3 * num_qubits)) - 1

            # Use 'u' gates to match the 3-parameter structure of the original dataset
            ansatz = EfficientSU2(
                num_qubits, su2_gates=["u"], entanglement="circular", reps=reps
            )

            # Bind the trained weights from the NTangled repository to the circuit
            bound_ansatz = ansatz.assign_parameters(
                flat_weights[: ansatz.num_parameters]
            )

            dataset = []
            for _ in range(n_samples):
                qc = QuantumCircuit(num_qubits)
                for q in range(num_qubits):
                    qc.u(
                        np.random.uniform(0, 2 * np.pi),
                        np.random.uniform(0, 2 * np.pi),
                        np.random.uniform(0, 2 * np.pi),
                        q,
                    )

                qc.compose(bound_ansatz, inplace=True)

                # Extract the exact statevector
                state = Statevector.from_instruction(qc).data

                # Convert complex state into real tensor
                real_state = np.hstack((np.real(state), np.imag(state)))
                dataset.append(real_state)

            return np.array(dataset, dtype=np.float32)

        print("Generating Class 0: Low Entanglement (CE=0.05)...")
        X_low = generate_states(weights_low, num_qubits, n_samples)

        print("Generating Class 1: High Entanglement (CE=0.35)...")
        X_high = generate_states(weights_high, num_qubits, n_samples)

        # Combine and assign labels
        X = np.vstack((X_low, X_high))
        y = np.hstack((np.zeros(n_samples), np.ones(n_samples))).astype(int)
        target_names = ["Low Entanglement", "High Entanglement"]

        print(
            f"Dataset: NTangled (Official) | Features: {X.shape[1]} (Real+Imag) | Complexity: Quantum Non-Local"
        )

    else:
        raise ValueError("Unknown Dataset Choice")

    return X, y, target_names
