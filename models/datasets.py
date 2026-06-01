import numpy as np
import pandas as pd

from sklearn.datasets import (
    load_wine,
    make_circles,
    make_moons,
    make_swiss_roll,
    load_breast_cancer,
    fetch_openml,
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder


def load_dataset(choice="swiss_roll", num_qubits=4):
    if choice == "swiss_roll":
        X, y_float = make_swiss_roll(n_samples=500, noise=0.1, random_state=42)
        y = np.array([1 if x > 9 else 0 for x in y_float])
        X = X[:, [0, 2]]
        target_names = ["Inner Roll", "Outer Roll"]
        print("Dataset: 2D Swiss Roll | Complexity: High Non-Linearity (Toy)")

    elif choice == "circles":
        # make_circles returns 2D coordinates and binary labels
        X, y = make_circles(n_samples=500, noise=0.1, factor=0.5, random_state=42)
        target_names = ["Inner Circle", "Outer Circle"]
        print("Dataset: 2D Circles | Complexity: Non-Linear")

    elif choice == "moons":
        # make_moons returns 2D coordinates and binary labels
        X, y = make_moons(n_samples=500, noise=0.1, random_state=42)
        target_names = ["Bottom Moon", "Top Moon"]
        print("Dataset: 2D Moons | Complexity: Non-Linear")

    elif choice == "wine":
        # make_circles returns 2D coordinates and binary labels
        wine_data = load_wine()
        X = wine_data.data
        y = wine_data.target
        target_names = wine_data.target_names.tolist()
        print("Dataset: Wine Quality | Complexity: Non-Linear")

    elif choice == "indian_pines":
        indian_pines_X_path = "datasets/indian-pines/indianpinearray.npy"
        indian_pines_y_path = "datasets/indian-pines/IPgt.npy"

        # Load the 3D hyperspectral cube and 2D ground truth labels from local files
        try:
            X_cube = np.load(indian_pines_X_path)
            y_2d = np.load(indian_pines_y_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find {indian_pines_X_path} or {indian_pines_y_path}. Please download them from Kaggle and place them in the working directory."
            )

        # Indian Pines X is typically (145, 145, 200) and y is (145, 145)
        # We need to flatten the spatial dimensions to get (n_samples, n_features)
        n_rows, n_cols, n_bands = X_cube.shape
        X = X_cube.reshape((n_rows * n_cols, n_bands))
        y = y_2d.reshape((n_rows * n_cols,))

        # Optional but highly recommended: Remove background pixels (Label 0)
        # Indian pines has 16 valid classes (1-16), and 0 represents unclassified background
        valid_pixels = y != 0
        X = X[valid_pixels]
        y = y[valid_pixels] - 1  # CrossEntropyLoss expects 0-indexed targets

        # Reduce the hyperspectral bands to a compact 30-dimensional PCA space.
        X = PCA(n_components=30).fit_transform(X)

        # Generate target names for the 16 distinct classes
        target_names = [
            "Alfalfa",
            "Corn-notill",
            "Corn-mintill",
            "Corn",
            "Grass-pasture",
            "Grass-trees",
            "Grass-pasture-mowed",
            "Hay-windrowed",
            "Oats",
            "Soybean-notill",
            "Soybean-mintill",
            "Soybean-clean",
            "Wheat",
            "Woods",
            "Buildings-Grass-Trees-Drives",
            "Stone-Steel-Towers",
        ]

        print(f"Dataset: Indian Pines | Complexity: High Dimensional ({n_bands} bands)")

    elif choice == "indian_pines_small":
        indian_pines_X_path = "datasets/indian-pines/indianpinearray.npy"
        indian_pines_y_path = "datasets/indian-pines/IPgt.npy"

        try:
            X_cube = np.load(indian_pines_X_path)
            y_2d = np.load(indian_pines_y_path)
        except FileNotFoundError:
            raise FileNotFoundError("Could not find datasets. Please check the path.")

        n_rows, n_cols, n_bands = X_cube.shape
        X = X_cube.reshape((n_rows * n_cols, n_bands))
        y = y_2d.reshape((n_rows * n_cols,))

        # Remove background
        valid_pixels = y != 0
        X = X[valid_pixels]
        y = y[valid_pixels] - 1  # 0-indexed

        # STEP 1: Make it Binary. Indian Pines has 16 classes. 
        binary_mask = (y == 9) | (y == 10)
        X = X[binary_mask]
        y = y[binary_mask]
        
        # Remap to 0 and 1 for the PyTorch CrossEntropy/BCE Loss
        y = np.where(y == 9, 0, 1)

        target_names = ["Soybean-notill", "Soybean-mintill"]
        print(f"Dataset: Indian Pines (Binary Subset) | Samples: {len(X)} | Features: 30 (PCA)")

    elif choice == "fingertips":
        dataset = "datasets/fingertips/fingertips-dataset.csv"

        df = pd.read_csv(dataset, sep=";")

        df['block'] = (df['state'] != df['state'].shift(1)).cumsum()
        blocks = df[df['state'] != 'unlabeled'].groupby(['block', 'state']).size().reset_index(name='count')
        
        active_cols = [c for c in df.columns if c not in ['time', 'state', 'block'] and '1' not in c]
        
        X_list, y_list = [], []
        for _, row in blocks.iterrows():
            block_data = df[df['block'] == row['block']][active_cols].values
            
            features = np.concatenate([
                np.mean(block_data, axis=0),
                np.std(block_data, axis=0),
                np.min(block_data, axis=0),
                np.max(block_data, axis=0)
            ])
            X_list.append(features)
            y_list.append(row['state'])
            
        X = np.array(X_list)
        y_raw = np.array(y_list)
        y = LabelEncoder().fit_transform(np.array(y_list))

        encoder = LabelEncoder()
        y = encoder.fit_transform(y_raw)

        target_names = encoder.classes_

        print(f"Dataset: Fingertips Accelerometer Data | Samples: {len(X)}")

    elif choice == "brain":
        brain_counts_path = "datasets/brain/brain_counts.csv"
        brain_metadata_path = "datasets/brain/brain_metadata.csv"

        try:
            counts_df = pd.read_csv(brain_counts_path, index_col=0)
            metadata_df = pd.read_csv(brain_metadata_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find {brain_counts_path} or {brain_metadata_path}. Please place the Tabula Muris brain CSVs in the working directory."
            )

        if "cell" not in metadata_df.columns or "cell_ontology_class" not in metadata_df.columns:
            raise ValueError(
                "brain_metadata.csv must contain 'cell' and 'cell_ontology_class' columns."
            )

        metadata_df = metadata_df.set_index("cell")

        # Join expression rows with cell-level annotations and keep only matched cells.
        joined = counts_df.join(metadata_df[["cell_ontology_class"]], how="inner")
        if joined.empty:
            raise ValueError("No overlapping cells were found between brain_counts.csv and brain_metadata.csv.")

        label_names = sorted(joined["cell_ontology_class"].dropna().unique().tolist())
        labels = pd.Categorical(joined.pop("cell_ontology_class"), categories=label_names, ordered=True)
        y = labels.codes.astype(int)
        target_names = label_names

        X = joined.to_numpy(dtype=np.float32)

        # # Reduce the gene expression space to a compact PCA representation for downstream models.
        # n_components = min(30, X.shape[0] - 1, X.shape[1])
        # if n_components < 2:
        #     raise ValueError("brain dataset does not have enough samples for PCA reduction.")
        # X = PCA(n_components=n_components).fit_transform(X)

        print(
            f"Dataset: Brain RNA-seq | Cells: {X.shape[0]} | PCA Features: {X.shape[1]} | Classes: {len(target_names)}"
        )


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

        if hasattr(colon.data, "toarray"):
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
