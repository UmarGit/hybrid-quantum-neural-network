import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Import your existing architectures
from fragments.quantum import get_quantum_circuit
from models.classical_models import CKAResCNet
from models.quantum_models import QKAResQNet


def load_and_preprocess(file_path="esp32_data_20260520_1058.csv"):
    df = pd.read_csv(file_path, sep=";")

    # 1. Isolate continuous gesture blocks
    df["block"] = (df["state"] != df["state"].shift(1)).cumsum()
    blocks = (
        df[df["state"] != "unlabeled"]
        .groupby(["block", "state"])
        .size()
        .reset_index(name="count")
    )

    # 2. Drop the dead sensor (#1) and system columns
    active_cols = [
        c for c in df.columns if c not in ["time", "state", "block"] and "1" not in c
    ]

    X_list, y_list = [], []
    for _, row in blocks.iterrows():
        block_data = df[df["block"] == row["block"]][active_cols].values

        # 3. Extract temporal geometric features (Flatten the trajectory)
        features = np.concatenate(
            [
                np.mean(block_data, axis=0),
                np.std(block_data, axis=0),
                np.min(block_data, axis=0),
                np.max(block_data, axis=0),
            ]
        )
        X_list.append(features)
        y_list.append(row["state"])

    X = np.array(X_list)
    y = LabelEncoder().fit_transform(np.array(y_list))

    return X, y, len(np.unique(y))


def train_pytorch(model, X_train, y_train, epochs=150, lr=0.01, verbose=True):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_t = torch.FloatTensor(X_train)
    y_t = torch.LongTensor(y_train)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(X_t)
        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % 25 == 0:
            print(f"    Epoch {epoch + 1:3d}/{epochs} | Loss: {loss.item():.4f}")
    return model


def main():
    print("=== ESP32 Accelerometer Gesture Recognition ===")
    print("Loading and Preprocessing Data...")
    X, y, num_classes = load_and_preprocess(
        "datasets/gesture/gesture-dataset.csv"
    )
    print(
        f"Data Ready: {X.shape[0]} samples, {X.shape[1]} features, {num_classes} classes.\n"
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    svm_scores, cka_scores, qka_scores = [], [], []

    # Hyperparameters
    epochs = 100
    lr = 0.005
    num_qubits = 4
    hidden_dim = num_qubits

    print(
        f"Start 5-Fold Cross Validation (num_qubits={num_qubits}, epochs={epochs}, lr={lr})"
    )

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        print(f"\n--- Fold {fold + 1}/5 ---")

        # Scale Data
        scaler = MinMaxScaler(feature_range=(0, np.pi))
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        y_train, y_test = y[train_idx], y[test_idx]

        # 1. Classical RBF-SVM
        print("  Training Classical SVM...")
        svm = SVC(kernel="rbf", C=10, gamma="scale")
        svm.fit(X_train, y_train)
        svm_acc = accuracy_score(y_test, svm.predict(X_test))
        svm_scores.append(svm_acc)
        print(f"  [Result] Classical SVM: {svm_acc * 100:.1f}%")

        X_test_t = torch.FloatTensor(X_test)

        # 3. CKAResCNet
        print("  Training CKAResCNet...")
        cka = CKAResCNet(
            input_dim=X.shape[1], output_dim=num_classes, hidden_dim=hidden_dim
        )
        cka = train_pytorch(cka, X_train, y_train, epochs=epochs, lr=lr, verbose=True)

        with torch.no_grad():
            cka_preds = torch.argmax(cka(X_test_t), dim=1).numpy()
        cka_acc = accuracy_score(y_test, cka_preds)
        cka_scores.append(cka_acc)
        print(f"  [Result] CKAResCNet   : {cka_acc * 100:.1f}%")

        # 5. QKAResQNet
        print("  Training QKAResQNet...")
        qcircuit2 = get_quantum_circuit(num_qubits)
        qka = QKAResQNet(
            qnn=qcircuit2,
            input_dim=X.shape[1],
            output_dim=num_classes,
            num_qubits=num_qubits,
        )
        qka = train_pytorch(qka, X_train, y_train, epochs=epochs, lr=lr, verbose=True)

        with torch.no_grad():
            qka_preds = torch.argmax(qka(X_test_t), dim=1).numpy()
        qka_acc = accuracy_score(y_test, qka_preds)
        qka_scores.append(qka_acc)
        print(f"  [Result] QKAResQNet   : {qka_acc * 100:.1f}%")

    print("\n==================================")
    print("=== FINAL RESULTS (5-Fold CV) ===")
    print("==================================")
    print(
        f"SVM         : {np.mean(svm_scores) * 100:5.2f}% ± {np.std(svm_scores) * 100:.2f}%"
    )
    print(
        f"CKAResCNet  : {np.mean(cka_scores) * 100:5.2f}% ± {np.std(cka_scores) * 100:.2f}%"
    )
    print(
        f"QKAResQNet  : {np.mean(qka_scores) * 100:5.2f}% ± {np.std(qka_scores) * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
