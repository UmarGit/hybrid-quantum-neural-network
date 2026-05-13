import csv
import time
import random
import io
import contextlib
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from models.datasets import load_dataset
from models.classical_models import (
    ClassicalMLP,
    IterativeClassicalNN,
    SplitAttentionClassicalNN,
    ResNet,
    CKAResCNet,
)
from models.quantum_models import (
    ClassicalQuantumMLP,
    IterativeQNN,
    SplitAttentionQNN,
    ResQNet,
    QKAResQNet,
)
from quantum import get_quantum_circuit


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(
    model: nn.Module,
    X_train: torch.FloatTensor,
    y_train: torch.LongTensor,
    epochs: int,
    lr: float,
    desc: str,
) -> tuple[nn.Module, float]:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start = time.time()
    model.train()
    for _ in tqdm(range(epochs), desc=desc, leave=False):
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
    training_time = time.time() - start

    return model, training_time


def evaluate_model(
    model: nn.Module,
    X_test: torch.FloatTensor,
    y_test: torch.LongTensor,
) -> tuple[float, float]:
    start = time.time()
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        preds = logits.argmax(dim=1)
        accuracy = (preds == y_test).float().mean().item()
    inference_time = time.time() - start
    return accuracy, inference_time


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def append_row(csv_path: Path, row: dict[str, object], headers: list[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def build_classical_model(
    model_name: str,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    n_chunks: int,
) -> nn.Module:
    if model_name == "ClassicalMLP":
        return ClassicalMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
        )
    if model_name == "IterativeClassicalNN":
        return IterativeClassicalNN(
            input_dim=input_dim,
            hidden_dim=8,
            intermediate_dim=hidden_dim,
            num_iterations=2,
        )
    if model_name == "SplitAttentionClassicalNN":
        return SplitAttentionClassicalNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_chunks=n_chunks,
        )
    if model_name == "ResNet":
        return ResNet(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            residual_gate_init=0.1,
        )
    if model_name == "CKAResCNet":
        return CKAResCNet(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            residual_gate_init=0.1,
        )
    raise ValueError(f"Unknown classical model: {model_name}")


def build_quantum_model(
    model_name: str,
    input_dim: int,
    output_dim: int,
    num_qubits: int,
    n_chunks: int,
    quiet_logs: bool,
) -> nn.Module:
    if quiet_logs:
        with contextlib.redirect_stdout(io.StringIO()):
            qnn = get_quantum_circuit(num_qubits=num_qubits)
    else:
        qnn = get_quantum_circuit(num_qubits=num_qubits)

    if model_name == "ClassicalQuantumMLP":
        return ClassicalQuantumMLP(
            qnn=qnn,
            input_dim=input_dim,
            output_dim=output_dim,
            num_qubits=num_qubits,
        )
    if model_name == "IterativeQNN":
        return IterativeQNN(
            qnn=qnn,
            input_dim=input_dim,
            num_qubits=num_qubits,
            hidden_dim=8,
            num_iterations=2,
        )
    if model_name == "SplitAttentionQNN":
        return SplitAttentionQNN(
            qnn=qnn,
            input_dim=input_dim,
            num_qubits=num_qubits,
            n_chunks=n_chunks,
        )
    if model_name == "ResQNet":
        return ResQNet(
            qnn=qnn,
            input_dim=input_dim,
            output_dim=output_dim,
            num_qubits=num_qubits,
            quantum_gate_limit=0.1,
        )
    if model_name == "QKAResQNet":
        return QKAResQNet(
            qnn=qnn,
            input_dim=input_dim,
            output_dim=output_dim,
            num_qubits=num_qubits,
            quantum_gate_limit=0.1,
        )
    raise ValueError(f"Unknown quantum model: {model_name}")


def prepare_data(dataset_name: str, test_size: float, seed: int, quiet_logs: bool):
    if quiet_logs:
        with contextlib.redirect_stdout(io.StringIO()):
            X, y, target_names = load_dataset(dataset_name)
    else:
        X, y, target_names = load_dataset(dataset_name)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
    )

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_train_tensor = torch.LongTensor(y_train)
    y_test_tensor = torch.LongTensor(y_test)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, target_names


def main() -> None:
    # Simple configuration: edit these values directly.
    datasets = [
        # "breast",
        # "swiss_roll",
        # "qsar",
        #  "colon",
        "leukemia",
        # "ntangled"
    ]
    run_classical = True
    run_quantum = True
    quiet_logs = True
    epochs = 20
    test_size = 0.9
    learning_rate = 0.1
    hidden_dim = 4
    n_chunks = 10
    num_qubits = 4
    out_dir = Path("results")
    seeds = range(0, 21)

    classical_models = [
        # "ClassicalMLP",
        # "IterativeClassicalNN",
        "SplitAttentionClassicalNN",
        # "ResNet",
        # "CKAResCNet",
    ]
    quantum_models = [
        # "ClassicalQuantumMLP",
        # "IterativeQNN",
        "SplitAttentionQNN",
        # "ResQNet",
        # "QKAResQNet",
    ]

    headers = [
        "timestamp",
        "dataset",
        "model",
        "accuracy",
        "training_time_sec",
        "inference_time_sec",
        "epochs",
        "test_size",
        "learning_rate",
        "hidden_dim",
        "n_chunks",
        "num_qubits",
        "input_dim",
        "num_classes",
        "train_size",
        "test_count",
        "total_params",
        "seed",
    ]

    enabled_model_count = 0
    if run_classical:
        enabled_model_count += len(classical_models)
    if run_quantum:
        enabled_model_count += len(quantum_models)

    if enabled_model_count == 0:
        raise ValueError("Enable at least one group: run_classical or run_quantum")

    total_runs = len(datasets) * len(seeds) * enabled_model_count
    progress = tqdm(total=total_runs, desc="All runs", dynamic_ncols=True)

    for seed_ in seeds:
        seed = 41 + seed_
        seed_everything(seed)

        classical_csv = out_dir / f"classical_results_{seed}.csv"
        quantum_csv = out_dir / f"quantum_results_{seed}.csv"

        for dataset_name in datasets:
            (
                X_train_tensor,
                X_test_tensor,
                y_train_tensor,
                y_test_tensor,
                target_names,
            ) = prepare_data(dataset_name, test_size, seed, quiet_logs=quiet_logs)

            progress.write(
                f"Dataset={dataset_name} | train={X_train_tensor.shape[0]} | test={X_test_tensor.shape[0]} | features={X_train_tensor.shape[1]} | seed={seed}"
            )

            input_dim = X_train_tensor.shape[1]
            output_dim = len(target_names)

            if run_classical:
                for model_name in classical_models:
                    model = build_classical_model(
                        model_name=model_name,
                        input_dim=input_dim,
                        output_dim=output_dim,
                        hidden_dim=hidden_dim,
                        n_chunks=n_chunks,
                    )

                    model, training_time = train_model(
                        model=model,
                        X_train=X_train_tensor,
                        y_train=y_train_tensor,
                        epochs=epochs,
                        lr=learning_rate,
                        desc=f"{dataset_name} | {model_name}",
                    )
                    accuracy, inference_time = evaluate_model(
                        model, X_test_tensor, y_test_tensor
                    )

                    row = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "dataset": dataset_name,
                        "model": model_name,
                        "accuracy": round(accuracy, 6),
                        "training_time_sec": round(training_time, 6),
                        "inference_time_sec": round(inference_time, 6),
                        "epochs": epochs,
                        "test_size": test_size,
                        "learning_rate": learning_rate,
                        "hidden_dim": hidden_dim,
                        "n_chunks": n_chunks,
                        "num_qubits": "",
                        "input_dim": input_dim,
                        "num_classes": output_dim,
                        "train_size": int(X_train_tensor.shape[0]),
                        "test_count": int(X_test_tensor.shape[0]),
                        "total_params": count_trainable_parameters(model),
                        "seed": seed,
                    }
                    append_row(classical_csv, row, headers)
                    progress.update(1)

            if run_quantum:
                for model_name in quantum_models:
                    model = build_quantum_model(
                        model_name=model_name,
                        input_dim=input_dim,
                        output_dim=output_dim,
                        num_qubits=num_qubits,
                        n_chunks=n_chunks,
                        quiet_logs=quiet_logs,
                    )

                    model, training_time = train_model(
                        model=model,
                        X_train=X_train_tensor,
                        y_train=y_train_tensor,
                        epochs=epochs,
                        lr=learning_rate,
                        desc=f"{dataset_name} | {model_name}",
                    )
                    accuracy, inference_time = evaluate_model(
                        model, X_test_tensor, y_test_tensor
                    )

                    row = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "dataset": dataset_name,
                        "model": model_name,
                        "accuracy": round(accuracy, 6),
                        "training_time_sec": round(training_time, 6),
                        "inference_time_sec": round(inference_time, 6),
                        "epochs": epochs,
                        "test_size": test_size,
                        "learning_rate": learning_rate,
                        "hidden_dim": "",
                        "n_chunks": n_chunks,
                        "num_qubits": num_qubits,
                        "input_dim": input_dim,
                        "num_classes": output_dim,
                        "train_size": int(X_train_tensor.shape[0]),
                        "test_count": int(X_test_tensor.shape[0]),
                        "total_params": count_trainable_parameters(model),
                        "seed": seed,
                    }
                    append_row(quantum_csv, row, headers)
                    progress.update(1)

    progress.close()
    print(f"Saved classical table: {classical_csv}")
    print(f"Saved quantum table: {quantum_csv}")


if __name__ == "__main__":
    main()
