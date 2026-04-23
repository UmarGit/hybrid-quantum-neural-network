import os
import time
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from models.classical_models import (
    BaseClassicalNN,
    ClassicalMLP,
    IterativeClassicalNN,
    SplitAttentionClassicalNN,
    ResNet,
)
from models.datasets import load_dataset


# Training
def train_classical(
    model: BaseClassicalNN,
    epochs: 10,
    X_train_tensor: torch.FloatTensor,
    y_train_tensor: torch.LongTensor | None,
):
    start_train = time.time()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train_tensor)
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
def inference_classical(
    model: BaseClassicalNN,
    X_test_tensor: torch.FloatTensor,
    y_test_tensor: torch.LongTensor | None,
):
    accuracy = 0.0
    start_train = time.time()
    model.eval()

    with torch.no_grad():
        output = model(X_test_tensor)
        _, predictions = torch.max(output, 1)
        if y_test_tensor is not None:
            accuracy = (predictions == y_test_tensor).sum().item() / len(y_test_tensor)

    end_train = time.time()
    inference_time = end_train - start_train

    return accuracy, inference_time, predictions


def plot_pca_decision_boundary_classical(
    model: BaseClassicalNN,
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
    plt.savefig("classical_decision_boundary.png")
    plt.show()


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seeds set to {seed}")


if __name__ == "__main__":
    # Data Preparation
    seed_everything(42)

    dataset_name = "leukemia"

    X, y, target_names = load_dataset(dataset_name)
    test_size = 0.9

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Standardize the data
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_train_tensor = torch.LongTensor(y_train)
    y_test_tensor = torch.LongTensor(y_test)

    hidden_dim = 4
    n_chunks = 2

    classicalqmlp = ClassicalMLP(
        input_dim=X_train_tensor.shape[1],
        output_dim=len(target_names),
        hidden_dim=hidden_dim,
    )
    iterativeqnn = IterativeClassicalNN(
        input_dim=X_train_tensor.shape[1],
        hidden_dim=8,
        intermediate_dim=hidden_dim,
        num_iterations=2,
    )
    splitattentionqnn = SplitAttentionClassicalNN(
        input_dim=X_train_tensor.shape[1], hidden_dim=hidden_dim, n_chunks=n_chunks
    )
    resqnet = ResNet(
        input_dim=X_train_tensor.shape[1],
        output_dim=len(target_names),
        hidden_dim=hidden_dim,
        residual_gate_init=0.1,
    )

    models = [classicalqmlp, iterativeqnn, splitattentionqnn, resqnet]
    titles = ["Classical", "Iterative", "Split Attention", "ResQnet"]

    epochs = 20

    for index, model in enumerate(models):
        trained_model, training_time, losses = train_classical(
            model, epochs, X_train_tensor, y_train_tensor
        )
        accuracy, inference_time, _ = inference_classical(
            trained_model, X_test_tensor, y_test_tensor
        )

        # torch.save(classical_model.state_dict(), "classical_model.pt")

        # print(y_test_tensor.shape)

        total_params = sum(
            p.numel() for p in trained_model.parameters() if p.requires_grad
        )

        print("-" * 30)
        print(f"       Classical MODEL RESULTS: {titles[index]}       ")
        print("-" * 30)
        print(f"Accuracy:           {accuracy * 100:.2f}%")
        print(f"Training Time:      {training_time:.5f} sec")
        print(f"Inference Time:      {inference_time:.5f} sec")
        print(f"Hidden Dim:           {hidden_dim}")
        print(f"N_Chunks:           {n_chunks}")
        print(f"Test Size:           {test_size}")
        print(f"Total number of parameters: {total_params}")
        print("-" * 30)

        # # Use the tensors from your successful run
        # plot_pca_decision_boundary_classical(
        #     trained_model, X_test_tensor, y_test_tensor, target_names
        # )
