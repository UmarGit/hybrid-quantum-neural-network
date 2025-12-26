import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


# Classical MLP Model
class ClassicalMLP(nn.Module):
    def __init__(self, input_dim = 2, output_dim = 3):
        super(ClassicalMLP, self).__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, 8), nn.ReLU(), nn.Linear(8, output_dim))

    def forward(self, x):
        return self.network(x)


# Training
def train_classical(
    model: ClassicalMLP,
    epochs: 10,
    X_train_tensor: torch.FloatTensor,
    y_train_tensor: torch.LongTensor | None,
):
    start_train = time.time()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train_tensor)
        print(output.shape, y_train_tensor.shape)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss.item():.4f}")

    end_train = time.time()
    training_time = end_train - start_train

    return model, training_time


# Inference & Evaluation
def inference_classical(
    model: ClassicalMLP,
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
    model: ClassicalMLP,
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

    _, _, predictions = inference_classical(model, grid_tensor, None)

    Z = predictions.reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.viridis)

    scatter = plt.scatter(
        X_scaled[:, 0], X_scaled[:, 1], c=y, edgecolors="k", s=60, cmap=plt.cm.viridis
    )

    plt.title("Classical MLP Boundary")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(handles=scatter.legend_elements()[0], labels=target_names)
    plt.savefig("classical_desision_boundary.png")

def plot_3d_decision_boundary_classical(model, X_scaled, y, target_names):
    x_min, x_max = X_scaled[:, 0].min() - 0.2, X_scaled[:, 0].max() + 0.2
    y_min, y_max = X_scaled[:, 1].min() - 0.2, X_scaled[:, 1].max() + 0.2
    h = 0.1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h, dtype="float"),
        np.arange(y_min, y_max, h, dtype="float"),
    )
    grid_tensor = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])

    _, _, predictions = inference_classical(model, grid_tensor, None)
    
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

    ax.set_title("3D Classical Decision Landscape")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Classes")
    
    legend1 = ax.legend(*scatter.legend_elements(), title="Data", loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.add_artist(legend1)

    ax.view_init(elev=30, azim=-60)
    
    plt.savefig("classical_3d_decision_boundary.png")
    print("3D Plot saved as 'classical_3d_decision_boundary.png'")
    plt.show()

if __name__ == "__main__":
    # Data Preparation
    iris = load_iris()

    X = iris.data
    y = iris.target
    target_names = iris.target_names.tolist()

    print(target_names)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pca = PCA(n_components=2)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    # Standardize the data
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_train_tensor = torch.LongTensor(y_train)
    y_test_tensor = torch.LongTensor(y_test)

    classical_model = ClassicalMLP()

    trained_model, training_time = train_classical(
        classical_model, 20, X_train_tensor, y_train_tensor
    )
    accuracy, inference_time, _ = inference_classical(
        trained_model, X_test_tensor, y_test_tensor
    )

    print(y_test_tensor.shape)

    print("-" * 30)
    print("       Classica MODEL RESULTS       ")
    print("-" * 30)
    print(f"Accuracy:           {accuracy * 100:.2f}%")
    print(f"Training Time:      {training_time:.5f} sec")
    print(f"Inference Time:      {inference_time:.5f} sec")
    print("-" * 30)

    # Use the tensors from your successful run
    plot_pca_decision_boundary_classical(
        trained_model, X_test_tensor, y_test_tensor, target_names
    )
