import abc
from typing import List
import torch
import torch.nn as nn
import numpy as np


class BaseClassicalNN(abc.ABC, nn.Module):
    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class ClassicalMLP(BaseClassicalNN):
    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 3,
        hidden_dim: int = 2,
    ) -> None:
        super(ClassicalMLP, self).__init__()

        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.hidden_dim: int = hidden_dim

        self.pre_network: nn.Linear = nn.Linear(input_dim, hidden_dim)
        self.activation: nn.Tanh = nn.Tanh()

        # Replace quantum layer with classical layer
        self.classical_layer: nn.Linear = nn.Linear(hidden_dim, hidden_dim)
        self.mid_activation = nn.Tanh()

        self.post_network: nn.Linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_network(x)
        x = self.activation(x)
        x = x * np.pi

        x = self.classical_layer(x)
        x = self.mid_activation(x)

        x = self.post_network(x)
        return x


class IterativeClassicalNN(BaseClassicalNN):
    """Classical iterative network equivalent to IterativeQNN."""

    def __init__(
        self,
        input_dim: int = 30,
        hidden_dim: int = 8,
        intermediate_dim: int = 4,
        num_iterations: int = 2,
    ) -> None:
        super(IterativeClassicalNN, self).__init__()

        self.input_dim: int = input_dim
        self.hidden_dim: int = hidden_dim
        self.intermediate_dim: int = intermediate_dim
        self.num_iterations: int = num_iterations

        self.input_proj: nn.Linear = nn.Linear(input_dim, hidden_dim)

        # The Recurrent Block (Input hidden_dim -> intermediate -> hidden_dim)
        self.pre_layer: nn.Linear = nn.Linear(hidden_dim, intermediate_dim)
        # Replace quantum layer with classical layer
        self.classical_layer: nn.Linear = nn.Linear(intermediate_dim, intermediate_dim)
        self.post_layer: nn.Linear = nn.Linear(intermediate_dim, hidden_dim)

        self.final_classify: nn.Linear = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z: torch.Tensor = torch.tanh(self.input_proj(x))

        for _ in range(self.num_iterations):
            prev_z: torch.Tensor = z

            # Classical processing
            layer_in: torch.Tensor = torch.tanh(self.pre_layer(z)) * np.pi
            layer_out: torch.Tensor = torch.tanh(self.classical_layer(layer_in))

            delta_z: torch.Tensor = torch.tanh(self.post_layer(layer_out))

            # Residual connection
            z = prev_z + delta_z

        return self.final_classify(z)


class SplitAttentionClassicalNN(BaseClassicalNN):
    """Classical multi-path network equivalent to SplitAttentionQNN."""

    def __init__(
        self,
        input_dim: int = 30,
        hidden_dim: int = 4,
        n_chunks: int = 3,
    ) -> None:
        super(SplitAttentionClassicalNN, self).__init__()

        self.input_dim: int = input_dim
        self.hidden_dim: int = hidden_dim
        self.n_chunks: int = n_chunks
        self.chunk_size: int = input_dim // n_chunks

        dummy_tensor = torch.empty(1, input_dim)
        dummy_chunks = torch.chunk(dummy_tensor, n_chunks, dim=1)

        # Compressors for each chunk (sizes now perfectly match)
        self.compressors: nn.ModuleList = nn.ModuleList(
            [nn.Linear(chunk.shape[1], hidden_dim) for chunk in dummy_chunks]
        )

        # Replace quantum layer with classical layers
        self.classical_layers: nn.ModuleList = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_chunks)]
        )

        # Attention mechanism
        self.attention: nn.Linear = nn.Linear(hidden_dim * n_chunks, n_chunks)
        self.classifier: nn.Linear = nn.Linear(hidden_dim * n_chunks, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks: tuple[torch.Tensor, ...] = torch.chunk(x, self.n_chunks, dim=1)
        classical_outputs: List[torch.Tensor] = []

        for i, chunk in enumerate(chunks):
            # Compress chunk
            c_in: torch.Tensor = torch.tanh(self.compressors[i](chunk)) * np.pi
            # Classical processing
            c_out: torch.Tensor = self.classical_layers[i](c_in)
            classical_outputs.append(c_out)

        combined: torch.Tensor = torch.cat(classical_outputs, dim=1)
        return self.classifier(combined)


class ResNet(BaseClassicalNN):
    """Classical residual network equivalent to ResQNet."""

    def __init__(
        self,
        input_dim: int = 30,
        output_dim: int = 2,
        hidden_dim: int = 4,
        residual_gate_init: float = 0.1,
    ) -> None:
        super(ResNet, self).__init__()

        self.input_dim: int = input_dim
        self.hidden_dim: int = hidden_dim
        self.output_dim: int = output_dim

        # Classical "Highway" (The Bypass)
        self.classical_highway: nn.Linear = nn.Linear(input_dim, output_dim)

        # Residual Path
        self.compressor: nn.Linear = nn.Linear(input_dim, hidden_dim)
        # Replace quantum layer with classical layer
        self.classical_layer: nn.Linear = nn.Linear(hidden_dim, hidden_dim)
        self.expander: nn.Linear = nn.Linear(hidden_dim, output_dim)

        # Gate to decide how much residual path to add
        self.residual_gate: nn.Parameter = nn.Parameter(
            torch.tensor(residual_gate_init, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Path 1: Classical highway
        classical_out: torch.Tensor = self.classical_highway(x)

        # Path 2: Residual path
        layer_in: torch.Tensor = torch.tanh(self.compressor(x)) * np.pi
        layer_out: torch.Tensor = self.classical_layer(layer_in)
        residual_out: torch.Tensor = self.expander(layer_out)

        # Combine: Output = Highway + (Gate * Residual)
        return classical_out + (self.residual_gate * residual_out)

