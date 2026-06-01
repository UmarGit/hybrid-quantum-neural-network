import abc
from typing import Generic, TypeVar
import torch
import torch.nn as nn
import numpy as np

# Qiskit Imports
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

QNNType = TypeVar("QNNType", bound=EstimatorQNN)


class BaseQNN(abc.ABC, nn.Module, Generic[QNNType]):
    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


# Classical MLP Model
class ClassicalQuantumMLP(BaseQNN[EstimatorQNN]):
    def __init__(
        self,
        qnn: EstimatorQNN,
        input_dim: int = 2,
        output_dim: int = 3,
        num_qubits: int = 2,
    ):
        super(ClassicalQuantumMLP, self).__init__()

        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.num_qubits: int = num_qubits

        self.pre_network = nn.Linear(input_dim, num_qubits)
        self.activation = nn.Tanh()

        self.qnn: EstimatorQNN = TorchConnector(qnn)

        self.post_network = nn.Linear(num_qubits, output_dim)

    def forward(self, x):
        x = self.pre_network.forward(x)
        x = self.activation.forward(x)

        x = x * np.pi

        x = self.qnn.forward(x)
        x = self.post_network.forward(x)

        return x


class IterativeQNN(BaseQNN[EstimatorQNN]):
    def __init__(
        self,
        qnn: EstimatorQNN,
        input_dim: int = 30,
        output_dim: int = 2,
        num_qubits: int = 4,
        hidden_dim: int = 8,
        num_iterations: int = 2,
    ):
        super(IterativeQNN, self).__init__()

        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.num_qubits: int = num_qubits
        self.hidden_dim: int = hidden_dim
        self.num_iterations: int = num_iterations

        # Dimensions must match for iterative loop: Input = Output
        # We project 30 features -> Hidden Dimension (e.g. 8)
        self.input_proj = nn.Linear(input_dim, self.hidden_dim)

        # The Recurrent Block (Input 8 -> QNN -> Output 8)
        self.pre_q = nn.Linear(self.hidden_dim, num_qubits)
        self.qnn: EstimatorQNN = TorchConnector(qnn)
        self.post_q = nn.Linear(num_qubits, self.hidden_dim)

        self.final_classify = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, x):
        # 1. Project to hidden space
        z = torch.tanh(self.input_proj(x))

        # 2. Iterative Refinement (Run the loop K times)
        steps = 2  # You can increase this to 3 or 4
        for _ in range(steps):
            # Remember the residual connection (z + ...) is crucial for convergence!
            prev_z = z

            # Classical -> Quantum
            q_in = torch.tanh(self.pre_q(z)) * np.pi
            q_out = self.qnn.forward(q_in)

            # Quantum -> Classical
            delta_z = torch.tanh(self.post_q(q_out))

            # Update state
            z = prev_z + delta_z

        return self.final_classify.forward(z)


class SplitAttentionQNN(BaseQNN[EstimatorQNN]):
    def __init__(
        self,
        qnn: EstimatorQNN,
        input_dim: int = 30,
        output_dim: int = 2,
        num_qubits: int = 4,
        n_chunks: int = 3,
    ):
        super(SplitAttentionQNN, self).__init__()

        # We split 30 features into 3 chunks of 10
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.num_qubits: int = num_qubits
        self.n_chunks: int = n_chunks
        self.chunk_size: int = input_dim // n_chunks

        dummy_tensor = torch.empty(1, input_dim)
        dummy_chunks = torch.chunk(dummy_tensor, n_chunks, dim=1)

        # Compressors for each chunk (sizes now perfectly match)
        self.compressors: nn.ModuleList = nn.ModuleList(
            [nn.Linear(chunk.shape[1], num_qubits) for chunk in dummy_chunks]
        )

        self.qnn: EstimatorQNN = TorchConnector(qnn)

        # Attention Mechanism: Decides which chunk's quantum output matters most
        self.attention = nn.Linear(num_qubits * self.n_chunks, self.n_chunks)
        self.classifier = nn.Linear(num_qubits * self.n_chunks, output_dim)

    def forward(self, x):
        # x shape: [batch, 30]
        chunks = torch.chunk(
            x, self.n_chunks, dim=1
        )  # Split into 3 tensors of [batch, 10]

        quantum_outputs = []
        for i, chunk in enumerate(chunks):
            # Compress 10 -> 4
            c_in = torch.tanh(self.compressors[i](chunk)) * np.pi
            # Run Quantum Circuit
            q_out = self.qnn.forward(c_in)  # [batch, 4]
            quantum_outputs.append(q_out)

        # Concatenate all quantum insights [batch, 12]
        combined = torch.cat(quantum_outputs, dim=1)

        # 1. Calculate attention scores for the 3 chunks [batch, 3]
        attn_logits = self.attention(combined)

        # 2. Convert to probabilities using Softmax [batch, 3]
        attn_weights = torch.softmax(attn_logits, dim=1)

        # 3. Reshape 'combined' so we can apply weights to each chunk
        # [batch, 12] -> [batch, 3, 4]
        combined_reshaped = combined.view(-1, self.n_chunks, self.num_qubits)

        # 4. Expand weights to match the 4 qubits: [batch, 3] -> [batch, 3, 1]
        # and multiply. Broadcasting will apply the weight to all 4 qubits of that chunk.
        attended_chunks = combined_reshaped * attn_weights.unsqueeze(-1)

        # 5. Flatten back to [batch, 12]
        attended_combined = attended_chunks.view(-1, self.n_chunks * self.num_qubits)

        # Final linear classification on the attention-weighted insights
        return self.classifier(attended_combined)


class ResQNet(BaseQNN[EstimatorQNN]):
    def __init__(
        self,
        qnn: EstimatorQNN,
        input_dim: int = 30,
        output_dim: int = 3,
        num_qubits: int = 4,
        quantum_gate_limit: float = 0.1,
    ):
        super(ResQNet, self).__init__()

        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.num_qubits: int = num_qubits
        self.quantum_gate_limit: int = quantum_gate_limit

        # Classical "Highway" (The Bypass)
        self.classical_highway = nn.Linear(input_dim, output_dim)

        # Quantum "Residual" Path
        self.compressor = nn.Linear(input_dim, num_qubits)
        self.qnn: EstimatorQNN = TorchConnector(qnn)
        self.expander = nn.Linear(num_qubits, output_dim)

        # Gate to decide how much quantum "flavor" to add
        self.quantum_gate = nn.Parameter(
            torch.tensor(self.quantum_gate_limit, dtype=torch.float32)
        )  # Start small

    def forward(self, x):
        # Path 1: Classical (Fast, handles broad patterns)
        classical_out = self.classical_highway.forward(x)

        # Path 2: Quantum (Captures complex correlations)
        q_in = torch.tanh(self.compressor(x)) * np.pi  # Scale for rotations
        q_out = self.qnn.forward(q_in)
        q_out = self.expander.forward(q_out)

        # Combine: Output = Classical + (Gate * Quantum)
        output = classical_out + (self.quantum_gate * q_out)

        return output


class QKAResQNet(BaseQNN[EstimatorQNN]):
    """
    Enhanced ResQNet utilizing Quantum Kernel Alignment (QKA).
    The quantum feature map is dynamically aligned to the dataset geometry.
    """

    def __init__(
        self,
        qnn: EstimatorQNN,
        input_dim: int = 30,
        output_dim: int = 2,
        num_qubits: int = 4,
        quantum_gate_limit: float = 0.1,
    ):
        super(QKAResQNet, self).__init__()

        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.num_qubits: int = num_qubits

        # 1. Classical "Highway"
        self.classical_highway = nn.Linear(input_dim, output_dim)

        # 2. Trainable Quantum Kernel Alignment (QKA) Layer
        self.qka_alignment = nn.Linear(input_dim, num_qubits, bias=False)

        # Initialize alignment weights to identity-like variance to start neutral
        nn.init.xavier_uniform_(self.qka_alignment.weight)

        # 3. Quantum Circuit & Expander
        self.qnn: EstimatorQNN = TorchConnector(qnn)
        self.expander = nn.Linear(num_qubits, output_dim)

        # 4. Dynamic Residual Gate
        self.quantum_gate = nn.Parameter(
            torch.tensor([quantum_gate_limit], dtype=torch.float32)
        )

    def forward(self, x):
        # Path 1: Classical bypass
        classical_out = self.classical_highway(x)

        # Path 2: Quantum Kernel Alignment Path
        # The QKA layer actively learns the optimal mapping into the Hilbert space
        aligned_features = self.qka_alignment(x)

        # Strict mapping to the Bloch sphere [-pi, pi]
        q_in = torch.tanh(aligned_features) * np.pi

        q_out = self.qnn(q_in)
        residual_out = self.expander(q_out)

        # Combine
        return classical_out + (self.quantum_gate * residual_out)
