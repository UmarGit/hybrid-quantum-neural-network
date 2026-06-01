"""
Models module for quantum-neural-network.

Contains classical, quantum, and hybrid neural network architectures,
as well as classical and quantum SVM implementations.
"""

# Classical Neural Networks
from .classical_models import (
    ClassicalMLP,
    IterativeClassicalNN,
    SplitAttentionClassicalNN,
    ResNet,
    CKAResCNet,
)

# Quantum Neural Networks
from .quantum_models import (
    ClassicalQuantumMLP,
    IterativeQNN,
    SplitAttentionQNN,
    ResQNet,
    QKAResQNet,
)

# SVM Models
from .svm_models import (
    ClassicalSVM,
    QuantumSVM,
)

__all__ = [
    # Classical NN
    "ClassicalMLP",
    "IterativeClassicalNN",
    "SplitAttentionClassicalNN",
    "ResNet",
    "CKAResCNet",
    # Quantum NN
    "ClassicalQuantumMLP",
    "IterativeQNN",
    "SplitAttentionQNN",
    "ResQNet",
    "QKAResQNet",
    # SVM
    "ClassicalSVM",
    "QuantumSVM",
]
