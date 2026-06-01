"""Run configuration and the matched-pair registry.

This module only *orchestrates* the user's existing model classes; it imports
them, never redefines them. Each classical counterpart is built from the
*quantum* model's own audited attributes (``num_qubits``, ``hidden_dim``,
``n_chunks`` ...) so the matched dimensions contain no magic numbers and stay
in lock-step with the quantum architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch.nn as nn

# Quantum / classical / SVM models are reused by import only.
from models.quantum_models import (
    ClassicalQuantumMLP,
    IterativeQNN,
    SplitAttentionQNN,
    ResQNet,
    QKAResQNet,
)
from models.classical_models import (
    ClassicalMLP,
    IterativeClassicalNN,
    SplitAttentionClassicalNN,
    ResNet,
    CKAResCNet,
)
from models.svm_models import ClassicalSVM, QuantumSVM


@dataclass(frozen=True)
class BenchmarkConfig:
    """Identical training configuration applied to both models in a pair."""

    dataset: str
    n_seeds: int = 20
    epochs: int = 50
    learning_rate: float = 0.1
    test_split: float = 0.9
    num_qubits: int = 4
    n_chunks: int = 3
    gate_limit: float = 0.1
    base_seed: int = 42
    entanglement: str = "full"
    noise_mode: str = "kingston"
    output_dir: str = "benchmark_results"


# A quantum builder wraps the prepared QNN plus data dims and run config into
# the quantum model. A classical builder receives the constructed quantum model
# plus data dims and returns the parameter-matched classical counterpart.
QuantumBuilder = Callable[[object, int, int, BenchmarkConfig], nn.Module]
ClassicalBuilder = Callable[[nn.Module, int, int, BenchmarkConfig], nn.Module]


def _quantum_mlp(qnn, in_dim: int, out_dim: int, cfg: BenchmarkConfig) -> nn.Module:
    return ClassicalQuantumMLP(
        qnn, input_dim=in_dim, output_dim=out_dim, num_qubits=cfg.num_qubits
    )


def _quantum_iterative(qnn, in_dim: int, out_dim: int, cfg: BenchmarkConfig) -> nn.Module:
    return IterativeQNN(
        qnn, input_dim=in_dim, output_dim=out_dim, num_qubits=cfg.num_qubits
    )


def _quantum_split_attention(
    qnn, in_dim: int, out_dim: int, cfg: BenchmarkConfig
) -> nn.Module:
    return SplitAttentionQNN(
        qnn,
        input_dim=in_dim,
        output_dim=out_dim,
        num_qubits=cfg.num_qubits,
        n_chunks=cfg.n_chunks,
    )


def _quantum_resnet(qnn, in_dim: int, out_dim: int, cfg: BenchmarkConfig) -> nn.Module:
    return ResQNet(
        qnn, input_dim=in_dim, output_dim=out_dim, num_qubits=cfg.num_qubits
    )


def _quantum_cka(qnn, in_dim: int, out_dim: int, cfg: BenchmarkConfig) -> nn.Module:
    return QKAResQNet(
        qnn,
        input_dim=in_dim,
        output_dim=out_dim,
        num_qubits=cfg.num_qubits,
        quantum_gate_limit=cfg.gate_limit,
    )


def _classical_mlp(
    q: nn.Module, in_dim: int, out_dim: int, cfg: BenchmarkConfig
) -> nn.Module:
    return ClassicalMLP(input_dim=in_dim, output_dim=out_dim, hidden_dim=cfg.num_qubits)


def _classical_iterative(
    q: nn.Module, in_dim: int, out_dim: int, cfg: BenchmarkConfig
) -> nn.Module:
    # hidden_dim mirrors the quantum model's outer width (not a config knob).
    return IterativeClassicalNN(
        input_dim=in_dim,
        output_dim=out_dim,
        hidden_dim=q.hidden_dim,
        intermediate_dim=cfg.num_qubits,
        num_iterations=q.num_iterations,
    )


def _classical_split_attention(
    q: nn.Module, in_dim: int, out_dim: int, cfg: BenchmarkConfig
) -> nn.Module:
    return SplitAttentionClassicalNN(
        input_dim=in_dim,
        output_dim=out_dim,
        hidden_dim=cfg.num_qubits,
        n_chunks=cfg.n_chunks,
    )


def _classical_resnet(
    q: nn.Module, in_dim: int, out_dim: int, cfg: BenchmarkConfig
) -> nn.Module:
    return ResNet(input_dim=in_dim, output_dim=out_dim, hidden_dim=cfg.num_qubits)


def _classical_cka(
    q: nn.Module, in_dim: int, out_dim: int, cfg: BenchmarkConfig
) -> nn.Module:
    return CKAResCNet(
        input_dim=in_dim,
        output_dim=out_dim,
        hidden_dim=cfg.num_qubits,
        residual_gate_init=cfg.gate_limit,
    )


@dataclass(frozen=True)
class ModelPair:
    """A quantum model and its parameter-matched classical counterpart.

    ``expected_delta`` is the audited absolute parameter difference that the
    pair must exhibit so the comparison isolates the quantum layer. It is passed
    to :class:`~benchmark.ablation.AblationValidator` per call.
    """

    name: str
    quantum_cls: type
    classical_cls: type
    expected_delta: int = 8
    quantum_builder: QuantumBuilder | None = None
    classical_builder: ClassicalBuilder | None = None
    is_svm: bool = False


# Audited matched pairs. ``expected_delta`` values are the *empirical* absolute
# parameter deltas for the committed model code at num_qubits=4. NOTE: the
# split-attention pair differs by 48 (one shared QNN of 12 params vs. three
# classical layers of 20), not the 28 quoted in the thesis spec — that figure
# is inconsistent with the current SplitAttention*NN code. Override per call if
# you change the architecture.
PAIR_REGISTRY: dict[str, ModelPair] = {
    "mlp": ModelPair(
        "mlp", ClassicalQuantumMLP, ClassicalMLP, 8, _quantum_mlp, _classical_mlp
    ),
    "iterative": ModelPair(
        "iterative",
        IterativeQNN,
        IterativeClassicalNN,
        8,
        _quantum_iterative,
        _classical_iterative,
    ),
    "split_attention": ModelPair(
        "split_attention",
        SplitAttentionQNN,
        SplitAttentionClassicalNN,
        48,
        _quantum_split_attention,
        _classical_split_attention,
    ),
    "resnet": ModelPair(
        "resnet", ResQNet, ResNet, 8, _quantum_resnet, _classical_resnet
    ),
    "kernel_alignment": ModelPair(
        "kernel_alignment", QKAResQNet, CKAResCNet, 8, _quantum_cka, _classical_cka
    ),
    "svm": ModelPair(
        "svm", QuantumSVM, ClassicalSVM, expected_delta=0, is_svm=True
    ),
}
