"""Parameter-parity ablation guard for matched quantum/classical pairs."""

from __future__ import annotations

import torch.nn as nn


def count_trainable_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters in a torch module.

    For the hybrid quantum models this includes the ``TorchConnector`` weights
    (the trainable quantum-ansatz parameters), exactly the quantity the
    ablation contrasts against the classical counterpart's extra layer.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AblationValidator:
    """Assert that a matched pair differs by exactly ``expected_delta`` params.

    The comparison is only meaningful if the two architectures are identical
    except for the quantum layer. ``expected_delta`` is supplied per call
    (default 8; 28 for split-attention pairs) and never hardcoded internally.
    """

    def __init__(
        self,
        q_model: nn.Module,
        c_model: nn.Module,
        expected_delta: int = 8,
    ) -> None:
        self.q_model = q_model
        self.c_model = c_model
        self.expected_delta = expected_delta

    def validate(self) -> int:
        """Return the absolute parameter delta, raising if it is unexpected."""
        q_params = count_trainable_parameters(self.q_model)
        c_params = count_trainable_parameters(self.c_model)
        delta = abs(q_params - c_params)

        assert delta == self.expected_delta, (
            "Ablation parameter-parity check failed: "
            f"|{q_params} (quantum) - {c_params} (classical)| = {delta}, "
            f"expected {self.expected_delta}. The matched pair is not isolating "
            "the quantum layer; adjust the classical counterpart's dimensions."
        )
        return delta
