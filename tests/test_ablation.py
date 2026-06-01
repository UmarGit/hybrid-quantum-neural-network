"""Tests for the parameter-parity ablation guard."""

import pytest
import torch.nn as nn

from benchmark.ablation import AblationValidator, count_trainable_parameters


def _module(in_dim: int, out_dim: int) -> nn.Module:
    """A bare linear module with a known parameter count (in*out + out)."""
    return nn.Linear(in_dim, out_dim)


def test_count_trainable_parameters() -> None:
    # Linear(4, 4) -> 4*4 weights + 4 bias = 20.
    assert count_trainable_parameters(_module(4, 4)) == 20


def test_validate_passes_on_expected_delta() -> None:
    # Linear(4,4)=20 vs Linear(2,4)=12 -> delta 8.
    validator = AblationValidator(_module(2, 4), _module(4, 4), expected_delta=8)
    assert validator.validate() == 8


def test_validate_passes_with_per_call_delta() -> None:
    # Linear(4,8)=40 vs Linear(4,4)=20 -> delta 20, supplied per call.
    validator = AblationValidator(_module(4, 8), _module(4, 4), expected_delta=20)
    assert validator.validate() == 20


def test_validate_raises_on_mismatch() -> None:
    validator = AblationValidator(_module(2, 4), _module(4, 4), expected_delta=99)
    with pytest.raises(AssertionError) as exc:
        validator.validate()
    assert "expected 99" in str(exc.value)
    assert "isolating the quantum layer" in str(exc.value)
