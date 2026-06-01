"""Tests for the audited Aer noise profiles."""

import pytest
from qiskit_aer.noise import NoiseModel

from benchmark.noise import KINGSTON, NOISELESS, TORINO, IBMNoiseModel


def test_audited_constants() -> None:
    # Values are pinned to the audited thesis spec; guard against drift.
    assert (TORINO.sx_error, TORINO.two_qubit_error, TORINO.readout_error) == (
        0.00033,
        0.0062,
        0.0303,
    )
    assert (
        KINGSTON.sx_error,
        KINGSTON.two_qubit_error,
        KINGSTON.readout_error,
    ) == (0.000286, 0.0020, 0.0109)


@pytest.mark.parametrize("mode", ["torino", "kingston"])
def test_noisy_model_built(mode: str) -> None:
    model = IBMNoiseModel(mode).build_model()
    assert isinstance(model, NoiseModel)
    # Readout + one/two-qubit gate errors should register at least one basis gate.
    assert model.noise_instructions


def test_noiseless_has_no_model() -> None:
    assert IBMNoiseModel(NOISELESS).build_model() is None


def test_backend_construction() -> None:
    backend = IBMNoiseModel(NOISELESS).build_backend(seed=42)
    assert backend.options.seed_simulator == 42


def test_unknown_mode_rejected() -> None:
    with pytest.raises(ValueError):
        IBMNoiseModel("ibm_fake")
