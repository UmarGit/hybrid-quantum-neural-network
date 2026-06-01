"""Aer noise models for audited IBM hardware profiles.

All hardware error rates live in the constants block below and nowhere else.
Profiles are audited single-/two-qubit gate and readout error rates taken from
IBM Quantum Platform calibration data.
"""

from __future__ import annotations

from dataclasses import dataclass

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error


# --------------------------------------------------------------------------- #
# AUDITED HARDWARE NOISE CONSTANTS (the only magic numbers in this package).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NoiseProfile:
    """Audited error rates for a single backend."""

    name: str
    sx_error: float  # single-qubit (sx) gate depolarizing error
    two_qubit_error: float  # two-qubit gate depolarizing error
    readout_error: float  # symmetric measurement bit-flip probability


TORINO = NoiseProfile(
    name="torino",
    sx_error=0.00033,
    two_qubit_error=0.0062,
    readout_error=0.0303,
)
KINGSTON = NoiseProfile(
    name="kingston",
    sx_error=0.000286,
    two_qubit_error=0.0020,
    readout_error=0.0109,
)

NOISELESS = "noiseless"

# Gate sets the depolarizing errors are applied to.
_ONE_QUBIT_GATES = ["rx", "ry", "rz", "h", "u", "sx", "x"]
_TWO_QUBIT_GATES = ["cx", "cz", "ecr"]

_PROFILES: dict[str, NoiseProfile] = {TORINO.name: TORINO, KINGSTON.name: KINGSTON}


class IBMNoiseModel:
    """Build an Aer ``NoiseModel`` / backend for a named profile.

    ``mode`` is one of ``"torino"``, ``"kingston"`` or ``"noiseless"``.
    """

    def __init__(self, mode: str = NOISELESS) -> None:
        mode = mode.lower()
        if mode != NOISELESS and mode not in _PROFILES:
            raise ValueError(
                f"Unknown noise mode {mode!r}; expected one of "
                f"{[NOISELESS, *_PROFILES]}."
            )
        self.mode = mode
        self.profile: NoiseProfile | None = _PROFILES.get(mode)

    def build_model(self) -> NoiseModel | None:
        """Return the Aer ``NoiseModel`` or ``None`` for the noiseless mode."""
        if self.profile is None:
            return None

        noise_model = NoiseModel()

        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(self.profile.sx_error, 1), _ONE_QUBIT_GATES
        )
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(self.profile.two_qubit_error, 2), _TWO_QUBIT_GATES
        )

        p = self.profile.readout_error
        noise_model.add_all_qubit_readout_error(
            ReadoutError([[1 - p, p], [p, 1 - p]])
        )
        return noise_model

    def build_backend(self, seed: int | None = None) -> AerSimulator:
        """Return an ``AerSimulator`` configured for this profile and seed."""
        noise_model = self.build_model()
        if noise_model is None:
            backend = AerSimulator()
        else:
            backend = AerSimulator(
                noise_model=noise_model,
                method="density_matrix",
                max_parallel_threads=0,
                max_parallel_experiments=0,
            )
        if seed is not None:
            backend.set_options(seed_simulator=seed)
        return backend
