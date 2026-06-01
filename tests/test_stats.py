"""Tests for the significance-based statistical verdict."""

import numpy as np

from benchmark.stats import StatisticalAnalyzer


def test_quantum_wins_when_higher_and_significant() -> None:
    rng = np.random.default_rng(0)
    classical = rng.normal(0.70, 0.01, size=20)
    quantum = rng.normal(0.85, 0.01, size=20)

    result = StatisticalAnalyzer().compare(classical, quantum)
    assert result.significant
    assert result.verdict == "quantum wins"
    assert result.quantum_mean > result.classical_mean


def test_classical_wins_when_higher_and_significant() -> None:
    rng = np.random.default_rng(1)
    classical = rng.normal(0.90, 0.01, size=20)
    quantum = rng.normal(0.75, 0.01, size=20)

    result = StatisticalAnalyzer().compare(classical, quantum)
    assert result.significant
    assert result.verdict == "classical wins"


def test_parity_when_indistinguishable() -> None:
    rng = np.random.default_rng(2)
    classical = rng.normal(0.80, 0.02, size=20)
    quantum = rng.normal(0.80, 0.02, size=20)

    result = StatisticalAnalyzer().compare(classical, quantum)
    assert not result.significant
    assert result.verdict == "parity"


def test_result_reports_both_p_values() -> None:
    result = StatisticalAnalyzer().compare(
        np.full(10, 0.5), np.linspace(0.4, 0.6, 10)
    )
    keys = result.as_dict()
    assert "welch_p" in keys
    assert "mannwhitney_p" in keys
