"""Statistical comparison of per-seed accuracy distributions.

The hypothesis tests reuse the logic from ``run_stat_tests.py`` verbatim
(Welch's t-test with ``equal_var=False`` and a two-sided Mann-Whitney U),
adding only a wins/parity verdict combining significance and direction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class ComparisonResult:
    """Outcome of comparing classical vs quantum accuracy samples."""

    classical_mean: float
    classical_std: float
    quantum_mean: float
    quantum_std: float
    welch_p: float
    mannwhitney_p: float
    significant: bool
    verdict: str  # "quantum wins" | "classical wins" | "parity"

    def as_dict(self) -> dict[str, float | str | bool]:
        return self.__dict__.copy()


class StatisticalAnalyzer:
    """Compare two accuracy samples and emit a significance-based verdict."""

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    def compare(
        self,
        classical: np.ndarray,
        quantum: np.ndarray,
    ) -> ComparisonResult:
        classical = np.asarray(classical, dtype=float)
        quantum = np.asarray(quantum, dtype=float)

        c_mean = float(np.mean(classical))
        q_mean = float(np.mean(quantum))
        c_std = float(np.std(classical, ddof=1))
        q_std = float(np.std(quantum, ddof=1))

        # Welch's t-test (does not assume equal variance) — matches run_stat_tests.
        _, welch_p = stats.ttest_ind(
            classical, quantum, equal_var=False, nan_policy="omit"
        )

        # Two-sided Mann-Whitney U, guarded exactly as in run_stat_tests.
        try:
            _, mw_p = stats.mannwhitneyu(
                classical, quantum, alternative="two-sided"
            )
        except ValueError:
            mw_p = float("nan")

        significant = bool(
            np.isfinite(welch_p)
            and np.isfinite(mw_p)
            and welch_p < self.alpha
            and mw_p < self.alpha
        )

        if significant:
            verdict = "quantum wins" if q_mean > c_mean else "classical wins"
        else:
            verdict = "parity"

        return ComparisonResult(
            classical_mean=c_mean,
            classical_std=c_std,
            quantum_mean=q_mean,
            quantum_std=q_std,
            welch_p=float(welch_p),
            mannwhitney_p=float(mw_p),
            significant=significant,
            verdict=verdict,
        )
