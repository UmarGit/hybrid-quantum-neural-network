"""Modular benchmarking package orchestrating the thesis QML models.

This package reuses the existing model, dataset and circuit code by import and
adds reproducible multi-seed benchmarking, ablation parity checks, audited Aer
noise profiles, and significance-based statistical comparison.
"""

from __future__ import annotations

from .ablation import AblationValidator, count_trainable_parameters
from .config import PAIR_REGISTRY, BenchmarkConfig, ModelPair
from .data import DataSplit, DataStarvationModule
from .engine import ExecutionEngine, ModelSummary, build_qnn
from .noise import KINGSTON, NOISELESS, TORINO, IBMNoiseModel, NoiseProfile
from .stats import ComparisonResult, StatisticalAnalyzer

__all__ = [
    "AblationValidator",
    "count_trainable_parameters",
    "BenchmarkConfig",
    "ModelPair",
    "PAIR_REGISTRY",
    "DataSplit",
    "DataStarvationModule",
    "ExecutionEngine",
    "ModelSummary",
    "build_qnn",
    "IBMNoiseModel",
    "NoiseProfile",
    "TORINO",
    "KINGSTON",
    "NOISELESS",
    "ComparisonResult",
    "StatisticalAnalyzer",
]
