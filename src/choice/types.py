"""Core types for choice analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.types import (
    DiscountType,
    TrainingSample,
    UtilityType,
    ValueFunctionParams,
)


@dataclass
class ChoiceModelSpec:
    """Specification for a choice model to train."""

    utility_type: UtilityType
    discount_type: DiscountType
    alpha: float = 1.0  # For power utility

    @property
    def name(self) -> str:
        """Human-readable name for this model."""
        utility_name = self.utility_type.value
        if self.utility_type == UtilityType.POWER:
            utility_name = f"power(α={self.alpha})"
        return f"{utility_name}_{self.discount_type.value}"


# All model combinations to try
ALL_MODEL_SPECS = [
    # Linear utility with different discount functions
    ChoiceModelSpec(UtilityType.LINEAR, DiscountType.EXPONENTIAL),
    ChoiceModelSpec(UtilityType.LINEAR, DiscountType.HYPERBOLIC),
    ChoiceModelSpec(UtilityType.LINEAR, DiscountType.QUASI_HYPERBOLIC),
    # Log utility with different discount functions
    ChoiceModelSpec(UtilityType.LOG, DiscountType.EXPONENTIAL),
    ChoiceModelSpec(UtilityType.LOG, DiscountType.HYPERBOLIC),
    ChoiceModelSpec(UtilityType.LOG, DiscountType.QUASI_HYPERBOLIC),
    # Power utility with different alphas
    ChoiceModelSpec(UtilityType.POWER, DiscountType.EXPONENTIAL, alpha=0.5),
    ChoiceModelSpec(UtilityType.POWER, DiscountType.HYPERBOLIC, alpha=0.5),
    ChoiceModelSpec(UtilityType.POWER, DiscountType.EXPONENTIAL, alpha=0.25),
    ChoiceModelSpec(UtilityType.POWER, DiscountType.EXPONENTIAL, alpha=0.75),
]


@dataclass
class SampleWithHorizon:
    """Training sample with its time horizon info."""

    sample: TrainingSample
    horizon_key: str  # Formatted horizon string


@dataclass
class ProblematicSample:
    """A sample that was filtered out due to data issues."""

    sample_index: int
    horizon_key: str
    reason: str
    details: dict


@dataclass
class AmbiguousSample:
    """A sample where time_horizon is between short_term_time and long_term_time."""

    sample_id: int
    time_horizon_years: float
    time_short_years: float
    time_long_years: float
    reward_short: float
    reward_long: float
    choice: str
    reason: str


@dataclass
class ExtractedSamples:
    """Result of extracting samples - valid samples and ambiguous ones separated."""

    valid_samples: list[SampleWithHorizon]
    ambiguous_samples: list[AmbiguousSample]


@dataclass
class LoadedSamples:
    """Result of loading samples from query IDs."""

    valid_samples: list[SampleWithHorizon]
    ambiguous_samples: list[AmbiguousSample]


# Evaluation types
@dataclass
class EvaluationMetrics:
    """Metrics from model evaluation."""

    accuracy: float
    num_samples: int
    num_correct: int
    short_term_accuracy: float
    long_term_accuracy: float
    n_short: int = 0
    n_long: int = 0
    per_horizon_accuracy: dict = field(default_factory=dict)


@dataclass
class TestResult:
    """Result from testing a model on a specific bucket."""

    bucket_name: str
    metrics: EvaluationMetrics
    predictions: list


@dataclass
class HorizonModelResult:
    """Result from training and testing a model on a specific horizon."""

    horizon: str  # "all", "6 months", "no_horizon", etc.
    spec: ChoiceModelSpec
    trained_params: ValueFunctionParams
    train_loss: float
    train_accuracy: float
    train_samples: int
    test_results: dict[str, TestResult]  # bucket_name -> TestResult
