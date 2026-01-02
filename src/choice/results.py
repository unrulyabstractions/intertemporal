"""Results handling and serialization for choice analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from .config import AnalysisConfig
from .types import HorizonModelResult

if TYPE_CHECKING:
    pass


@dataclass
class AnalysisResult:
    """Full result from choice analysis."""

    config: AnalysisConfig
    train_data_path: str
    test_data_path: str
    model_results: dict[str, list[HorizonModelResult]]  # discount_type -> results
    model_results_no_conflicts: dict[
        str, list[HorizonModelResult]
    ]  # Same but with conflicts filtered
    train_verification: Optional[object]  # Train data alignment verification
    test_verification: Optional[object]  # Test data alignment verification
    train_consistency: Optional[dict]  # Train consistency analysis
    test_consistency: Optional[dict]  # Test consistency analysis
    timestamp: str
    viz_dir: Optional[Path] = None  # Visualization output directory


def serialize_analysis_result(
    result: AnalysisResult,
    serialize_verification_fn=None,
    serialize_consistency_fn=None,
) -> dict:
    """
    Serialize analysis result to dict for saving.

    Args:
        result: AnalysisResult to serialize
        serialize_verification_fn: Optional function to serialize verification
        serialize_consistency_fn: Optional function to serialize consistency analysis
    """
    models_dict = {}
    # Track best model per horizon
    best_model_per_horizon: dict[str, tuple[str, float]] = {}

    for discount_type, horizon_results in result.model_results.items():
        models_dict[discount_type] = []
        for hr in horizon_results:
            test_results_dict = {}
            for bucket, tr in hr.test_results.items():
                test_results_dict[bucket] = {
                    "accuracy": tr.metrics.accuracy,
                    "num_samples": tr.metrics.num_samples,
                    "num_correct": tr.metrics.num_correct,
                    "short_term_accuracy": tr.metrics.short_term_accuracy,
                    "long_term_accuracy": tr.metrics.long_term_accuracy,
                }

            # Track best model for this horizon
            test_acc = (
                hr.test_results[hr.horizon].metrics.accuracy
                if hr.horizon in hr.test_results
                else hr.train_accuracy
            )
            if (
                hr.horizon not in best_model_per_horizon
                or test_acc > best_model_per_horizon[hr.horizon][1]
            ):
                best_model_per_horizon[hr.horizon] = (discount_type, test_acc)

            models_dict[discount_type].append(
                {
                    "train_horizon": hr.horizon,
                    "train_samples": hr.train_samples,
                    "train_loss": hr.train_loss,
                    "train_accuracy": hr.train_accuracy,
                    "theta": hr.trained_params.discount.theta,
                    "test_results": test_results_dict,
                }
            )

    # Serialize verifications
    verification_dict = {}
    if result.train_verification and serialize_verification_fn:
        verification_dict["train"] = serialize_verification_fn(
            result.train_verification
        )
    if result.test_verification and serialize_verification_fn:
        verification_dict["test"] = serialize_verification_fn(result.test_verification)

    # Serialize consistency analysis
    consistency_dict = {}
    if result.train_consistency and serialize_consistency_fn:
        consistency_dict["train"] = serialize_consistency_fn(result.train_consistency)
    if result.test_consistency and serialize_consistency_fn:
        consistency_dict["test"] = serialize_consistency_fn(result.test_consistency)

    # Build best_choice_model dict
    best_choice_model = {
        horizon: {"model": model_name, "accuracy": accuracy}
        for horizon, (model_name, accuracy) in best_model_per_horizon.items()
    }

    return {
        "train_data": result.train_data_path,
        "test_data": result.test_data_path,
        "config": {
            "learning_rate": result.config.learning_rate,
            "num_iterations": result.config.num_iterations,
            "temperature": result.config.temperature,
        },
        "verification": verification_dict if verification_dict else None,
        "consistency": consistency_dict if consistency_dict else None,
        "best_choice_model": best_choice_model,
        "models": models_dict,
    }


def print_summary(result: AnalysisResult, horizon_sort_key=None) -> None:
    """
    Print clear summary of results.

    Args:
        result: AnalysisResult to summarize
        horizon_sort_key: Optional function to sort horizons
    """
    print()
    print("=" * 80)
    print("  CHOICE MODEL ANALYSIS RESULTS (Per Time Horizon Category)")
    print("=" * 80)
    print()
    print(f"  Train data: {result.train_data_path}")
    print(f"  Test data:  {result.test_data_path}")
    print(
        f"  Config:  lr={result.config.learning_rate}, "
        f"iters={result.config.num_iterations}, "
        f"temp={result.config.temperature}"
    )
    print()

    for discount_type, horizon_results in result.model_results.items():
        print("=" * 80)
        print(f"  DISCOUNT TYPE: {discount_type.upper()}")
        print("=" * 80)
        print()

        # Sort results if sort key provided
        if horizon_sort_key:
            horizon_results = sorted(horizon_results, key=lambda x: horizon_sort_key(x.horizon))

        # Header - simpler since we only test on same horizon
        print(
            f"  {'Horizon Category':<16} │ {'N_train':>7} {'θ':>8} │ {'Train Acc':>10} │ {'N_test':>7} {'Test Acc':>10}"
        )
        print("  " + "-" * 17 + "┼" + "-" * 17 + "┼" + "-" * 12 + "┼" + "-" * 19)

        # Results rows
        for hr in horizon_results:
            train_acc = f"{hr.train_accuracy:.1%}"
            theta = f"{hr.trained_params.discount.theta:.4f}"
            horizon_label = (
                hr.horizon if hr.horizon != "no_horizon" else "No Horizon Given"
            )

            # Test results for same horizon
            if hr.horizon in hr.test_results:
                test_result = hr.test_results[hr.horizon]
                test_acc = f"{test_result.metrics.accuracy:.1%}"
                n_test = test_result.metrics.num_samples
            else:
                test_acc = "---"
                n_test = 0

            print(
                f"  {horizon_label:<16} │ {hr.train_samples:>7} {theta:>8} │ {train_acc:>10} │ {n_test:>7} {test_acc:>10}"
            )

        print()

        # Summary statistics
        total_train = sum(hr.train_samples for hr in horizon_results)
        total_test = sum(
            hr.test_results[hr.horizon].metrics.num_samples
            for hr in horizon_results
            if hr.horizon in hr.test_results
        )
        weighted_test_acc = 0.0
        if total_test > 0:
            for hr in horizon_results:
                if hr.horizon in hr.test_results:
                    n = hr.test_results[hr.horizon].metrics.num_samples
                    weighted_test_acc += (
                        hr.test_results[hr.horizon].metrics.accuracy * n
                    )
            weighted_test_acc /= total_test
            print(f"  Totals: {total_train} train samples, {total_test} test samples")
            print(f"  Weighted avg test accuracy: {weighted_test_acc:.1%}")
        print()

    print("=" * 80)
    print("  INTERPRETATION")
    print("=" * 80)
    print()
    print(
        "  Each row shows a separate model trained ONLY on that time horizon category."
    )
    print(
        "  • θ (theta) = fitted discount rate. Higher θ = more impatient (prefer short-term)"
    )
    print("  • Train Acc = accuracy of fitted model on training data (same horizon)")
    print("  • Test Acc = accuracy on held-out test data (same horizon)")
    print("  • 'No Horizon Given' = samples where time_horizon was not specified")
    print()
