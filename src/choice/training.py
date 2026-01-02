"""Model training and evaluation for choice analysis."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from src.choice_model.value_function import ValueFunction
from src.types import (
    DiscountFunctionParams,
    TrainingSample,
    ValueFunctionParams,
)

from .types import (
    ChoiceModelSpec,
    EvaluationMetrics,
    HorizonModelResult,
    SampleWithHorizon,
    TestResult,
)

if TYPE_CHECKING:
    from .config import AnalysisConfig


def train_model(
    spec: ChoiceModelSpec,
    train_samples: list[TrainingSample],
    learning_rate: float = 0.01,
    num_iterations: int = 100,
    temperature: float = 0.0,
) -> tuple[ValueFunction, ValueFunctionParams, float, float]:
    """
    Train a choice model and return the trained model.

    Args:
        spec: Model specification (utility type, discount type, alpha)
        train_samples: List of training samples
        learning_rate: Learning rate for optimization
        num_iterations: Number of training iterations
        temperature: Temperature for softmax (0 = deterministic)

    Returns:
        Tuple of (trained_model, params, loss, accuracy)
    """
    # Initialize value function
    discount_params = DiscountFunctionParams(
        discount_type=spec.discount_type,
        theta=0.1,  # Initial value
        beta=0.7,  # For quasi-hyperbolic
        delta=0.99,  # For quasi-hyperbolic
    )
    value_params = ValueFunctionParams(
        utility_type=spec.utility_type,
        alpha=spec.alpha,  # For power utility
        discount=discount_params,
    )

    vf = ValueFunction.from_params(value_params)

    # Train
    result = vf.train(
        train_samples,
        learning_rate=learning_rate,
        num_iterations=num_iterations,
        temperature=temperature,
    )

    return vf, result.params, result.loss, result.accuracy


def test_model(
    vf: ValueFunction,
    test_samples: list[SampleWithHorizon],
    bucket_name: str,
) -> TestResult:
    """
    Test a model on a set of samples.

    Args:
        vf: Trained ValueFunction model
        test_samples: List of test samples
        bucket_name: Name for this test bucket

    Returns:
        TestResult with metrics and predictions
    """
    predictions = []
    correct = 0
    short_correct, short_total = 0, 0
    long_correct, long_total = 0, 0
    per_horizon: dict = {}

    for s in test_samples:
        actual = s.sample.chosen
        pair = s.sample.question.pair

        predicted = vf.predict(pair)
        predictions.append((predicted, actual))

        is_correct = predicted == actual
        if is_correct:
            correct += 1

        if actual == "short_term":
            short_total += 1
            if is_correct:
                short_correct += 1
        else:
            long_total += 1
            if is_correct:
                long_correct += 1

        horizon = s.horizon_key
        if horizon not in per_horizon:
            per_horizon[horizon] = {"correct": 0, "total": 0}
        per_horizon[horizon]["total"] += 1
        if is_correct:
            per_horizon[horizon]["correct"] += 1

    total = len(predictions)
    per_horizon_accuracy = {
        h: c["correct"] / c["total"] if c["total"] > 0 else 0.0
        for h, c in per_horizon.items()
    }

    metrics = EvaluationMetrics(
        accuracy=correct / total if total > 0 else 0.0,
        num_samples=total,
        num_correct=correct,
        short_term_accuracy=short_correct / short_total if short_total > 0 else 0.0,
        long_term_accuracy=long_correct / long_total if long_total > 0 else 0.0,
        n_short=short_total,
        n_long=long_total,
        per_horizon_accuracy=per_horizon_accuracy,
    )

    return TestResult(
        bucket_name=bucket_name,
        metrics=metrics,
        predictions=predictions,
    )


def train_and_test_by_horizon(
    spec: ChoiceModelSpec,
    train_buckets: dict[str, list[TrainingSample]],
    test_samples: list[SampleWithHorizon],
    learning_rate: float = 0.01,
    num_iterations: int = 100,
    temperature: float = 0.0,
    horizon_sort_key=None,
) -> list[HorizonModelResult]:
    """
    Train separate models for each horizon category and test on same category.

    Each time horizon category (including "no_horizon" for null time_horizon)
    is treated independently - train on category's training data, test on
    category's test data.

    Args:
        spec: Model specification
        train_buckets: Dict mapping horizon_key to training samples
        test_samples: List of test samples (will be bucketed internally)
        learning_rate: Learning rate for optimization
        num_iterations: Number of training iterations
        temperature: Temperature for softmax
        horizon_sort_key: Optional function to sort horizon keys

    Returns:
        List of HorizonModelResult for each horizon
    """
    results = []

    # Get horizon keys sorted by time duration (no_horizon last)
    if horizon_sort_key:
        horizon_keys = sorted(train_buckets.keys(), key=horizon_sort_key)
    else:
        horizon_keys = sorted(train_buckets.keys())

    # Bucket test samples by horizon
    test_by_horizon: dict[str, list[SampleWithHorizon]] = defaultdict(list)
    for s in test_samples:
        test_by_horizon[s.horizon_key].append(s)

    for horizon in horizon_keys:
        train_samples = train_buckets[horizon]
        if len(train_samples) == 0:
            continue

        # Train model on this horizon's training data
        vf, params, loss, accuracy = train_model(
            spec,
            train_samples,
            learning_rate=learning_rate,
            num_iterations=num_iterations,
            temperature=temperature,
        )

        # Test ONLY on same horizon's test data
        test_results = {}
        if horizon in test_by_horizon and len(test_by_horizon[horizon]) > 0:
            test_results[horizon] = test_model(vf, test_by_horizon[horizon], horizon)

        results.append(
            HorizonModelResult(
                horizon=horizon,
                spec=spec,
                trained_params=params,
                train_loss=loss,
                train_accuracy=accuracy,
                train_samples=len(train_samples),
                test_results=test_results,
            )
        )

    return results
