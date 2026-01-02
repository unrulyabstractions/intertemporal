"""Consistency analysis for choice data."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from .types import SampleWithHorizon

if TYPE_CHECKING:
    from typing import Callable


def extract_constraints(
    samples: list[SampleWithHorizon],
    theta_constraint_from_sample: Callable,
) -> dict[str, list]:
    """
    Extract θ constraints from samples, grouped by horizon.

    Args:
        samples: List of samples with horizon info
        theta_constraint_from_sample: Function to create ThetaConstraint from sample data
    """
    constraints_by_horizon: dict[str, list] = defaultdict(list)

    for s in samples:
        pair = s.sample.question.pair
        choice = s.sample.chosen

        # Normalize choice
        if choice in ("short_term", "a", pair.short_term.label):
            choice_normalized = "short_term"
        else:
            choice_normalized = "long_term"

        # Get time_horizon in years (None if not set)
        time_horizon_years = None
        if s.sample.question.time_horizon is not None:
            time_horizon_years = s.sample.question.time_horizon.to_years()

        constraint = theta_constraint_from_sample(
            reward_short=pair.short_term.reward.value,
            time_short=pair.short_term.time.to_years(),
            reward_long=pair.long_term.reward.value,
            time_long=pair.long_term.time.to_years(),
            choice=choice_normalized,
            horizon_key=s.horizon_key,
            time_horizon=time_horizon_years,
        )

        constraints_by_horizon[s.horizon_key].append(constraint)

    return dict(constraints_by_horizon)


def analyze_consistency(
    samples: list[SampleWithHorizon],
    consistency_analysis_class,
    theta_constraint_from_sample: Callable,
) -> dict:
    """
    Analyze consistency of choices for each horizon bucket.

    Args:
        samples: List of samples with horizon info
        consistency_analysis_class: ConsistencyAnalysis class with analyze() method
        theta_constraint_from_sample: Function to create ThetaConstraint
    """
    constraints_by_horizon = extract_constraints(samples, theta_constraint_from_sample)

    analyses = {}
    for horizon_key, constraints in constraints_by_horizon.items():
        analyses[horizon_key] = consistency_analysis_class.analyze(
            constraints, horizon_key
        )

    return analyses


def filter_consistent_samples(
    samples: list[SampleWithHorizon],
    consistency: dict,
    theta_constraint_from_sample: Callable,
) -> list[SampleWithHorizon]:
    """
    Filter out samples that conflict with the best theta for their horizon.

    A sample is consistent if its implied constraint is satisfied by the
    best theta found for that horizon category.

    Args:
        samples: List of samples to filter
        consistency: Dict mapping horizon_key to ConsistencyAnalysis
        theta_constraint_from_sample: Function to create ThetaConstraint
    """
    consistent_samples = []

    for s in samples:
        horizon_key = s.horizon_key
        if horizon_key not in consistency:
            # No analysis for this horizon, keep the sample
            consistent_samples.append(s)
            continue

        analysis = consistency[horizon_key]
        if not analysis.constraints:
            consistent_samples.append(s)
            continue

        # Recreate the constraint for this sample
        pair = s.sample.question.pair
        choice = s.sample.chosen
        if choice in ("short_term", "a", pair.short_term.label):
            choice_normalized = "short_term"
        else:
            choice_normalized = "long_term"

        constraint = theta_constraint_from_sample(
            reward_short=pair.short_term.reward.value,
            time_short=pair.short_term.time.to_years(),
            reward_long=pair.long_term.reward.value,
            time_long=pair.long_term.time.to_years(),
            choice=choice_normalized,
            horizon_key=horizon_key,
        )

        # Check if this sample is consistent with the best theta
        if analysis.is_sample_consistent(constraint):
            consistent_samples.append(s)

    return consistent_samples
