"""Data loading and preprocessing for choice analysis."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from src.types import (
    IntertemporalOption,
    PreferencePair,
    PreferenceQuestion,
    RewardValue,
    TimeValue,
    TrainingSample,
)

from .types import (
    AmbiguousSample,
    ExtractedSamples,
    LoadedSamples,
    ProblematicSample,
    SampleWithHorizon,
)

if TYPE_CHECKING:
    pass

# Default paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
PREFERENCE_DATA_DIR = PROJECT_ROOT / "out" / "preference_data"


def find_preference_data_by_query_id(
    query_id: str,
    preference_data_dir: Path = PREFERENCE_DATA_DIR,
) -> Path:
    """Find preference data file by query_id suffix."""
    if not preference_data_dir.exists():
        raise FileNotFoundError(
            f"Preference data directory not found: {preference_data_dir}"
        )

    pattern = f"*_{query_id}.json"
    matches = list(preference_data_dir.glob(pattern))

    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Preference data not found for query_id: {query_id} in {preference_data_dir}"
    )


def format_horizon(time_horizon: list | None) -> str:
    """Format time horizon for display. None -> 'no_horizon'."""
    if time_horizon is None:
        return "no_horizon"
    return f"{time_horizon[0]} {time_horizon[1]}"


def is_ambiguous_sample(
    time_horizon_years: float,
    time_short_years: float,
    time_long_years: float,
) -> bool:
    """
    Check if a sample is ambiguous.

    A sample is ambiguous if time_horizon is BETWEEN short_term_time and long_term_time.
    In this case, we cannot determine a clear expected choice based on horizon alignment.
    """
    return time_short_years <= time_horizon_years <= time_long_years


def filter_problematic_samples(
    samples: list[SampleWithHorizon],
) -> tuple[list[SampleWithHorizon], list[ProblematicSample]]:
    """
    Filter out problematic samples and document them.

    Problematic samples include:
    - time_long <= time_short (long-term option not actually longer)

    Returns:
        (valid_samples, problematic_samples)
    """
    valid = []
    problematic = []

    for i, s in enumerate(samples):
        pair = s.sample.question.pair
        time_short = pair.short_term.time.to_years()
        time_long = pair.long_term.time.to_years()
        time_horizon = None
        if s.sample.question.time_horizon is not None:
            time_horizon = s.sample.question.time_horizon.to_years()

        issues = []

        # Check if long_term is actually longer than short_term
        if time_long <= time_short:
            issues.append(
                {
                    "type": "time_long_not_greater",
                    "description": f"Long-term time ({time_long:.4f} years) <= short-term time ({time_short:.4f} years)",
                }
            )

        if issues:
            problematic.append(
                ProblematicSample(
                    sample_index=i,
                    horizon_key=s.horizon_key,
                    reason="; ".join(issue["type"] for issue in issues),
                    details={
                        "choice": s.sample.chosen,
                        "reward_short": pair.short_term.reward.value,
                        "time_short_years": time_short,
                        "reward_long": pair.long_term.reward.value,
                        "time_long_years": time_long,
                        "time_horizon_years": time_horizon,
                        "issues": issues,
                    },
                )
            )
        else:
            valid.append(s)

    return valid, problematic


def serialize_problematic_samples(problematic: list[ProblematicSample]) -> dict:
    """Serialize problematic samples for JSON output."""
    by_reason: dict[str, list] = defaultdict(list)
    for p in problematic:
        by_reason[p.reason].append(
            {
                "sample_index": p.sample_index,
                "horizon_key": p.horizon_key,
                **p.details,
            }
        )

    return {
        "summary": {
            "total_problematic": len(problematic),
            "by_reason": {
                reason: len(samples) for reason, samples in by_reason.items()
            },
            "interpretation": "These samples were excluded from analysis due to data quality issues.",
        },
        "samples_by_reason": dict(by_reason),
    }


def extract_samples_with_horizons(
    data_path: Path,
    load_preference_data_fn,
    filter_ambiguous: bool = False,
) -> ExtractedSamples:
    """
    Extract training samples with time horizon info from preference data.

    Args:
        data_path: Path to preference data JSON file
        load_preference_data_fn: Function to load preference data (from scripts/common)
        filter_ambiguous: If True, filter out samples where time_horizon is between
            short_term_time and long_term_time. Default False (keep all samples).
    """
    preference_data = load_preference_data_fn(data_path)

    valid_samples = []
    ambiguous_samples = []

    for idx, pref in enumerate(preference_data.preferences):
        if pref.choice == "unknown":
            continue

        # Parse times
        time_short_years = TimeValue(
            pref.preference_pair.short_term.time[0],
            pref.preference_pair.short_term.time[1],
        ).to_years()
        time_long_years = TimeValue(
            pref.preference_pair.long_term.time[0],
            pref.preference_pair.long_term.time[1],
        ).to_years()

        # Handle null time_horizon
        if pref.time_horizon is None:
            time_horizon_tv = None
            horizon_key = "no_horizon"
            time_horizon_years = None
        else:
            time_horizon_tv = TimeValue(pref.time_horizon[0], pref.time_horizon[1])
            horizon_key = format_horizon(pref.time_horizon)
            time_horizon_years = time_horizon_tv.to_years()

        # Check if ambiguous (only for samples with time_horizon)
        is_ambig = time_horizon_years is not None and is_ambiguous_sample(
            time_horizon_years, time_short_years, time_long_years
        )

        if filter_ambiguous and is_ambig:
            ambiguous_samples.append(
                AmbiguousSample(
                    sample_id=idx,
                    time_horizon_years=time_horizon_years,
                    time_short_years=time_short_years,
                    time_long_years=time_long_years,
                    reward_short=pref.preference_pair.short_term.reward,
                    reward_long=pref.preference_pair.long_term.reward,
                    choice=pref.choice,
                    reason=f"time_horizon ({time_horizon_years:.2f}yr) is between short_term_time ({time_short_years:.2f}yr) and long_term_time ({time_long_years:.2f}yr)",
                )
            )
            continue  # Skip ambiguous samples from main analysis

        pair = PreferencePair(
            short_term=IntertemporalOption(
                time=TimeValue(
                    pref.preference_pair.short_term.time[0],
                    pref.preference_pair.short_term.time[1],
                ),
                reward=RewardValue(pref.preference_pair.short_term.reward),
                label=pref.preference_pair.short_term.label,
            ),
            long_term=IntertemporalOption(
                time=TimeValue(
                    pref.preference_pair.long_term.time[0],
                    pref.preference_pair.long_term.time[1],
                ),
                reward=RewardValue(pref.preference_pair.long_term.reward),
                label=pref.preference_pair.long_term.label,
            ),
        )

        question = PreferenceQuestion(
            pair=pair,
            time_horizon=time_horizon_tv,
        )

        training_sample = TrainingSample(question=question, chosen=pref.choice)
        valid_samples.append(
            SampleWithHorizon(sample=training_sample, horizon_key=horizon_key)
        )

    return ExtractedSamples(
        valid_samples=valid_samples, ambiguous_samples=ambiguous_samples
    )


def bucket_samples_by_horizon(
    samples: list[SampleWithHorizon],
) -> dict[str, list[TrainingSample]]:
    """
    Bucket samples by time horizon.

    Returns dict with individual horizon keys (e.g., "6 months", "no_horizon").
    Note: "no_horizon" is treated as its own category for samples with null time_horizon.
    """
    buckets: dict[str, list[TrainingSample]] = defaultdict(list)

    for s in samples:
        buckets[s.horizon_key].append(s.sample)

    return dict(buckets)


def load_samples_from_query_ids(
    query_ids: list[str],
    load_preference_data_fn,
    preference_data_dir: Path = PREFERENCE_DATA_DIR,
) -> LoadedSamples:
    """
    Load and combine samples from multiple query IDs.

    Args:
        query_ids: List of query IDs to load
        load_preference_data_fn: Function to load preference data
        preference_data_dir: Directory containing preference data files
    """
    all_valid = []
    all_ambiguous = []

    for query_id in query_ids:
        path = find_preference_data_by_query_id(query_id, preference_data_dir)
        extracted = extract_samples_with_horizons(path, load_preference_data_fn)
        all_valid.extend(extracted.valid_samples)
        all_ambiguous.extend(extracted.ambiguous_samples)

    return LoadedSamples(valid_samples=all_valid, ambiguous_samples=all_ambiguous)


def class_balanced_train_test_split(
    samples: list[SampleWithHorizon],
    train_ratio: float = 0.7,
    random_seed: int = 42,
) -> tuple[list[SampleWithHorizon], list[SampleWithHorizon]]:
    """
    Split samples into train/test while preserving class balance.

    Stratifies by choice (short_term vs long_term) to ensure both sets
    have similar class distributions.
    """
    rng = random.Random(random_seed)

    # Group by choice
    by_choice: dict[str, list[SampleWithHorizon]] = defaultdict(list)
    for s in samples:
        by_choice[s.sample.chosen].append(s)

    train = []
    test = []

    for choice, choice_samples in by_choice.items():
        rng.shuffle(choice_samples)
        n_train = int(len(choice_samples) * train_ratio)
        train.extend(choice_samples[:n_train])
        test.extend(choice_samples[n_train:])

    # Shuffle final lists
    rng.shuffle(train)
    rng.shuffle(test)

    return train, test
