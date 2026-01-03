#!/usr/bin/env python
"""
Experiment with activation steering using trained probes.

Uses probe directions to steer model behavior toward short-term or long-term choices.
Tests steering on dataset samples and measures the minimum steering magnitude
needed to flip choices.

Creates heatmaps showing:
- Min steering magnitude to flip choice (per layer/position)
- Separate plots for short->long and long->short flips
- Gray cells for "never flipped", black "FAIL" for degeneration

Config file options:
    {
      "probes_id": "7c7dfcf3d544239e7badadfd89859c38",  // OR use contrastive_id
      "contrastive_id": "abc123...",  // Optional: CAA vectors
      "query_id": "7af8b316feb64ef4dd8ac94497fedf5b",  // References preference_data
      "probe_types": ["choice"],
      "starting_steering_strength": 500.0,  // Max magnitude (positive=long, negative=short)
      "max_samples": 10,
      "sample_indices": [123, 456, 789],  // Optional: specific sample IDs
      "specific_probes": [  // Optional: test only these trained probes
        {"layer": 10, "token_pos": 5},
        {"layer": -1, "token_pos": "I select:"}  // -1 = last layer, text matching
      ],
      "specific_contrastive": [  // Optional: test only these contrastive vectors
        {"layer": 15, "token_pos": 8}
      ],
      "only_after_horizon": true,
      "only_before_choice": true  // Only test probes before choice selection
    }

Outputs:
    - results/steering_*.json: Full results with all sample data
    - debug_steering.json: Quick summary showing if steering worked, failures, etc.
    - viz/steering_short_to_long.png: Min magnitude heatmap
    - viz/steering_long_to_short.png: Min magnitude heatmap

Usage:
    python scripts/try_steering.py
    python scripts/try_steering.py --config default_steering
    python scripts/try_steering.py --all-probes
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np

# Add project root and scripts to path
PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from src.common.io import ensure_dir, get_timestamp, load_json, save_json
from src.common.schema_utils import SchemaClass
from src.common.schemas import FormattingConfig
from src.model_runner import ModelRunner
from src.probes import (
    LoadedProbe,
    ProbeType,
    load_probes_from_dir,
)
from src.steering import SteeringConfig as SteeringDirectionConfig, SteeringOption, TokenPositionTarget

from common import (
    build_prompt_from_question,
    clear_memory,
    determine_choice,
    find_dataset_by_id,
    find_preference_data_by_query_id,
    load_dataset_output,
    load_preference_data,
    parse_label_from_response,
    DatasetOutput,
    PreferenceDataOutput,
    QuestionOutput,
)


# =============================================================================
# Steering Configuration Schema
# =============================================================================


@dataclass
class ProbeSpec:
    """Specification for a specific probe to test.

    Attributes:
        layer: Layer index (can be negative, e.g., -1 for last layer)
        token_pos: Token position - either int (actual position in sequence) or str for text matching
    """

    layer: int
    token_pos: Union[int, str]

    def matches(
        self,
        probe_layer: int,
        probe_token_position: Optional[int],
        probe_token_position_idx: int,
        n_layers: int,
        token_info: Optional[dict] = None,
    ) -> bool:
        """Check if this spec matches a probe.

        Args:
            probe_layer: The probe's layer
            probe_token_position: The probe's actual token position in sequence (may be None for old probes)
            probe_token_position_idx: The probe's token position index (relative, for fallback)
            n_layers: Total number of layers in model
            token_info: Optional token info for text matching
        """
        # Resolve negative layer indices
        target_layer = self.layer if self.layer >= 0 else n_layers + self.layer
        if probe_layer != target_layer:
            return False

        # Match token position
        if isinstance(self.token_pos, int):
            # Match against actual token_position if available, else fall back to idx
            if probe_token_position is not None:
                return probe_token_position == self.token_pos
            else:
                # Fallback for old probes without token_position
                return probe_token_position_idx == self.token_pos
        else:
            # String matching - need token_info to resolve
            if token_info is None:
                return False
            specs = token_info.get("specs", [])
            if probe_token_position_idx < len(specs):
                spec = specs[probe_token_position_idx]
                # TokenPositionSpec is a dataclass with a 'spec' field containing the dict
                # Check if the text matches
                if hasattr(spec, "spec") and isinstance(spec.spec, dict):
                    if spec.spec.get("text") == self.token_pos:
                        return True
                elif isinstance(spec, dict) and spec.get("text") == self.token_pos:
                    return True
            return False


@dataclass
class TrySteeringConfigSchema(SchemaClass):
    """Schema for steering experiment config - used for deterministic folder IDs."""

    query_id: str  # References preference_data for baselines
    starting_steering_strength: (
        float  # Max magnitude to test (used for both +/- directions)
    )
    max_samples: int
    only_after_horizon: bool
    only_before_choice: bool
    # Steering source - at least one must be set
    probes_id: Optional[str] = None  # Probe config ID for trained probes
    contrastive_id: Optional[str] = None  # CAA config ID for contrastive vectors
    probe_types: tuple[str, ...] = ("choice",)
    # specific_probes/specific_contrastive affect identity if set
    specific_probes: Optional[tuple[tuple[int, Union[int, str]], ...]] = None
    specific_contrastive: Optional[tuple[tuple[int, Union[int, str]], ...]] = None
    specific_positions: Optional[tuple[Union[int, dict], ...]] = None  # Token positions for steering (int or spec dict)
    batch_size: int = 8
    fractions: Optional[tuple[float, ...]] = None  # Exact multipliers (neg=short, pos=long)
    # Baseline filtering (affects which samples are used)
    baseline_choice: Optional[str] = None  # "short" or "long" - filter by baseline choice
    baseline_horizon: Optional[tuple[tuple, tuple]] = None  # [[min_val, unit], [max_val, unit]] inclusive
    # Note: sample_indices not included in schema - it's for reproducibility, not identity


@dataclass
class TrySteeringConfig:
    """Configuration for steering experiment.

    Attributes:
        query_id: References preference_data file to use as baseline.
                 Replaces dataset_id - the dataset is derived from preference_data.
        probes_id: Probe config ID for trained probes (optional if contrastive_id set).
        contrastive_id: CAA config ID for contrastive vectors (optional if probes_id set).
        starting_steering_strength: Max magnitude to test. Positive steering pushes
                 toward long_term, negative toward short_term.
        max_samples: Maximum samples to use (0 = all). Ignored if sample_indices provided.
        sample_indices: If specified, use these exact sample IDs from the preference data
                       instead of random sampling. Takes precedence over max_samples.
        specific_probes: If specified, only test these specific trained probes.
                        Each entry is {"layer": int, "token_pos": int|str}.
                        Layer can be negative (-1 = last layer).
                        token_pos can be int (position index) or str (text to match).
        specific_contrastive: If specified, only test these specific contrastive vectors.
                             Same format as specific_probes.
        only_after_horizon: If True, only test probes after time horizon position.
        only_before_choice: If True, only test probes before choice selection position.
        batch_size: Batch size for processing multiple prompts.
        controls: Number of random direction control vectors to run per sample/magnitude.
                 False/0 = disabled, True/1 = one control, N = N controls.
        fractions: Exact multipliers of starting_steering_strength to test.
                  Use negative for short_term direction, positive for long_term.
                  Default is [-1.0, -0.4, -0.1, 0.1, 0.4, 1.0] (symmetric).
        specific_positions: If set, apply steering only at these token positions
                           (uses APPLY_TO_TOKEN_POSITION instead of APPLY_TO_ALL).
        baseline_choice: If set, only use samples with this baseline choice ("short" or "long").
        baseline_horizon: If set, only use samples with time horizon in this range.
                         Format: [[min_val, unit], [max_val, unit]] (inclusive).
                         Example: [[3, "months"], [1, "years"]] for 3 months to 1 year.
    """

    query_id: str  # References preference_data for baselines
    probes_id: Optional[str] = None  # Probe config ID
    contrastive_id: Optional[str] = None  # CAA config ID
    probe_types: list[str] = field(default_factory=lambda: ["choice"])
    starting_steering_strength: float = (
        500.0  # Max magnitude (applies to both +/- directions)
    )
    max_samples: int = 0  # 0 = use all samples
    only_after_horizon: bool = True  # Only test probes after time horizon
    only_before_choice: bool = True  # Only test probes before choice selection
    specific_probes: Optional[list[ProbeSpec]] = None  # Specific trained probes to test
    specific_contrastive: Optional[list[ProbeSpec]] = None  # Specific contrastive vectors to test
    sample_indices: Optional[list[int]] = None  # Specific sample IDs to use
    batch_size: int = 8  # Batch size for processing multiple prompts
    controls: Union[bool, int] = False  # Number of random control vectors (False/0=off, True/1=one, N=N)
    fractions: Optional[list[float]] = None  # Exact multipliers (neg=short, pos=long)
    specific_positions: Optional[list[Union[int, dict]]] = None  # Token positions (int or spec dict)
    baseline_choice: Optional[str] = None  # "short" or "long" - filter samples by baseline choice
    baseline_horizon: Optional[list[list]] = None  # [[min_val, unit], [max_val, unit]] inclusive range

    def get_schema(self) -> TrySteeringConfigSchema:
        """Convert to schema for ID generation."""
        # Convert specific_probes to hashable tuple format
        specific_probes_tuple = None
        if self.specific_probes:
            specific_probes_tuple = tuple(
                (sp.layer, sp.token_pos) for sp in self.specific_probes
            )
        # Convert specific_contrastive to hashable tuple format
        specific_contrastive_tuple = None
        if self.specific_contrastive:
            specific_contrastive_tuple = tuple(
                (sp.layer, sp.token_pos) for sp in self.specific_contrastive
            )
        # Convert fractions to tuple if present
        fractions_tuple = tuple(self.fractions) if self.fractions else None
        # Convert specific_positions to hashable tuple if present
        specific_positions_tuple = None
        if self.specific_positions:
            # Convert dicts to hashable tuples
            hashable_positions = []
            for pos in self.specific_positions:
                if isinstance(pos, dict):
                    # Convert dict to sorted tuple of items for hashing
                    hashable_positions.append(tuple(sorted(pos.items())))
                else:
                    hashable_positions.append(pos)
            specific_positions_tuple = tuple(hashable_positions)
        # Convert baseline_horizon to hashable tuple if present
        baseline_horizon_tuple = None
        if self.baseline_horizon:
            baseline_horizon_tuple = (
                tuple(self.baseline_horizon[0]),
                tuple(self.baseline_horizon[1]),
            )
        return TrySteeringConfigSchema(
            query_id=self.query_id,
            probes_id=self.probes_id,
            contrastive_id=self.contrastive_id,
            probe_types=tuple(sorted(self.probe_types)),
            starting_steering_strength=self.starting_steering_strength,
            max_samples=self.max_samples,
            only_after_horizon=self.only_after_horizon,
            only_before_choice=self.only_before_choice,
            specific_probes=specific_probes_tuple,
            specific_contrastive=specific_contrastive_tuple,
            specific_positions=specific_positions_tuple,
            batch_size=self.batch_size,
            fractions=fractions_tuple,
            baseline_choice=self.baseline_choice,
            baseline_horizon=baseline_horizon_tuple,
        )

    def get_id(self) -> str:
        """Get deterministic config ID."""
        return self.get_schema().get_id()


def load_steering_config(path: Path) -> TrySteeringConfig:
    """Load steering config from JSON file."""
    data = load_json(path)

    # Parse specific_probes if present
    specific_probes = None
    if "specific_probes" in data:
        specific_probes = [
            ProbeSpec(layer=sp["layer"], token_pos=sp["token_pos"])
            for sp in data["specific_probes"]
        ]

    # Parse specific_contrastive if present
    specific_contrastive = None
    if "specific_contrastive" in data:
        specific_contrastive = [
            ProbeSpec(layer=sp["layer"], token_pos=sp["token_pos"])
            for sp in data["specific_contrastive"]
        ]

    # Support both old "probe_config_id" and new "probes_id"
    probes_id = data.get("probes_id") or data.get("probe_config_id")
    contrastive_id = data.get("contrastive_id")

    if not probes_id and not contrastive_id:
        raise ValueError("Config must specify at least one of: probes_id, contrastive_id")

    # Get query_id - can be explicit or derived from contrastive metadata
    query_id = data.get("query_id")
    if not query_id and contrastive_id:
        # Derive from contrastive vector metadata
        contrastive_results_path = (
            PROJECT_ROOT / "out" / "contrastive" / contrastive_id / "results" / "caa_results.json"
        )
        if contrastive_results_path.exists():
            contrastive_meta = load_json(contrastive_results_path)
            query_ids = contrastive_meta.get("query_ids", [])
            if query_ids:
                query_id = query_ids[0]  # Use first query_id
    if not query_id:
        raise ValueError("Config must specify query_id, or contrastive_id must have valid metadata")

    return TrySteeringConfig(
        query_id=query_id,
        probes_id=probes_id,
        contrastive_id=contrastive_id,
        probe_types=data.get("probe_types", ["choice"]),
        starting_steering_strength=data.get("starting_steering_strength", 500.0),
        max_samples=data.get("max_samples", 0),  # 0 = use all samples
        only_after_horizon=data.get("only_after_horizon", True),
        only_before_choice=data.get("only_before_choice", True),
        specific_probes=specific_probes,
        specific_contrastive=specific_contrastive,
        sample_indices=data.get("sample_indices"),  # None = random sampling
        batch_size=data.get("batch_size", 8),
        controls=data.get("controls", False),
        fractions=data.get("fractions"),  # None = use default
        specific_positions=data.get("specific_positions"),  # None = apply to all
        baseline_choice=data.get("baseline_choice"),  # None = any choice
        baseline_horizon=data.get("baseline_horizon"),  # None = any horizon
    )


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class SampleSteeringResult:
    """Result from steering a single sample."""

    sample_id: int
    time_horizon: Optional[list]
    baseline_choice: str  # "short_term", "long_term", or "unknown"
    # Maps strength -> choice ("short_term", "long_term", "unknown", or "degenerate")
    steered_choices: dict[float, str]
    # Min magnitude that caused flip (None if never flipped)
    min_flip_magnitude_to_long: Optional[float] = None  # short->long
    min_flip_magnitude_to_short: Optional[float] = None  # long->short
    # Maps strength -> response text for unknown/degenerate results (for debugging)
    failed_responses: Optional[dict[float, str]] = None
    # Reward values for analysis
    short_reward: Optional[float] = None
    long_reward: Optional[float] = None


@dataclass
class ProbeSteeringResult:
    """Result from steering with a specific probe."""

    probe_id: str
    probe_type: str
    layer: int
    token_position_idx: int
    after_horizon: bool
    samples: list[SampleSteeringResult]


@dataclass
class SteeringExperimentOutput:
    """Full output from steering experiment."""

    steering_id: str
    query_id: str
    dataset_id: str  # Derived from preference_data
    model_name: str
    steering_strengths: list[float]
    n_samples: int
    sample_indices: list[int]  # Actual sample IDs used (for reproducibility)
    probe_results: list[ProbeSteeringResult]
    timestamp: str
    probes_id: Optional[str] = None  # Probe config ID (if using probes)
    contrastive_id: Optional[str] = None  # CAA config ID (if using contrastive vectors)


# =============================================================================
# Preference Data Loading
# =============================================================================


def time_to_months(time_value: list) -> float:
    """Convert a time value like [3, "months"] or [1, "years"] to months.

    Args:
        time_value: [value, unit] where unit is "days", "weeks", "months", "years"

    Returns:
        Time in months (as float for comparison)
    """
    value, unit = time_value[0], time_value[1].lower()
    # Normalize plural forms
    if unit.endswith("s"):
        unit = unit[:-1]

    conversion = {
        "day": 1 / 30,  # Approximate
        "week": 7 / 30,  # Approximate
        "month": 1,
        "year": 12,
    }
    if unit not in conversion:
        raise ValueError(f"Unknown time unit: {unit}")
    return value * conversion[unit]


def matches_horizon_filter(
    long_term_time: list,
    baseline_horizon: list[list],
) -> bool:
    """Check if a sample's long_term time falls within the baseline_horizon range.

    Args:
        long_term_time: [value, unit] for the long-term option
        baseline_horizon: [[min_val, unit], [max_val, unit]] inclusive range

    Returns:
        True if within range (inclusive)
    """
    sample_months = time_to_months(long_term_time)
    min_months = time_to_months(baseline_horizon[0])
    max_months = time_to_months(baseline_horizon[1])
    return min_months <= sample_months <= max_months


@dataclass
class SampleBaseline:
    """Baseline data for a single sample from preference_data."""

    sample_id: int
    prompt: str
    baseline_choice: str  # "short_term", "long_term", or "unknown"
    time_horizon: Optional[list]
    short_label: str
    long_label: str
    short_value: str  # For fallback parsing (reward formatted with commas)
    long_value: str


@dataclass
class PreferenceDataInfo:
    """Loaded preference data with baselines and metadata."""

    preference_data: PreferenceDataOutput
    baselines: dict[int, SampleBaseline]  # sample_id -> baseline
    dataset_id: str
    model_name: str
    query_id: str
    formatting_config: FormattingConfig


def load_preference_data_for_steering(
    query_id: str,
    max_samples: int = 0,
    sample_indices: Optional[list[int]] = None,
    baseline_choice: Optional[str] = None,
    baseline_horizon: Optional[list[list]] = None,
) -> PreferenceDataInfo:
    """
    Load preference data by query_id and build baseline map.

    Args:
        query_id: Query ID to find preference_data file
        max_samples: Max samples to use (0 = all). Ignored if sample_indices provided.
        sample_indices: Specific sample IDs to use
        baseline_choice: Optional filter - "short" or "long" to only include samples
            where the baseline choice was short_term or long_term
        baseline_horizon: Optional filter - [[min_val, unit], [max_val, unit]] to only
            include samples where long_term time horizon is within range (inclusive)

    Returns:
        PreferenceDataInfo with baselines and metadata
    """
    import random

    # Find preference data file
    pref_path = find_preference_data_by_query_id(query_id)
    if pref_path is None:
        raise FileNotFoundError(f"Preference data with query_id '{query_id}' not found")

    # Load preference data
    pref_data = load_preference_data(pref_path)

    # Get dataset_id from metadata
    dataset_id = pref_data.metadata.dataset_id
    model_name = pref_data.metadata.model

    # Load dataset to get question texts for building prompts
    dataset_path = find_dataset_by_id(dataset_id)
    if dataset_path is None:
        raise FileNotFoundError(f"Dataset with id '{dataset_id}' not found")
    dataset = load_dataset_output(dataset_path)

    # Build question map: sample_id -> QuestionOutput
    question_map: dict[int, QuestionOutput] = {
        q.sample_id: q for q in dataset.questions
    }

    # Load formatting config for building prompts
    formatting_config_name = pref_data.metadata.query_config.get("formatting", {}).get(
        "name", "default_formatting"
    )
    formatting_config_path = (
        PROJECT_ROOT
        / "scripts"
        / "configs"
        / "formatting"
        / f"{formatting_config_name}.json"
    )
    from src.dataset_generator import DatasetGenerator

    formatting_config = DatasetGenerator.load_formatting_config(formatting_config_path)

    # Build baseline map from all preferences
    all_baselines: dict[int, SampleBaseline] = {}
    for pref in pref_data.preferences:
        pair = pref.preference_pair

        # Get question from dataset and build prompt
        question = question_map.get(pref.sample_id)
        if question is None:
            print(f"Warning: sample_id {pref.sample_id} not found in dataset, skipping")
            continue

        prompt = build_prompt_from_question(
            question, formatting_config, model_name=model_name
        )

        # Apply baseline_choice filter
        if baseline_choice is not None:
            expected_choice = f"{baseline_choice}_term"  # "short" -> "short_term"
            if pref.choice != expected_choice:
                continue

        # Apply baseline_horizon filter (on sample's time_horizon)
        if baseline_horizon is not None:
            if pref.time_horizon is None:
                continue  # Skip samples without time_horizon
            if not matches_horizon_filter(pref.time_horizon, baseline_horizon):
                continue

        all_baselines[pref.sample_id] = SampleBaseline(
            sample_id=pref.sample_id,
            prompt=prompt,
            baseline_choice=pref.choice,
            time_horizon=pref.time_horizon,
            short_label=pair.short_term.label,
            long_label=pair.long_term.label,
            short_value=f"{pair.short_term.reward:,.0f}",
            long_value=f"{pair.long_term.reward:,.0f}",
        )

    # Report filtering results
    total_prefs = len(pref_data.preferences)
    if baseline_choice or baseline_horizon:
        filters_desc = []
        if baseline_choice:
            filters_desc.append(f"choice={baseline_choice}")
        if baseline_horizon:
            min_h, max_h = baseline_horizon
            filters_desc.append(f"horizon=[{min_h[0]} {min_h[1]}, {max_h[0]} {max_h[1]}]")
        print(f"  Filtered: {len(all_baselines)}/{total_prefs} samples match {', '.join(filters_desc)}")

    # Select samples
    if sample_indices is not None:
        # Use specific sample IDs
        selected_baselines = {}
        missing = []
        for sid in sample_indices:
            if sid in all_baselines:
                selected_baselines[sid] = all_baselines[sid]
            else:
                missing.append(sid)
        if missing:
            print(f"Warning: sample IDs not found in preference_data: {missing}")
        print(f"Using {len(selected_baselines)} specified samples")
    else:
        # Random sampling (max_samples=0 means use all)
        all_ids = sorted(all_baselines.keys())  # Sort for consistent ordering
        if max_samples > 0 and max_samples < len(all_ids):
            random.seed(42)
            selected_ids = sorted(random.sample(all_ids, max_samples))
            print(f"Sampled {max_samples} samples from {len(all_ids)}")
        else:
            selected_ids = all_ids
            print(f"Using all {len(all_ids)} samples")
        selected_baselines = {sid: all_baselines[sid] for sid in selected_ids}

    return PreferenceDataInfo(
        preference_data=pref_data,
        baselines=selected_baselines,
        dataset_id=dataset_id,
        model_name=model_name,
        query_id=query_id,
        formatting_config=formatting_config,
    )


def validate_model_matches(pref_model: str, probe_model: str) -> None:
    """
    Validate that preference_data model matches probe model.

    Raises ValueError if models don't match.
    """
    # Normalize model names (handle both full and short names)
    pref_short = pref_model.split("/")[-1]
    probe_short = probe_model.split("/")[-1]

    if pref_short != probe_short:
        raise ValueError(
            f"Model mismatch: preference_data uses '{pref_model}' "
            f"but probes use '{probe_model}'"
        )


def find_dataset_by_id(dataset_id: str) -> Path:
    """Find dataset file by ID."""
    datasets_dir = PROJECT_ROOT / "out" / "datasets"
    pattern = f"*_{dataset_id}.json"
    matches = list(datasets_dir.glob(pattern))
    if matches:
        return matches[0]
    # Try without underscore
    pattern = f"*{dataset_id}*.json"
    matches = list(datasets_dir.glob(pattern))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Dataset with ID {dataset_id} not found in {datasets_dir}")


def load_dataset_samples(
    dataset_id: str,
    n_samples: int = 10,
    sample_indices: Optional[list[int]] = None,
) -> tuple[DatasetOutput, list[QuestionOutput], list[int]]:
    """Load dataset and prepare samples for steering.

    Args:
        dataset_id: Dataset ID to load
        n_samples: Number of random samples (ignored if sample_indices provided)
        sample_indices: Specific sample IDs to use (takes precedence over n_samples)

    Returns:
        Tuple of (dataset, questions, sample_ids_used)
    """
    import random

    dataset_path = find_dataset_by_id(dataset_id)
    dataset = load_dataset_output(dataset_path)

    questions = list(dataset.questions)
    questions_by_id = {q.sample_id: q for q in questions}

    if sample_indices is not None:
        # Use specific sample IDs
        selected = []
        missing = []
        for sid in sample_indices:
            if sid in questions_by_id:
                selected.append(questions_by_id[sid])
            else:
                missing.append(sid)
        if missing:
            print(f"Warning: sample IDs not found in dataset: {missing}")
        print(
            f"Using {len(selected)} specified samples: {[q.sample_id for q in selected]}"
        )
        return dataset, selected, [q.sample_id for q in selected]

    # Random sampling
    if n_samples < len(questions):
        random.seed(42)  # Reproducible sampling
        questions = random.sample(questions, n_samples)
        print(f"Sampled {n_samples} questions from {len(dataset.questions)}")

    sample_ids = [q.sample_id for q in questions]
    return dataset, questions, sample_ids


# =============================================================================
# Model Loading
# =============================================================================


def load_model_for_steering(model_name: str) -> ModelRunner:
    """Load a model for steering experiments using ModelRunner."""
    return ModelRunner(model_name)


# =============================================================================
# Choice Parsing
# =============================================================================
#
# Uses shared utilities from common.utils:
#   - parse_label_from_response(): Regex-based label extraction
#   - determine_choice(): Maps label to "short_term" or "long_term"
#
# See scripts/common/utils.py for implementation details.


def parse_choice_from_response(
    response: str,
    short_label: str,
    long_label: str,
    choice_prefix: str = "I select:",
    debug: bool = False,
    short_value: Optional[str] = None,
    long_value: Optional[str] = None,
) -> str:
    """
    Parse choice from model response.

    Uses shared parsing from common.utils, with additional degeneration detection
    for steering experiments where extreme steering can produce garbage output.

    Args:
        response: Model generated text
        short_label: Label for short-term option (e.g., "[A]")
        long_label: Label for long-term option (e.g., "[B]")
        choice_prefix: Expected choice prefix (e.g., "I select:")
        debug: If True, print responses that fail to parse
        short_value: Optional value text for short option (for fallback parsing)
        long_value: Optional value text for long option (for fallback parsing)

    Returns:
        "short_term", "long_term", "unknown", or "degenerate"
    """
    import re

    # Check for degeneration (garbage output from extreme steering)
    if not response or len(response.strip()) < 5:
        if debug:
            print(f"      [DEGENERATE] Response too short: {repr(response)}")
        return "degenerate"

    # Check for repetitive/garbage patterns
    if len(set(response)) < 5:  # Very few unique chars
        if debug:
            print(f"      [DEGENERATE] Repetitive: {repr(response[:100])}")
        return "degenerate"

    try:
        chosen_label = parse_label_from_response(
            response, [short_label, long_label], choice_prefix, ""
        )
        choice = determine_choice(chosen_label, short_label, long_label)
        if choice in ("short_term", "long_term"):
            return choice

        # Fallback: try to detect choice by value when label is missing
        # This handles cases where steering causes model to drop the label format
        if short_value and long_value:
            # Extract significant numbers from values (e.g., "4,000" from "4,000 housing units")
            short_numbers = [
                n for n in re.findall(r"[\d,]+", short_value) if len(n) >= 3
            ]
            long_numbers = [n for n in re.findall(r"[\d,]+", long_value) if len(n) >= 3]

            short_found = any(num in response for num in short_numbers)
            long_found = any(num in response for num in long_numbers)

            if short_found and not long_found:
                if debug:
                    print(
                        f"      [FALLBACK] Detected short_term by value: {short_numbers}"
                    )
                return "short_term"
            elif long_found and not short_found:
                if debug:
                    print(
                        f"      [FALLBACK] Detected long_term by value: {long_numbers}"
                    )
                return "long_term"

        if debug:
            print(f"      [UNKNOWN] Labels: {short_label}, {long_label}")
            print(f"      [UNKNOWN] Parsed label: {repr(chosen_label)}")
            print(f"      [UNKNOWN] Response: {repr(response[:200])}")
        return "unknown"
    except Exception as e:
        if debug:
            print(f"      [EXCEPTION] {e}")
            print(f"      [EXCEPTION] Response: {repr(response[:200])}")
        return "unknown"


# =============================================================================
# Prompt Building - Uses shared module from common.prompt_builder
# =============================================================================
#
# The build_prompt_from_question function is now imported from common.prompt_builder
# This consolidates the prompt building logic that was previously duplicated
# across try_steering.py and query_llm.py.
#
# See scripts/common/prompt_builder.py for implementation details.


# =============================================================================
# Steering Experiment
# =============================================================================


def generate_steering_strengths(
    starting_strength: float,
    fractions: Optional[list[float]] = None,
) -> list[float]:
    """
    Generate a list of steering strengths for grid search from starting strength.

    Each fraction is used as an exact multiplier of starting_strength.
    Use negative fractions to steer toward short_term, positive for long_term.

    Args:
        starting_strength: Base steering magnitude
        fractions: Exact multipliers to apply. Defaults to [-1.0, -0.4, -0.1, 0.1, 0.4, 1.0]
                  Use negative values for short_term direction, positive for long_term.

    Returns:
        Sorted list of steering strengths

    Examples:
        fractions=[0.5, 1.0] -> only positive (long_term) steering
        fractions=[-1.0, 1.0] -> both directions at full strength
        fractions=[-0.5, -0.1, 0.1, 0.5] -> symmetric around 0
    """
    if fractions is None:
        # Default: symmetric around 0
        fractions = [-1.0, -0.4, -0.1, 0.1, 0.4, 1.0]
    strengths = [starting_strength * frac for frac in fractions]
    return sorted(strengths)


def get_probe_confidence(
    runner: ModelRunner,
    probe: LoadedProbe,
    prompt: str,
    steering: SteeringDirectionConfig,
    temperature: float = 100.0,
) -> float:
    """
    Get probe's prediction confidence after applying steering.

    Uses the probe's direction vector to compute confidence via dot product,
    then applies a temperature-scaled sigmoid for a smoother output.

    Args:
        runner: Model runner
        probe: Loaded probe with direction vector
        prompt: Text prompt
        steering: Steering configuration
        temperature: Temperature for sigmoid (lower = sharper, higher = smoother)

    Returns:
        Probability of long_term prediction (0-1), or 0.5 if failed
    """
    # Get activation at probe's layer/position with steering applied
    activation = runner.get_activation_with_steering(
        prompt,
        layer=probe.layer,
        position=probe.token_position_idx,
        steering=steering,
    )

    if activation is None:
        return 0.5  # Position out of bounds

    # Use dot product with probe direction for confidence
    # This gives a continuous value representing alignment with "long_term" direction
    try:
        direction = probe.direction.flatten()
        activation = activation.flatten()

        # Compute dot product (logit)
        logit = float(np.dot(direction, activation))

        # Add intercept if available
        if probe.intercept is not None:
            logit += float(np.asarray(probe.intercept).flatten()[0])

        # Apply temperature-scaled sigmoid for smoother output
        # Higher temperature = more spread out probabilities (less extreme 0/1)
        scaled_logit = logit / temperature
        prob = 1.0 / (1.0 + np.exp(-scaled_logit))

        return float(prob)
    except Exception:
        return 0.5  # Fallback on error


def resolve_specific_positions(
    prompt: str,
    specific_positions: list[Union[int, dict]],
    tokenizer,
    apply_chat_template: bool = True,
) -> list[int]:
    """
    Resolve abstract token position specs to actual token indices.

    Args:
        prompt: The prompt text (raw user content)
        specific_positions: List of positions (int or TokenPosition-style dict)
        tokenizer: Model tokenizer
        apply_chat_template: If True, apply chat template before resolving positions.
                            This is required for correct position resolution since the
                            model runs on the templated prompt.

    Returns:
        List of resolved token indices (in the templated prompt space)
    """
    from src.common.schemas import TokenPosition

    # Apply chat template if requested (matches what ModelRunner does)
    if apply_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        try:
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            formatted_prompt = prompt
    else:
        formatted_prompt = prompt

    tokens = tokenizer.encode(formatted_prompt)
    token_strs = [tokenizer.decode([t]) for t in tokens]

    resolved = []
    for pos in specific_positions:
        if isinstance(pos, int):
            # Literal position
            if pos < len(tokens):
                resolved.append(pos)
        elif isinstance(pos, dict):
            # Abstract position spec - parse as TokenPosition
            tp = TokenPosition.from_dict(pos)
            if tp.text is not None:
                # Text-based search - find the LAST occurrence and its ending token
                full_text = "".join(token_strs)
                last_start = full_text.rfind(tp.text)
                if last_start != -1:
                    # Find the token that contains the last character of this occurrence
                    target_end_char = last_start + len(tp.text) - 1
                    cum_chars = 0
                    for i, tok in enumerate(token_strs):
                        cum_chars += len(tok)
                        if cum_chars > target_end_char:
                            resolved.append(i)
                            break
            elif tp.prompt_index is not None:
                # Index into prompt
                idx = tp.prompt_index
                if idx < 0:
                    idx = len(tokens) + idx
                if 0 <= idx < len(tokens):
                    resolved.append(idx)
            elif tp.index is not None or tp.continuation_index is not None:
                # These don't make sense for prompt-only context, skip
                pass
        elif isinstance(pos, str):
            # Legacy: plain string means find text - find LAST occurrence
            full_text = "".join(token_strs)
            last_start = full_text.rfind(pos)
            if last_start != -1:
                # Find the token that contains the last character of this occurrence
                target_end_char = last_start + len(pos) - 1
                cum_chars = 0
                for i, tok in enumerate(token_strs):
                    cum_chars += len(tok)
                    if cum_chars > target_end_char:
                        resolved.append(i)
                        break

    return resolved


def print_steering_positions_per_sample(
    runner: ModelRunner,
    baselines: dict[int, SampleBaseline],
    specific_positions: list[Union[int, dict]],
) -> None:
    """Print tokens at specific steering positions for each sample."""
    print("\n  Steering positions per sample (in templated prompt):")
    print("  " + "-" * 60)

    tokenizer = runner.model.tokenizer

    for sample_id, baseline in sorted(baselines.items()):
        # Resolve positions using templated prompt (same as model uses)
        resolved = resolve_specific_positions(baseline.prompt, specific_positions, tokenizer)

        # Format prompt the same way for token lookup
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": baseline.prompt}]
            try:
                formatted_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                formatted_prompt = baseline.prompt
        else:
            formatted_prompt = baseline.prompt

        tokens = tokenizer.encode(formatted_prompt)
        token_strs = []
        for pos in resolved:
            if pos < len(tokens):
                token_id = tokens[pos]
                word = tokenizer.decode([token_id])
                # Clean up for display
                word_display = repr(word.replace("\n", "\\n"))
                token_strs.append(f"pos {pos}: {word_display}")
            else:
                token_strs.append(f"pos {pos}: <OOB>")
        print(f"  Sample {sample_id}: {', '.join(token_strs)}")

    print("  " + "-" * 60)


def run_steering_for_probe(
    runner: ModelRunner,
    probe: LoadedProbe,
    baselines: dict[int, SampleBaseline],
    steering_strengths: list[float],
    choice_prefix: str,
    max_new_tokens: int = 9,
    debug: bool = True,
    batch_size: int = 8,
    controls: Union[bool, int] = False,
    specific_positions: Optional[list[Union[int, dict]]] = None,
) -> ProbeSteeringResult:
    """Run steering experiment for a single probe using baselines from preference_data.

    Uses baselines from preference_data instead of generating them, enabling
    batch processing of steered generations per strength.

    Args:
        runner: ModelRunner for generation
        probe: Probe to use for steering
        baselines: Dict mapping sample_id -> SampleBaseline (from preference_data)
        steering_strengths: List of steering strengths to test
        choice_prefix: Prefix before choice label (e.g., "I select:")
        max_new_tokens: Max tokens to generate
        debug: Enable debug output
        batch_size: Batch size for processing (used for batched generation)
        controls: If True, also run with random direction vectors as control
        specific_positions: If set, apply steering only at these token positions.
                           Can be int (literal position) or dict (TokenPosition spec).
                           Uses APPLY_TO_TOKEN_POSITION instead of APPLY_TO_ALL.
    """
    layer = probe.layer
    probe_id = f"{probe.probe_type.value}_layer{layer}_pos{probe.token_position_idx}"

    # Check if probe is after horizon using metadata
    after_horizon = False
    if hasattr(probe, "metadata") and probe.metadata:
        pos_info = probe.metadata.get("position", {})
        after_horizon = pos_info.get("after_time_horizon", False)

    print(f"\n  Probe: {probe_id} (after_horizon={after_horizon})")

    # Debug: check steering direction properties
    if debug:
        direction = probe.direction
        norm = np.linalg.norm(direction)
        print(f"    Direction norm: {norm:.4f}, shape: {direction.shape}")
        print(f"    Direction range: [{direction.min():.4f}, {direction.max():.4f}]")

    # Build list of samples with their info
    sample_list = list(baselines.values())
    n_samples = len(sample_list)
    non_zero_strengths = sorted([s for s in steering_strengths if s != 0])

    # Generate random directions for controls (same norm as probe direction)
    # Normalize controls: False->0, True->1, int->int
    n_controls = int(controls) if isinstance(controls, (bool, int)) else 0
    random_directions = []
    if n_controls > 0:
        rng = np.random.RandomState(42)  # Fixed seed for reproducibility
        probe_norm = np.linalg.norm(probe.direction)
        for i in range(n_controls):
            rand_dir = rng.randn(len(probe.direction))
            rand_dir = rand_dir / np.linalg.norm(rand_dir) * probe_norm
            random_directions.append(rand_dir)
        print(f"  Controls: {n_controls} random direction(s) with same norm ({probe_norm:.4f})")

    # Initialize results dict: sample_id -> steered_choices dict
    results: dict[int, dict[float, str]] = {bl.sample_id: {} for bl in sample_list}
    # Track failed responses (unknown/degenerate) for debugging
    failed_responses: dict[int, dict[float, str]] = {
        bl.sample_id: {} for bl in sample_list
    }

    # Process each sample, going through all magnitudes per sample
    for sample_idx, baseline in enumerate(sample_list):
        # Resolve abstract positions to actual indices for this sample
        resolved_positions = None
        if specific_positions:
            resolved_positions = resolve_specific_positions(
                baseline.prompt,
                specific_positions,
                runner.model.tokenizer,
            )
            if not resolved_positions:
                print(f"Warning: No positions resolved for sample {baseline.sample_id}, skipping")
                continue

        print(f"\n{'=' * 80}", flush=True)
        print(f"SAMPLE {sample_idx + 1}/{n_samples} (id={baseline.sample_id})")
        print(f"{'=' * 80}")
        if baseline.time_horizon:
            print(f"TIME HORIZON: {baseline.time_horizon}")
        print(f"PROMPT: {baseline.prompt[:300]}...")
        print(f"BASELINE CHOICE: {baseline.baseline_choice}")
        print(f"Short: {baseline.short_label} ({baseline.short_value}), Long: {baseline.long_label} ({baseline.long_value})")
        if resolved_positions:
            print(f"STEERING POSITIONS: {resolved_positions}")
        print()

        # Store strength 0 as baseline
        if 0 in steering_strengths or 0.0 in steering_strengths:
            results[baseline.sample_id][0] = baseline.baseline_choice
            results[baseline.sample_id][0.0] = baseline.baseline_choice

        # Table header for probability progression
        # Calculate column widths based on label lengths (min 5 for data values like "0.00")
        short_col_w = max(5, len(f"P({baseline.short_label})"))
        long_col_w = max(5, len(f"P({baseline.long_label})"))

        # Probe column header: for choice probes, class 1 = long_term
        # Use "Probe" to avoid confusion with choice label columns
        probe_header = "Probe"

        if n_controls > 0:
            # Build control columns header - must match data format exactly
            # Data format: {c_abbrev:>4} {ctrl_sp:>5.2f} {ctrl_lp:>5.2f} {ctrl_pc:>5.2f} = 22 chars
            ctrl_headers = " | ".join([f"{'C'+str(i+1):>4} {'P(S)':>5} {'P(L)':>5} {'Prb':>5}" for i in range(n_controls)])
            print(f"{'Steer Mag':>10} | {'Choice':>12} | {'P(' + baseline.short_label + ')':>{short_col_w}} | {'P(' + baseline.long_label + ')':>{long_col_w}} | {probe_header:>6} | {ctrl_headers} | Result")
            header_len = 49 + short_col_w + long_col_w + n_controls * 25
            print("-" * header_len)
        else:
            print(f"{'Steer Mag':>10} | {'Choice':>12} | {'P(' + baseline.short_label + ')':>{short_col_w}} | {'P(' + baseline.long_label + ')':>{long_col_w}} | {probe_header:>6} | Result")
            print("-" * (55 + short_col_w + long_col_w))

        # Track flips for this sample
        first_flip_strength = None
        flip_direction = None

        # Process all magnitudes for this sample
        for strength in non_zero_strengths:
            # Determine steering option based on resolved_positions
            if resolved_positions:
                steering_option = SteeringOption.APPLY_TO_TOKEN_POSITION
                token_position = TokenPositionTarget(indices=resolved_positions)
            else:
                steering_option = SteeringOption.APPLY_TO_ALL
                token_position = None

            # Create steering config
            steering_config = SteeringDirectionConfig(
                direction=probe.direction,
                layer=layer,
                strength=strength,
                option=steering_option,
                token_position=token_position,
            )

            # Generate steered response
            response = runner.generate_with_steering(
                baseline.prompt,
                steering=steering_config,
                max_new_tokens=max_new_tokens,
            )

            # Parse choice
            choice = parse_choice_from_response(
                response,
                baseline.short_label,
                baseline.long_label,
                debug=debug,
                short_value=baseline.short_value,
                long_value=baseline.long_value,
            )
            results[baseline.sample_id][strength] = choice

            # Capture failed responses
            if choice in ("unknown", "degenerate"):
                failed_responses[baseline.sample_id][strength] = response

            # Get choice token probabilities
            label_probs = runner.get_label_probs_with_steering(
                baseline.prompt,
                (baseline.short_label, baseline.long_label),
                steering_config,
                choice_prefix=choice_prefix,
            )
            short_prob = label_probs.prob1
            long_prob = label_probs.prob2

            # Get probe confidence (probability of long_term) - only for trained probes
            is_contrastive = probe.metadata.get("source") == "contrastive"
            if is_contrastive:
                probe_conf = None  # No probe confidence for contrastive vectors
            else:
                probe_conf = get_probe_confidence(
                    runner, probe, baseline.prompt, steering_config
                )

            # Run controls with random directions if enabled
            ctrl_results = []  # List of (choice, short_prob, long_prob, probe_conf) tuples
            for ctrl_idx, rand_dir in enumerate(random_directions):
                ctrl_steering_config = SteeringDirectionConfig(
                    direction=rand_dir,
                    layer=layer,
                    strength=strength,
                    option=steering_option,
                    token_position=token_position,
                )
                ctrl_response = runner.generate_with_steering(
                    baseline.prompt,
                    steering=ctrl_steering_config,
                    max_new_tokens=max_new_tokens,
                )
                ctrl_choice = parse_choice_from_response(
                    ctrl_response,
                    baseline.short_label,
                    baseline.long_label,
                    debug=debug,
                    short_value=baseline.short_value,
                    long_value=baseline.long_value,
                )
                # Get control label probabilities
                ctrl_label_probs = runner.get_label_probs_with_steering(
                    baseline.prompt,
                    (baseline.short_label, baseline.long_label),
                    ctrl_steering_config,
                    choice_prefix=choice_prefix,
                )
                # Get control probe confidence (only for trained probes)
                if is_contrastive:
                    ctrl_probe_conf = None
                else:
                    ctrl_probe_conf = get_probe_confidence(
                        runner, probe, baseline.prompt, ctrl_steering_config
                    )
                ctrl_results.append((
                    ctrl_choice,
                    ctrl_label_probs.prob1,  # short prob
                    ctrl_label_probs.prob2,  # long prob
                    ctrl_probe_conf,
                ))

            # Check for flip
            is_flip = (
                baseline.baseline_choice in ("short_term", "long_term")
                and choice in ("short_term", "long_term")
                and choice != baseline.baseline_choice
            )

            # Track first flip
            if is_flip and first_flip_strength is None:
                first_flip_strength = strength
                flip_direction = f"{baseline.baseline_choice} -> {choice}"

            # Result indicator
            if choice == "degenerate":
                result_str = "DEGEN"
            elif choice == "unknown":
                result_str = "???"
            elif is_flip:
                result_str = "FLIP!"
            else:
                result_str = "No flip"

            # Print row (probe_conf shows P(long_term) from probe, or N/A for contrastive)
            probe_conf_str = "N/A" if probe_conf is None else f"{probe_conf:.2f}"
            if ctrl_results:
                # Build control columns: choice (abbrev), P(S), P(L), probe
                ctrl_strs = []
                for ctrl_choice, ctrl_sp, ctrl_lp, ctrl_pc in ctrl_results:
                    # Abbreviate choice for compact display
                    if ctrl_choice == "short_term":
                        c_abbrev = "ST"
                    elif ctrl_choice == "long_term":
                        c_abbrev = "LT"
                    elif ctrl_choice == "degenerate":
                        c_abbrev = "DEG"
                    else:
                        c_abbrev = "??"
                    ctrl_pc_str = "N/A" if ctrl_pc is None else f"{ctrl_pc:.2f}"
                    ctrl_strs.append(f"{c_abbrev:>4} {ctrl_sp:>5.2f} {ctrl_lp:>5.2f} {ctrl_pc_str:>5}")
                ctrl_output = " | ".join(ctrl_strs)
                print(f"{strength:>+10.1f} | {choice:>12} | {short_prob:>{short_col_w}.2f} | {long_prob:>{long_col_w}.2f} | {probe_conf_str:>6} | {ctrl_output} | {result_str}")
            else:
                print(f"{strength:>+10.1f} | {choice:>12} | {short_prob:>{short_col_w}.2f} | {long_prob:>{long_col_w}.2f} | {probe_conf_str:>6} | {result_str}")

        # Sample summary
        summary_width = (49 + short_col_w + long_col_w + n_controls * 25) if n_controls > 0 else (55 + short_col_w + long_col_w)
        print("-" * summary_width)
        if first_flip_strength is not None:
            print(f">>> FIRST FLIP at strength {first_flip_strength:+.1f}: {flip_direction} <<<")
        else:
            print("No flip observed")
        print(f"{'=' * summary_width}\n", flush=True)

    # Build final SampleSteeringResult list
    samples = []
    for baseline in sample_list:
        steered_choices = results[baseline.sample_id]
        baseline_choice = baseline.baseline_choice

        # Find minimum flip magnitudes
        min_flip_to_long = None
        min_flip_to_short = None

        if baseline_choice == "short_term":
            # Look for flip to long_term
            for s in sorted(steering_strengths, key=abs):
                if steered_choices.get(s) == "long_term":
                    min_flip_to_long = abs(s)
                    break

        elif baseline_choice == "long_term":
            # Look for flip to short_term
            for s in sorted(steering_strengths, key=abs):
                if steered_choices.get(s) == "short_term":
                    min_flip_to_short = abs(s)
                    break

        # Only include failed_responses if there are any
        sample_failed = failed_responses[baseline.sample_id]

        # Parse reward values (remove commas from formatted strings)
        try:
            short_reward = float(baseline.short_value.replace(",", ""))
        except (ValueError, AttributeError):
            short_reward = None
        try:
            long_reward = float(baseline.long_value.replace(",", ""))
        except (ValueError, AttributeError):
            long_reward = None

        samples.append(
            SampleSteeringResult(
                sample_id=baseline.sample_id,
                time_horizon=baseline.time_horizon,
                baseline_choice=baseline_choice,
                steered_choices=steered_choices,
                min_flip_magnitude_to_long=min_flip_to_long,
                min_flip_magnitude_to_short=min_flip_to_short,
                failed_responses=sample_failed if sample_failed else None,
                short_reward=short_reward,
                long_reward=long_reward,
            )
        )

    return ProbeSteeringResult(
        probe_id=probe_id,
        probe_type=probe.probe_type.value,
        layer=layer,
        token_position_idx=probe.token_position_idx,
        after_horizon=after_horizon,
        samples=samples,
    )


def filter_probes_after_horizon(
    probes: dict[str, LoadedProbe],
    index: dict,
) -> dict[str, LoadedProbe]:
    """Filter to only probes that are after time horizon."""
    filtered = {}

    # Get probe summaries with position info
    probe_summaries = {p["id"]: p for p in index.get("probes", [])}

    for probe_id, probe in probes.items():
        summary = probe_summaries.get(probe_id, {})
        if summary.get("after_horizon", False):
            filtered[probe_id] = probe

    if not filtered:
        print("  Warning: No probes with after_horizon=True found, using all probes")
        return probes

    return filtered


def filter_probes_before_choice(
    probes: dict[str, LoadedProbe],
    index: dict,
) -> dict[str, LoadedProbe]:
    """Filter to only probes that are before choice selection."""
    filtered = {}

    # Get probe summaries with position info
    probe_summaries = {p["id"]: p for p in index.get("probes", [])}

    for probe_id, probe in probes.items():
        summary = probe_summaries.get(probe_id, {})
        # after_choice_made=False means before choice
        if not summary.get("after_choice_made", True):
            filtered[probe_id] = probe

    if not filtered:
        print(
            "  Warning: No probes with after_choice_made=False found, using all probes"
        )
        return probes

    return filtered


def filter_probes_by_specs(
    probes: dict[str, LoadedProbe],
    specs: list[ProbeSpec],
    n_layers: int,
    token_info: Optional[dict] = None,
) -> dict[str, LoadedProbe]:
    """Filter probes to only those matching the given specifications.

    Args:
        probes: Dict of probe_id -> LoadedProbe
        specs: List of ProbeSpec to match
        n_layers: Total number of layers in model (for negative indexing)
        token_info: Token position info for text matching

    Returns:
        Filtered dict of probes matching any of the specs
    """
    filtered = {}

    for probe_id, probe in probes.items():
        for spec in specs:
            # Get actual token_position from metadata if available
            token_position = probe.metadata.get("token_position") if probe.metadata else None
            if spec.matches(
                probe.layer,
                token_position,
                probe.token_position_idx,
                n_layers,
                token_info,
            ):
                filtered[probe_id] = probe
                break  # No need to check other specs once matched

    if not filtered:
        # Build helpful error message showing available probes
        available = []
        for probe_id, probe in list(probes.items())[:5]:
            token_pos = probe.metadata.get("token_position") if probe.metadata else None
            pos_info = f"token_pos={token_pos}" if token_pos else f"idx={probe.token_position_idx}"
            available.append(f"  - layer={probe.layer}, {pos_info}")
        more = f"  ... and {len(probes) - 5} more" if len(probes) > 5 else ""

        raise ValueError(
            f"No probes matched specific_probes specs.\n"
            f"Specs: {[(s.layer, s.token_pos) for s in specs]}\n"
            f"Available probes (first 5):\n" + "\n".join(available) + (f"\n{more}" if more else "")
        )

    return filtered


def find_best_probe_before_choice(
    probes: dict[str, LoadedProbe],
    index: dict,
    probe_types: list[str],
) -> Optional[tuple[str, LoadedProbe]]:
    """
    Find the best probe among those before choice selection.

    Returns the probe with highest accuracy among probes where after_choice_made=False.
    Only considers probes of the requested types.
    """
    probe_summaries = {p["id"]: p for p in index.get("probes", [])}

    best_probe_id = None
    best_accuracy = -1.0

    for probe_id, probe in probes.items():
        # Check if this probe type is requested
        if probe.probe_type.value not in probe_types:
            continue

        summary = probe_summaries.get(probe_id, {})

        # Only consider probes before choice selection
        if summary.get("after_choice_made", True):
            continue

        accuracy = summary.get("accuracy", 0.0)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_probe_id = probe_id

    if best_probe_id is None:
        return None

    print(f"  Best probe before choice: {best_probe_id} (accuracy={best_accuracy:.4f})")
    return best_probe_id, probes[best_probe_id]


def load_contrastive_vectors_as_probes(vectors_dir: Path) -> dict[str, LoadedProbe]:
    """
    Load contrastive (CAA) vectors as LoadedProbe objects for steering.

    This allows using CAA vectors with the same interface as trained probes.
    The loaded probes will have probe_type="contrastive".

    Args:
        vectors_dir: Directory containing CAA vectors (e.g., out/contrastive/{id}/vectors/)

    Returns:
        Dict mapping probe_id to LoadedProbe
    """
    probes = {}

    # Load index
    index_path = vectors_dir / "index.json"
    if not index_path.exists():
        print(f"  Warning: No contrastive index found at {index_path}")
        return probes

    index = load_json(index_path)

    for vec_info in index.get("vectors", []):
        layer = vec_info["layer"]
        position = vec_info["position"]
        filename = vec_info["file"]

        # Load direction vector
        direction_path = vectors_dir / filename
        if not direction_path.exists():
            continue

        direction = np.load(direction_path)

        # Create a LoadedProbe-like object
        probe_id = f"caa_layer{layer}_pos{position}"
        probe = LoadedProbe(
            model=None,  # CAA doesn't have sklearn model (uses direction directly)
            direction=direction,
            intercept=None,
            probe_type=ProbeType.CHOICE,  # Treat as choice type for filtering
            layer=layer,
            token_position_idx=position,
            n_features=len(direction),
            model_name="",  # Will be filled from CAA results
            metadata={
                "source": "contrastive",
                "token_position": vec_info.get("token_position"),  # Actual sequence position
                "token_position_word": vec_info.get("token_position_word"),  # Word at position
                "n_short": vec_info.get("n_short", 0),
                "n_long": vec_info.get("n_long", 0),
            },
        )
        probes[probe_id] = probe

    print(f"  Loaded {len(probes)} contrastive vectors")
    return probes


def run_steering_experiment(
    config: TrySteeringConfig,
    debug: bool = True,
    output_dir: Optional[Path] = None,
) -> SteeringExperimentOutput:
    """
    Run full steering experiment with incremental saving.

    Results are saved after each probe completes, allowing:
    - Progress persistence if interrupted
    - Lower memory usage (cleared after each probe)
    - Real-time visibility into experiment progress

    Args:
        config: Steering experiment configuration
        debug: If True, print debug output
        output_dir: Directory for output (auto-created if None)

    Returns:
        SteeringExperimentOutput with all results
    """
    # Determine model name from steering sources
    probe_model_name = None
    probes_dir = None

    # Load from probes if available
    if config.probes_id:
        probes_base = PROJECT_ROOT / "out" / "probes" / config.probes_id
        if not probes_base.exists():
            raise FileNotFoundError(f"Probes config not found: {probes_base}")

        probes_dir = probes_base / "probes"
        if not probes_dir.exists():
            raise FileNotFoundError(f"Probes directory not found: {probes_dir}")

        # Load probe index (new format)
        index_path = probes_dir / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Probe index not found: {index_path}")

        index = load_json(index_path)
        # New format uses "model", old format used "model_name"
        probe_model_name = index.get("model", index.get("model_name"))
        if not probe_model_name:
            raise ValueError("No model name found in probe index")

        print(f"Probes ID: {config.probes_id}")
        print(f"Probe model: {probe_model_name}")

    # Load from contrastive vectors if available
    contrastive_dir = None
    contrastive_model_name = None
    if config.contrastive_id:
        contrastive_base = PROJECT_ROOT / "out" / "contrastive" / config.contrastive_id
        if not contrastive_base.exists():
            raise FileNotFoundError(f"Contrastive config not found: {contrastive_base}")

        contrastive_dir = contrastive_base / "vectors"
        results_path = contrastive_base / "results" / "caa_results.json"
        if results_path.exists():
            caa_results = load_json(results_path)
            contrastive_model_name = caa_results.get("model")
            print(f"Contrastive ID: {config.contrastive_id}")
            print(f"Contrastive model: {contrastive_model_name}")

    # Determine which model to use
    if probe_model_name and contrastive_model_name:
        if probe_model_name != contrastive_model_name:
            raise ValueError(
                f"Model mismatch: probes use {probe_model_name}, "
                f"contrastive uses {contrastive_model_name}"
            )
    steering_model_name = probe_model_name or contrastive_model_name
    if not steering_model_name:
        raise ValueError("No model name found in probes or contrastive vectors")

    # Load preference data for baselines
    print(f"\nLoading preference data: {config.query_id}")
    pref_info = load_preference_data_for_steering(
        config.query_id,
        max_samples=config.max_samples,
        sample_indices=config.sample_indices,
        baseline_choice=config.baseline_choice,
        baseline_horizon=config.baseline_horizon,
    )
    baselines = pref_info.baselines
    dataset_id = pref_info.dataset_id
    model_name = pref_info.model_name
    choice_prefix = pref_info.formatting_config.choice_prefix
    sample_ids = list(baselines.keys())

    print(f"  Loaded {len(baselines)} samples with baselines: {sample_ids}")
    print(f"  Dataset ID: {dataset_id}")
    print(f"  Model: {model_name}")

    # Validate model matches between preference_data and steering sources
    validate_model_matches(model_name, steering_model_name)

    # Load model using ModelRunner
    runner = load_model_for_steering(model_name)

    # Print steering positions per sample if specific_positions is set
    if config.specific_positions:
        print_steering_positions_per_sample(runner, baselines, config.specific_positions)

    # Load probes and contrastive vectors separately for independent filtering
    trained_probes = {}
    contrastive_probes = {}
    index = {}

    if probes_dir:
        trained_probes = load_probes_from_dir(probes_dir)
        index_path = probes_dir / "index.json"
        if index_path.exists():
            index = load_json(index_path)

    # Load contrastive vectors as "probes" (they have same interface)
    contrastive_index = {}
    if contrastive_dir:
        contrastive_probes = load_contrastive_vectors_as_probes(contrastive_dir)
        caa_index_path = contrastive_dir / "index.json"
        if caa_index_path.exists():
            contrastive_index = load_json(caa_index_path)
        # Use contrastive index for token info if no probes index
        if not index:
            index = contrastive_index

    n_layers = index.get("n_layers", 28)  # Default for common models

    # Load token_info for text-based probe matching in specific_probes
    token_info = None
    if config.probes_id:
        token_info = get_token_position_info_from_probe_data(config.probes_id)

    # Filter trained probes by type
    trained_by_type = {
        pid: p
        for pid, p in trained_probes.items()
        if p.probe_type.value in config.probe_types
    }

    # Apply specific_probes filter to trained probes
    if config.specific_probes:
        trained_by_type = filter_probes_by_specs(
            trained_by_type, config.specific_probes, n_layers, token_info
        )
        print(f"  Filtered to {len(trained_by_type)} trained probes matching specific_probes")
    elif trained_by_type:
        # Apply position filters only if no specific_probes
        if config.only_after_horizon:
            trained_by_type = filter_probes_after_horizon(trained_by_type, index)
        if config.only_before_choice:
            trained_by_type = filter_probes_before_choice(trained_by_type, index)

    # Filter contrastive vectors by type
    contrastive_by_type = {
        pid: p
        for pid, p in contrastive_probes.items()
        if p.probe_type.value in config.probe_types
    }

    # Apply specific_contrastive filter to contrastive vectors
    if config.specific_contrastive:
        # Load token_info for contrastive if available
        caa_token_info = None
        if config.contrastive_id:
            caa_token_info = get_token_position_info_from_contrastive_data(config.contrastive_id)
        # Fall back to probe token_info if contrastive doesn't have one
        if caa_token_info is None:
            caa_token_info = token_info
        contrastive_by_type = filter_probes_by_specs(
            contrastive_by_type, config.specific_contrastive, n_layers, caa_token_info
        )
        print(f"  Filtered to {len(contrastive_by_type)} contrastive vectors matching specific_contrastive")
    elif contrastive_by_type:
        # Apply position filters only if no specific_contrastive
        caa_idx = contrastive_index or index
        if config.only_after_horizon:
            contrastive_by_type = filter_probes_after_horizon(contrastive_by_type, caa_idx)
        if config.only_before_choice:
            contrastive_by_type = filter_probes_before_choice(contrastive_by_type, caa_idx)

    # Merge filtered probes and contrastive vectors
    probes_by_type = {**trained_by_type, **contrastive_by_type}

    print(f"\nTesting {len(probes_by_type)} probes:")
    for pid in sorted(probes_by_type.keys()):
        print(f"  - {pid}")

    # Generate steering strengths from starting value
    steering_strengths = generate_steering_strengths(
        config.starting_steering_strength, config.fractions
    )
    print(f"\nSteering strengths: {steering_strengths}")

    # Setup output directory and incremental saver
    if output_dir is None:
        output_base = PROJECT_ROOT / "out" / "steering"
        output_dir = output_base / config.get_id()
    results_dir = output_dir / "results"
    timestamp = get_timestamp()

    saver = IncrementalSaver(
        results_dir=results_dir,
        timestamp=timestamp,
        steering_id=config.get_id(),
        query_id=config.query_id,
        dataset_id=dataset_id,
        model_name=model_name,
        steering_strengths=steering_strengths,
        n_samples=len(baselines),
        sample_indices=sample_ids,
        probes_id=config.probes_id,
        contrastive_id=config.contrastive_id,
    )
    print(f"\nResults will be saved incrementally to: {saver.get_results_path()}")

    # Run steering for each probe with incremental saving and memory clearing
    # Convert to list to allow deletion during iteration
    probe_items = list(probes_by_type.items())
    probe_results = []

    for i, (probe_id, probe) in enumerate(probe_items):
        print(f"\n[{i + 1}/{len(probe_items)}] Processing {probe_id}...")

        # Skip if already completed (allows resumption)
        if saver.is_completed(probe_id):
            print("    Skipping (already completed)")
            # Still delete probe from memory
            del probes_by_type[probe_id]
            continue

        result = run_steering_for_probe(
            runner,
            probe,
            baselines,
            steering_strengths,
            choice_prefix=choice_prefix,
            debug=debug,
            batch_size=config.batch_size,
            controls=config.controls,
            specific_positions=config.specific_positions,
        )
        probe_results.append(result)

        # Save result immediately
        saver.add_probe_result(result)

        # Delete probe from memory - we no longer need it
        del probes_by_type[probe_id]

        # Clear memory after each probe to prevent accumulation
        clear_memory()

    # Reconstruct probe_results from saved data for return value
    # (includes any previously completed probes if resuming)
    all_saved_results = saver.get_probe_results()

    # Convert back to ProbeSteeringResult objects
    final_probe_results = []
    for pr_data in all_saved_results:
        samples = []
        for s_data in pr_data.get("samples", []):
            steered_choices = {
                float(k): v for k, v in s_data.get("steered_choices", {}).items()
            }
            # Deserialize failed_responses if present
            failed_responses_data = s_data.get("failed_responses")
            failed_responses = None
            if failed_responses_data:
                failed_responses = {
                    float(k): v for k, v in failed_responses_data.items()
                }
            samples.append(
                SampleSteeringResult(
                    sample_id=s_data["sample_id"],
                    time_horizon=s_data.get("time_horizon"),
                    baseline_choice=s_data["baseline_choice"],
                    steered_choices=steered_choices,
                    min_flip_magnitude_to_long=s_data.get("min_flip_to_long"),
                    min_flip_magnitude_to_short=s_data.get("min_flip_to_short"),
                    failed_responses=failed_responses,
                    short_reward=s_data.get("short_reward"),
                    long_reward=s_data.get("long_reward"),
                )
            )
        final_probe_results.append(
            ProbeSteeringResult(
                probe_id=pr_data["probe_id"],
                probe_type=pr_data["probe_type"],
                layer=pr_data["layer"],
                token_position_idx=pr_data["token_position_idx"],
                after_horizon=pr_data.get("after_horizon", False),
                samples=samples,
            )
        )

    return SteeringExperimentOutput(
        steering_id=config.get_id(),
        query_id=config.query_id,
        dataset_id=dataset_id,
        model_name=model_name,
        steering_strengths=steering_strengths,
        n_samples=len(baselines),
        sample_indices=sample_ids,
        probe_results=final_probe_results,
        timestamp=timestamp,
        probes_id=config.probes_id,
        contrastive_id=config.contrastive_id,
    )


# =============================================================================
# Visualization - Steering Heatmaps
# =============================================================================


def create_steering_heatmaps(
    output: SteeringExperimentOutput,
    output_dir: Path,
    probe_index: Optional[dict] = None,
    token_info: Optional[dict] = None,
) -> None:
    """
    Create heatmaps showing minimum steering magnitude to flip choice.

    Creates separate heatmaps for each probe type:
    - viz/{probe_type}/heatmap_short_to_long.png
    - viz/{probe_type}/heatmap_long_to_short.png

    Cell values:
    - Number: min |strength| that caused flip
    - Gray "N/A": never flipped
    - Black "FAIL": degeneration (can't parse)

    Args:
        output: Steering experiment results
        output_dir: Directory to save visualizations
        probe_index: Full probe index
        token_info: Dict with 'tokens' and 'resolved_positions' for x-axis labels
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    viz_dir = output_dir / "viz"
    ensure_dir(viz_dir)

    # Group probe results by type
    probes_by_type: dict[str, list[ProbeSteeringResult]] = {}
    for pr in output.probe_results:
        ptype = pr.probe_type if isinstance(pr.probe_type, str) else pr.probe_type.value
        if ptype not in probes_by_type:
            probes_by_type[ptype] = []
        probes_by_type[ptype].append(pr)

    for probe_type, type_probes in probes_by_type.items():
        type_viz_dir = viz_dir / probe_type
        ensure_dir(type_viz_dir)

        # Get layers and positions for this probe type
        if probe_index is not None:
            all_probes = probe_index.get("probes", [])
            relevant_probes = [p for p in all_probes if p.get("type") == probe_type]
            layers = sorted(set(p["layer"] for p in relevant_probes))
            positions = sorted(set(p["position"] for p in relevant_probes))
        else:
            layers = sorted(set(pr.layer for pr in type_probes))
            positions = sorted(set(pr.token_position_idx for pr in type_probes))

        if not layers or not positions:
            print(f"  No probes to visualize for {probe_type}")
            continue

        # Build position labels like train_probes heatmaps
        def build_position_labels():
            from src.plotting.common import format_token_position_label

            labels = []
            specs = token_info.get("specs", []) if token_info else []
            for pos_idx in positions:
                # Build spec label like heatmaps do
                if specs and pos_idx < len(specs):
                    spec_label = format_token_position_label(specs[pos_idx])
                else:
                    spec_label = f"pos_{pos_idx}"

                if token_info:
                    token = token_info.get("tokens", {}).get(pos_idx, "")
                    resolved_pos = token_info.get("resolved_positions", {}).get(
                        pos_idx, "?"
                    )
                    token_display = repr(token) if token else ""
                    labels.append(f"{spec_label}\n[{resolved_pos}] {token_display}")
                else:
                    labels.append(spec_label)
            return labels

        position_labels = build_position_labels()

        # Build matrices for each flip direction
        # short->long matrix
        sl_matrix = np.full((len(layers), len(positions)), np.nan)
        sl_fail = np.zeros((len(layers), len(positions)), dtype=bool)
        sl_never = np.zeros((len(layers), len(positions)), dtype=bool)
        sl_not_tested = np.ones(
            (len(layers), len(positions)), dtype=bool
        )  # Start as all not tested

        # long->short matrix
        ls_matrix = np.full((len(layers), len(positions)), np.nan)
        ls_fail = np.zeros((len(layers), len(positions)), dtype=bool)
        ls_never = np.zeros((len(layers), len(positions)), dtype=bool)
        ls_not_tested = np.ones(
            (len(layers), len(positions)), dtype=bool
        )  # Start as all not tested

        for pr in type_probes:
            if pr.layer not in layers or pr.token_position_idx not in positions:
                continue

            li = layers.index(pr.layer)
            pi = positions.index(pr.token_position_idx)

            # Mark as tested
            sl_not_tested[li, pi] = False
            ls_not_tested[li, pi] = False

            # Collect flip magnitudes and check for degeneration
            sl_flips = []
            ls_flips = []
            has_degenerate = False

            for sample in pr.samples:
                # Check for degeneration
                for choice in sample.steered_choices.values():
                    if choice == "degenerate":
                        has_degenerate = True
                        break

                if sample.min_flip_magnitude_to_long is not None:
                    sl_flips.append(sample.min_flip_magnitude_to_long)
                if sample.min_flip_magnitude_to_short is not None:
                    ls_flips.append(sample.min_flip_magnitude_to_short)

            # Short->Long
            if has_degenerate and not sl_flips:
                sl_fail[li, pi] = True
            elif sl_flips:
                sl_matrix[li, pi] = np.mean(sl_flips)
            else:
                sl_never[li, pi] = True
                sl_matrix[li, pi] = (
                    0  # Set to 0 so it shows green (tested but never flipped)
                )

            # Long->Short
            if has_degenerate and not ls_flips:
                ls_fail[li, pi] = True
            elif ls_flips:
                ls_matrix[li, pi] = np.mean(ls_flips)
            else:
                ls_never[li, pi] = True
                ls_matrix[li, pi] = (
                    0  # Set to 0 so it shows green (tested but never flipped)
                )

        # Verify at least one probe result was written
        tested_count = np.sum(~sl_not_tested)
        if tested_count == 0:
            print(f"  Warning: No probe results written for {probe_type}")
            continue

        # Create heatmaps
        max_strength = max(abs(s) for s in output.steering_strengths)

        for matrix, fail, never, not_tested, title, filename in [
            (
                sl_matrix,
                sl_fail,
                sl_never,
                sl_not_tested,
                "Short→Long Flip",
                "heatmap_short_to_long.png",
            ),
            (
                ls_matrix,
                ls_fail,
                ls_never,
                ls_not_tested,
                "Long→Short Flip",
                "heatmap_long_to_short.png",
            ),
        ]:
            # Calculate figure size based on data (similar to train_probes)
            fig_height = max(6, len(layers) * 0.5 + 1)
            fig_width = max(12, len(positions) * 1.5)
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))

            # Create a custom colormap that has white for masked values
            cmap = plt.cm.RdYlGn_r.copy()  # Red=high magnitude, Green=low magnitude
            cmap.set_bad(color="white")  # Masked values will be white
            norm = Normalize(vmin=0, vmax=max_strength)

            # Create the display matrix - start with actual values
            display_matrix = matrix.copy()

            # Mask cells that are not tested (shows as white)
            display_matrix = np.ma.array(display_matrix)
            display_matrix[not_tested] = np.ma.masked

            # Plot heatmap with origin='lower' so layer 0 is at bottom (like train_probes)
            im = ax.imshow(
                display_matrix, cmap=cmap, norm=norm, aspect="auto", origin="lower"
            )

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label("Min |Steering| to Flip", fontsize=10)

            # Add text annotations
            for i in range(len(layers)):
                for j in range(len(positions)):
                    if not_tested[i, j]:
                        # Not tested - light gray text on white
                        ax.text(
                            j,
                            i,
                            "N/A",
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="#cccccc",
                            style="italic",
                        )
                    elif fail[i, j]:
                        ax.text(
                            j,
                            i,
                            "FAIL",
                            ha="center",
                            va="center",
                            fontsize=8,
                            fontweight="bold",
                            color="white",
                            bbox=dict(boxstyle="round", facecolor="black", alpha=0.8),
                        )
                    elif never[i, j]:
                        # Tested but never flipped - show max magnitude tested
                        ax.text(
                            j,
                            i,
                            f"NEVER\n(max {max_strength:.0f})",
                            ha="center",
                            va="center",
                            fontsize=7,
                            color="#666666",
                        )
                    elif not np.isnan(matrix[i, j]):
                        val = matrix[i, j]
                        color = "white" if val > max_strength * 0.6 else "black"
                        ax.text(
                            j,
                            i,
                            f"{val:.1f}",
                            ha="center",
                            va="center",
                            fontsize=8,
                            fontweight="bold",
                            color=color,
                        )

            # Labels - match train_probes style with token info
            ax.set_xticks(range(len(positions)))
            ax.set_xticklabels(position_labels, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(layers)))
            ax.set_yticklabels([f"Layer {l}" for l in layers])
            ax.set_xlabel("Token Position (sequence order)", fontsize=11)
            ax.set_ylabel("Layer", fontsize=11)
            ax.set_title(
                f"[{probe_type}] Min Steering Magnitude for {title}\n"
                f"Model: {output.model_name} | n={output.n_samples}",
                fontsize=12,
            )

            plt.tight_layout()
            plt.savefig(type_viz_dir / filename, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close()
            print(f"  Saved: {type_viz_dir / filename}")


def get_probe_viz_dir(viz_dir: Path, pr: ProbeSteeringResult) -> Path:
    """Get the visualization directory for a probe: viz/{probe_type}/layer_{L}_position_{P}/"""
    ptype = pr.probe_type if isinstance(pr.probe_type, str) else pr.probe_type.value
    probe_dir = viz_dir / ptype / f"layer_{pr.layer}_position_{pr.token_position_idx}"
    ensure_dir(probe_dir)
    return probe_dir


def create_summary_plots(output: SteeringExperimentOutput, output_dir: Path) -> None:
    """Create summary bar plots of flip rates."""
    import matplotlib.pyplot as plt

    viz_dir = output_dir / "viz"
    ensure_dir(viz_dir)

    for pr in output.probe_results:
        probe_viz_dir = get_probe_viz_dir(viz_dir, pr)

        fig, ax = plt.subplots(figsize=(10, 6))

        strengths = sorted(output.steering_strengths)

        # Count flips, unflipped, and failed (degenerate + unknown)
        flip_counts = []
        unflip_counts = []
        failed_counts = []

        for s in strengths:
            flips = 0
            unflipped = 0
            failed = 0
            for sample in pr.samples:
                choice = sample.steered_choices.get(s, "unknown")
                if choice in ("degenerate", "unknown"):
                    failed += 1
                elif choice in ("short_term", "long_term"):
                    if choice != sample.baseline_choice:
                        flips += 1
                    else:
                        unflipped += 1
                else:
                    # Other unexpected choices count as failed
                    failed += 1
            flip_counts.append(flips)
            unflip_counts.append(unflipped)
            failed_counts.append(failed)

        x = range(len(strengths))
        width = 0.25

        ax.bar(
            [i - width for i in x],
            flip_counts,
            width,
            label="Flipped",
            color="steelblue",
        )
        ax.bar(
            x,
            unflip_counts,
            width,
            label="Unflipped",
            color="gray",
        )
        ax.bar(
            [i + width for i in x],
            failed_counts,
            width,
            label="Failed",
            color="crimson",
        )

        ax.set_xticks(x)
        ax.set_xticklabels([f"{s:+.1f}" for s in strengths])
        ax.set_xlabel("Steering Strength")
        ax.set_ylabel("Count")
        ax.set_title(f"Steering Effects: {pr.probe_id}")
        ax.legend()
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.savefig(probe_viz_dir / "summary.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved: {probe_viz_dir / 'summary.png'}")


def create_probe_analysis_plots(
    result: ProbeSteeringResult,
    steering_strengths: list[float],
    viz_dir: Path,
) -> None:
    """
    Create detailed analysis plots for a single probe showing flip distribution
    by steering magnitude, time horizon, reward ratio, and degenerate counts.
    """
    import matplotlib.pyplot as plt

    ensure_dir(viz_dir)
    probe_id = result.probe_id

    # Collect data for analysis
    data = []
    for s in result.samples:
        if s.baseline_choice not in ("short_term", "long_term"):
            continue  # Skip unknown baselines

        # Calculate reward ratio
        reward_ratio = None
        if s.short_reward and s.long_reward and s.short_reward > 0:
            reward_ratio = s.long_reward / s.short_reward

        # Get time horizon value (if available)
        horizon_value = None
        if s.time_horizon and len(s.time_horizon) >= 1:
            horizon_value = s.time_horizon[0]

        for strength in steering_strengths:
            choice = s.steered_choices.get(strength, "unknown")
            is_flip = (
                choice in ("short_term", "long_term") and choice != s.baseline_choice
            )
            is_degen = choice == "degenerate"

            data.append(
                {
                    "sample_id": s.sample_id,
                    "strength": strength,
                    "baseline": s.baseline_choice,
                    "choice": choice,
                    "is_flip": is_flip,
                    "is_degen": is_degen,
                    "reward_ratio": reward_ratio,
                    "horizon": horizon_value,
                    "short_reward": s.short_reward,
                    "long_reward": s.long_reward,
                }
            )

    if not data:
        return

    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Steering Analysis: {probe_id}", fontsize=14, fontweight="bold")

    # 1. Flip rate and degeneration rate by steering magnitude
    ax = axes[0, 0]
    strengths_sorted = sorted(set(d["strength"] for d in data))
    flip_by_strength = {s: 0 for s in strengths_sorted}
    degen_by_strength = {s: 0 for s in strengths_sorted}
    total_by_strength = {s: 0 for s in strengths_sorted}

    for d in data:
        total_by_strength[d["strength"]] += 1
        if d["is_flip"]:
            flip_by_strength[d["strength"]] += 1
        if d["is_degen"]:
            degen_by_strength[d["strength"]] += 1

    # Calculate rates as percentages
    flip_rates = [
        flip_by_strength[s] / total_by_strength[s] * 100 if total_by_strength[s] > 0 else 0
        for s in strengths_sorted
    ]
    degen_rates = [
        degen_by_strength[s] / total_by_strength[s] * 100 if total_by_strength[s] > 0 else 0
        for s in strengths_sorted
    ]

    x = range(len(strengths_sorted))
    width = 0.35
    bars1 = ax.bar(
        [i - width / 2 for i in x],
        flip_rates,
        width,
        label="Flip Rate",
        color="steelblue",
    )
    bars2 = ax.bar(
        [i + width / 2 for i in x],
        degen_rates,
        width,
        label="Degen Rate",
        color="crimson",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:+.0f}" for s in strengths_sorted], rotation=45, ha="right")
    ax.set_xlabel("Steering Strength")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Flip & Degeneration Rate by Steering Magnitude")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)

    # Add count labels on bars
    for bar, s in zip(bars1, strengths_sorted):
        if bar.get_height() > 0:
            ax.annotate(
                f"n={total_by_strength[s]}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    # 2. Flip rate by reward ratio (binned)
    ax = axes[0, 1]
    ratios_with_flips = [
        (d["reward_ratio"], d["is_flip"]) for d in data if d["reward_ratio"] is not None
    ]
    if ratios_with_flips:
        # Bin by reward ratio
        ratio_bins = [0, 2, 5, 10, 20, 50, float("inf")]
        bin_labels = ["<2", "2-5", "5-10", "10-20", "20-50", ">50"]
        flip_by_bin = {l: 0 for l in bin_labels}
        total_by_bin = {l: 0 for l in bin_labels}

        for ratio, is_flip in ratios_with_flips:
            for i, (lo, hi) in enumerate(zip(ratio_bins[:-1], ratio_bins[1:])):
                if lo <= ratio < hi:
                    label = bin_labels[i]
                    total_by_bin[label] += 1
                    if is_flip:
                        flip_by_bin[label] += 1
                    break

        non_empty = [l for l in bin_labels if total_by_bin[l] > 0]
        rates = [
            flip_by_bin[l] / total_by_bin[l] * 100 if total_by_bin[l] > 0 else 0
            for l in non_empty
        ]
        counts = [total_by_bin[l] for l in non_empty]

        bars = ax.bar(range(len(non_empty)), rates, color="steelblue")
        ax.set_xticks(range(len(non_empty)))
        ax.set_xticklabels(non_empty)
        ax.set_xlabel("Long/Short Reward Ratio")
        ax.set_ylabel("Flip Rate (%)")
        ax.set_title("Flip Rate by Reward Ratio")
        ax.set_ylim(0, 100)

        # Add count labels on bars
        for bar, count in zip(bars, counts):
            ax.annotate(
                f"n={count}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=8,
            )
    else:
        ax.text(
            0.5, 0.5, "No reward data", ha="center", va="center", transform=ax.transAxes
        )
        ax.set_title("Flip Rate by Reward Ratio")

    # 3. Flip rate by time horizon
    ax = axes[1, 0]
    horizons_with_flips = [
        (d["horizon"], d["is_flip"]) for d in data if d["horizon"] is not None
    ]
    if horizons_with_flips:
        horizon_values = sorted(set(h for h, _ in horizons_with_flips))
        flip_by_horizon = {h: 0 for h in horizon_values}
        total_by_horizon = {h: 0 for h in horizon_values}

        for horizon, is_flip in horizons_with_flips:
            total_by_horizon[horizon] += 1
            if is_flip:
                flip_by_horizon[horizon] += 1

        rates = [
            flip_by_horizon[h] / total_by_horizon[h] * 100
            if total_by_horizon[h] > 0
            else 0
            for h in horizon_values
        ]
        counts = [total_by_horizon[h] for h in horizon_values]

        bars = ax.bar(range(len(horizon_values)), rates, color="seagreen")
        ax.set_xticks(range(len(horizon_values)))
        ax.set_xticklabels([str(h) for h in horizon_values])
        ax.set_xlabel("Time Horizon")
        ax.set_ylabel("Flip Rate (%)")
        ax.set_title("Flip Rate by Time Horizon")
        ax.set_ylim(0, 100)

        for bar, count in zip(bars, counts):
            ax.annotate(
                f"n={count}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=8,
            )
    else:
        ax.text(
            0.5,
            0.5,
            "No horizon data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Flip Rate by Time Horizon")

    # 4. Summary statistics
    ax = axes[1, 1]
    ax.axis("off")

    total_samples = len(set(d["sample_id"] for d in data))
    total_flips = sum(1 for d in data if d["is_flip"])
    total_degen = sum(1 for d in data if d["is_degen"])
    total_trials = len(data)

    # Flips by direction
    sl_flips = sum(1 for d in data if d["is_flip"] and d["baseline"] == "short_term")
    ls_flips = sum(1 for d in data if d["is_flip"] and d["baseline"] == "long_term")

    summary_text = f"""
Summary Statistics
{"=" * 40}

Probe: {probe_id}
Layer: {result.layer}, Position: {result.token_position_idx}

Samples tested: {total_samples}
Total trials: {total_trials}

Flips: {total_flips} ({total_flips / total_trials * 100:.1f}%)
  - Short→Long: {sl_flips}
  - Long→Short: {ls_flips}

Degenerates: {total_degen} ({total_degen / total_trials * 100:.1f}%)

Steering strengths tested: {len(steering_strengths)}
Range: [{min(steering_strengths):+.0f}, {max(steering_strengths):+.0f}]
"""

    ax.text(
        0.1,
        0.9,
        summary_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(viz_dir / "analysis.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def create_steering_effects_by_baseline(
    output: "SteeringExperimentOutput",
    output_dir: Path,
    probe_types: Optional[list[str]] = None,
) -> None:
    """
    Create steering effects plots grouped by baseline type (short_term vs long_term).

    Shows separate bars/lines for samples that started with short-term vs long-term choice.
    """
    import matplotlib.pyplot as plt

    viz_dir = output_dir / "viz"
    ensure_dir(viz_dir)

    if probe_types is None:
        probe_types = ["choice", "time_horizon_category"]

    for probe_type_filter in probe_types:
        # Filter probes by type
        filtered_probes = [
            pr for pr in output.probe_results if pr.probe_type == probe_type_filter
        ]
        if not filtered_probes:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))
        strengths = sorted(output.steering_strengths)

        # Collect data by baseline type
        flip_rates_short = []  # Samples starting as short_term
        flip_rates_long = []   # Samples starting as long_term
        degen_rates_short = []
        degen_rates_long = []
        n_short = 0
        n_long = 0

        for s_idx, strength in enumerate(strengths):
            flips_short, total_short = 0, 0
            flips_long, total_long = 0, 0
            degen_short, degen_long = 0, 0

            for pr in filtered_probes:
                for sample in pr.samples:
                    choice = sample.steered_choices.get(strength, "unknown")

                    if sample.baseline_choice == "short_term":
                        total_short += 1
                        if choice == "degenerate":
                            degen_short += 1
                        elif choice == "long_term":
                            flips_short += 1
                    elif sample.baseline_choice == "long_term":
                        total_long += 1
                        if choice == "degenerate":
                            degen_long += 1
                        elif choice == "short_term":
                            flips_long += 1

            if s_idx == 0:
                n_short = total_short
                n_long = total_long

            flip_rates_short.append(
                flips_short / total_short * 100 if total_short > 0 else 0
            )
            flip_rates_long.append(
                flips_long / total_long * 100 if total_long > 0 else 0
            )
            degen_rates_short.append(
                degen_short / total_short * 100 if total_short > 0 else 0
            )
            degen_rates_long.append(
                degen_long / total_long * 100 if total_long > 0 else 0
            )

        x = range(len(strengths))
        width = 0.2

        # Plot flip rates
        ax.bar(
            [i - width * 1.5 for i in x],
            flip_rates_short,
            width,
            label=f"Flip S→L (n={n_short})",
            color="steelblue",
        )
        ax.bar(
            [i - width * 0.5 for i in x],
            flip_rates_long,
            width,
            label=f"Flip L→S (n={n_long})",
            color="darkorange",
        )
        # Plot degen rates
        ax.bar(
            [i + width * 0.5 for i in x],
            degen_rates_short,
            width,
            label="Degen (S baseline)",
            color="lightcoral",
            alpha=0.7,
        )
        ax.bar(
            [i + width * 1.5 for i in x],
            degen_rates_long,
            width,
            label="Degen (L baseline)",
            color="salmon",
            alpha=0.7,
        )

        ax.set_xticks(x)
        ax.set_xticklabels([f"{s:+.0f}" for s in strengths], rotation=45, ha="right")
        ax.set_xlabel("Steering Strength")
        ax.set_ylabel("Rate (%)")
        ax.set_title(f"Steering Effects by Baseline Type [{probe_type_filter}]")
        ax.set_ylim(0, 100)
        ax.legend(loc="upper left")
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        type_viz_dir = viz_dir / probe_type_filter
        ensure_dir(type_viz_dir)
        plt.savefig(type_viz_dir / "steering_by_baseline.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved: {type_viz_dir / 'steering_by_baseline.png'}")


def create_steering_effects_by_horizon_category(
    output: "SteeringExperimentOutput",
    output_dir: Path,
    probe_types: Optional[list[str]] = None,
) -> None:
    """
    Create steering effects plots grouped by time horizon category.

    Categories: < 1 year, 1-5 years, 5-10 years, > 10 years
    """
    import matplotlib.pyplot as plt

    viz_dir = output_dir / "viz"
    ensure_dir(viz_dir)

    if probe_types is None:
        probe_types = ["choice", "time_horizon_category"]

    # Define horizon bins (in years)
    horizon_bins = [(0, 1), (1, 5), (5, 10), (10, float("inf"))]
    bin_labels = ["< 1 year", "1-5 years", "5-10 years", "> 10 years"]

    def get_horizon_bin(horizon_value: Optional[float]) -> Optional[str]:
        if horizon_value is None:
            return None
        for (lo, hi), label in zip(horizon_bins, bin_labels):
            if lo <= horizon_value < hi:
                return label
        return None

    for probe_type_filter in probe_types:
        filtered_probes = [
            pr for pr in output.probe_results if pr.probe_type == probe_type_filter
        ]
        if not filtered_probes:
            continue

        fig, ax = plt.subplots(figsize=(14, 6))
        strengths = sorted(output.steering_strengths)

        # Collect data by horizon category
        flip_rates_by_cat = {label: [] for label in bin_labels}
        degen_rates_by_cat = {label: [] for label in bin_labels}
        n_by_cat = {label: 0 for label in bin_labels}

        for s_idx, strength in enumerate(strengths):
            flips_by_cat = {label: 0 for label in bin_labels}
            total_by_cat = {label: 0 for label in bin_labels}
            degen_by_cat = {label: 0 for label in bin_labels}

            for pr in filtered_probes:
                for sample in pr.samples:
                    if sample.baseline_choice not in ("short_term", "long_term"):
                        continue

                    # Get horizon value
                    horizon_value = None
                    if sample.time_horizon and len(sample.time_horizon) >= 1:
                        horizon_value = sample.time_horizon[0]

                    cat = get_horizon_bin(horizon_value)
                    if cat is None:
                        continue

                    choice = sample.steered_choices.get(strength, "unknown")
                    total_by_cat[cat] += 1

                    if choice == "degenerate":
                        degen_by_cat[cat] += 1
                    elif choice in ("short_term", "long_term") and choice != sample.baseline_choice:
                        flips_by_cat[cat] += 1

            if s_idx == 0:
                n_by_cat = {cat: total_by_cat[cat] for cat in bin_labels}

            for cat in bin_labels:
                flip_rates_by_cat[cat].append(
                    flips_by_cat[cat] / total_by_cat[cat] * 100
                    if total_by_cat[cat] > 0
                    else 0
                )
                degen_rates_by_cat[cat].append(
                    degen_by_cat[cat] / total_by_cat[cat] * 100
                    if total_by_cat[cat] > 0
                    else 0
                )

        # Plot as grouped bars
        x = range(len(strengths))
        n_cats = len(bin_labels)
        width = 0.8 / n_cats
        colors = ["steelblue", "seagreen", "darkorange", "crimson"]

        for i, (cat, color) in enumerate(zip(bin_labels, colors)):
            if n_by_cat[cat] == 0:
                continue
            offset = (i - n_cats / 2 + 0.5) * width
            ax.bar(
                [xi + offset for xi in x],
                flip_rates_by_cat[cat],
                width,
                label=f"{cat} (n={n_by_cat[cat]})",
                color=color,
                alpha=0.8,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([f"{s:+.0f}" for s in strengths], rotation=45, ha="right")
        ax.set_xlabel("Steering Strength")
        ax.set_ylabel("Flip Rate (%)")
        ax.set_title(f"Steering Effects by Time Horizon Category [{probe_type_filter}]")
        ax.set_ylim(0, 100)
        ax.legend(loc="upper left")
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        type_viz_dir = viz_dir / probe_type_filter
        ensure_dir(type_viz_dir)
        plt.savefig(type_viz_dir / "steering_by_horizon.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved: {type_viz_dir / 'steering_by_horizon.png'}")


def create_flip_rate_vs_reward_ratio(
    output: "SteeringExperimentOutput",
    output_dir: Path,
    probe_types: Optional[list[str]] = None,
) -> None:
    """
    Create scatter/line plot showing relationship between flip rate and reward ratio.

    X-axis: Reward ratio (long_reward / short_reward)
    Y-axis: Flip rate at each steering strength
    """
    import matplotlib.pyplot as plt

    viz_dir = output_dir / "viz"
    ensure_dir(viz_dir)

    if probe_types is None:
        probe_types = ["choice", "time_horizon_category"]

    # Define reward ratio bins
    ratio_bins = [(0, 1.5), (1.5, 3), (3, 6), (6, 15), (15, float("inf"))]
    bin_labels = ["1-1.5x", "1.5-3x", "3-6x", "6-15x", ">15x"]

    def get_ratio_bin(ratio: Optional[float]) -> Optional[str]:
        if ratio is None or ratio <= 0:
            return None
        for (lo, hi), label in zip(ratio_bins, bin_labels):
            if lo <= ratio < hi:
                return label
        return None

    for probe_type_filter in probe_types:
        filtered_probes = [
            pr for pr in output.probe_results if pr.probe_type == probe_type_filter
        ]
        if not filtered_probes:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left plot: Flip rate by reward ratio bin for each steering strength
        ax = axes[0]
        strengths = sorted(output.steering_strengths)
        # Use only a few representative strengths for clarity
        key_strengths = [
            s for s in strengths if abs(s) in [50, 100, 150, 200] or s == min(strengths) or s == max(strengths)
        ]
        if not key_strengths:
            key_strengths = strengths[:4] + strengths[-4:] if len(strengths) > 8 else strengths
        key_strengths = sorted(set(key_strengths))

        colors = plt.cm.viridis([i / len(key_strengths) for i in range(len(key_strengths))])

        for strength_idx, strength in enumerate(key_strengths):
            flip_by_bin = {label: 0 for label in bin_labels}
            total_by_bin = {label: 0 for label in bin_labels}

            for pr in filtered_probes:
                for sample in pr.samples:
                    if sample.baseline_choice not in ("short_term", "long_term"):
                        continue
                    if not sample.short_reward or not sample.long_reward:
                        continue

                    ratio = sample.long_reward / sample.short_reward if sample.short_reward > 0 else None
                    bin_label = get_ratio_bin(ratio)
                    if bin_label is None:
                        continue

                    choice = sample.steered_choices.get(strength, "unknown")
                    total_by_bin[bin_label] += 1
                    if choice in ("short_term", "long_term") and choice != sample.baseline_choice:
                        flip_by_bin[bin_label] += 1

            non_empty = [l for l in bin_labels if total_by_bin[l] > 0]
            rates = [
                flip_by_bin[l] / total_by_bin[l] * 100 if total_by_bin[l] > 0 else 0
                for l in non_empty
            ]

            if non_empty:
                ax.plot(
                    range(len(non_empty)),
                    rates,
                    marker="o",
                    label=f"s={strength:+.0f}",
                    color=colors[strength_idx],
                )

        ax.set_xticks(range(len(bin_labels)))
        ax.set_xticklabels(bin_labels)
        ax.set_xlabel("Long/Short Reward Ratio")
        ax.set_ylabel("Flip Rate (%)")
        ax.set_title(f"Flip Rate by Reward Ratio [{probe_type_filter}]")
        ax.set_ylim(0, 100)
        ax.legend(loc="upper right", fontsize=8)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)

        # Right plot: Heatmap of flip rate (strength x ratio_bin)
        ax = axes[1]
        flip_matrix = []
        n_matrix = []

        for strength in strengths:
            row_flips = []
            row_n = []
            for label in bin_labels:
                flip_count = 0
                total_count = 0

                for pr in filtered_probes:
                    for sample in pr.samples:
                        if sample.baseline_choice not in ("short_term", "long_term"):
                            continue
                        if not sample.short_reward or not sample.long_reward:
                            continue

                        ratio = sample.long_reward / sample.short_reward if sample.short_reward > 0 else None
                        bin_label = get_ratio_bin(ratio)
                        if bin_label != label:
                            continue

                        choice = sample.steered_choices.get(strength, "unknown")
                        total_count += 1
                        if choice in ("short_term", "long_term") and choice != sample.baseline_choice:
                            flip_count += 1

                rate = flip_count / total_count * 100 if total_count > 0 else float("nan")
                row_flips.append(rate)
                row_n.append(total_count)
            flip_matrix.append(row_flips)
            n_matrix.append(row_n)

        import numpy as np
        flip_array = np.array(flip_matrix)
        im = ax.imshow(flip_array, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=100)
        ax.set_xticks(range(len(bin_labels)))
        ax.set_xticklabels(bin_labels)
        ax.set_yticks(range(len(strengths)))
        ax.set_yticklabels([f"{s:+.0f}" for s in strengths])
        ax.set_xlabel("Long/Short Reward Ratio")
        ax.set_ylabel("Steering Strength")
        ax.set_title(f"Flip Rate Heatmap [{probe_type_filter}]")
        plt.colorbar(im, ax=ax, label="Flip Rate (%)")

        plt.tight_layout()
        type_viz_dir = viz_dir / probe_type_filter
        ensure_dir(type_viz_dir)
        plt.savefig(type_viz_dir / "flip_vs_reward.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved: {type_viz_dir / 'flip_vs_reward.png'}")


def create_steering_effects_suite(
    output: "SteeringExperimentOutput",
    output_dir: Path,
    probe_types: Optional[list[str]] = None,
) -> None:
    """
    Create comprehensive steering effects plots for each probe/steering vector.

    Generates the following plots in viz/{probe_type}/layer_{L}_position_{P}/:
    1. steering_effects_all.png - All data combined
    2. steering_effects_short_baseline.png - Short-term baseline only
    3. steering_effects_long_baseline.png - Long-term baseline only
    4. steering_effects_horizon_le1yr.png - Horizon <= 1 year
    5. steering_effects_horizon_gt1yr.png - Horizon > 1 year
    6. flip_direction_counts.png - Bar plot with L→S and S→L flip counts per magnitude
    """
    import matplotlib.pyplot as plt

    viz_dir = output_dir / "viz"
    ensure_dir(viz_dir)

    if probe_types is None:
        probe_types = ["choice", "time_horizon_category"]

    strengths = sorted(output.steering_strengths)

    for pr in output.probe_results:
        ptype = pr.probe_type if isinstance(pr.probe_type, str) else pr.probe_type.value
        if ptype not in probe_types:
            continue

        probe_viz_dir = get_probe_viz_dir(viz_dir, pr)

        # Collect sample data for this probe
        all_samples = []
        for sample in pr.samples:
            horizon_value = None
            if sample.time_horizon and len(sample.time_horizon) >= 1:
                horizon_value = sample.time_horizon[0]

            all_samples.append({
                "sample": sample,
                "baseline": sample.baseline_choice,
                "horizon": horizon_value,
            })

        def create_steering_bar_plot(
            samples: list,
            title: str,
            filename: str,
        ) -> None:
            """Create a steering effects bar plot for a subset of samples."""
            if not samples:
                return

            fig, ax = plt.subplots(figsize=(10, 6))

            flip_counts = []
            unflip_counts = []
            failed_counts = []

            for strength in strengths:
                flips = 0
                unflipped = 0
                failed = 0
                for s_data in samples:
                    sample = s_data["sample"]
                    choice = sample.steered_choices.get(strength, "unknown")
                    if choice in ("degenerate", "unknown"):
                        failed += 1
                    elif choice in ("short_term", "long_term"):
                        if choice != sample.baseline_choice:
                            flips += 1
                        else:
                            unflipped += 1
                    else:
                        failed += 1
                flip_counts.append(flips)
                unflip_counts.append(unflipped)
                failed_counts.append(failed)

            x = range(len(strengths))
            width = 0.25

            ax.bar([i - width for i in x], flip_counts, width, label="Flipped", color="steelblue")
            ax.bar(x, unflip_counts, width, label="Unflipped", color="gray")
            ax.bar([i + width for i in x], failed_counts, width, label="Failed", color="crimson")

            ax.set_xticks(x)
            ax.set_xticklabels([f"{s:+.0f}" for s in strengths], rotation=45, ha="right")
            ax.set_xlabel("Steering Strength")
            ax.set_ylabel("Count")
            ax.set_title(f"{title}\n{pr.probe_id} (n={len(samples)})")
            ax.legend()
            ax.yaxis.grid(True, linestyle="--", alpha=0.7)

            plt.tight_layout()
            plt.savefig(probe_viz_dir / filename, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close()
            print(f"  Saved: {probe_viz_dir / filename}")

        # 1. All data combined
        create_steering_bar_plot(
            all_samples,
            "Steering Effects: All Data",
            "steering_effects_all.png",
        )

        # 2. Short-term baseline only
        short_baseline = [s for s in all_samples if s["baseline"] == "short_term"]
        create_steering_bar_plot(
            short_baseline,
            "Steering Effects: Short-Term Baseline",
            "steering_effects_short_baseline.png",
        )

        # 3. Long-term baseline only
        long_baseline = [s for s in all_samples if s["baseline"] == "long_term"]
        create_steering_bar_plot(
            long_baseline,
            "Steering Effects: Long-Term Baseline",
            "steering_effects_long_baseline.png",
        )

        # 4. Horizon <= 1 year
        horizon_le1yr = [s for s in all_samples if s["horizon"] is not None and s["horizon"] <= 1]
        create_steering_bar_plot(
            horizon_le1yr,
            "Steering Effects: Horizon ≤ 1 Year",
            "steering_effects_horizon_le1yr.png",
        )

        # 5. Horizon > 1 year
        horizon_gt1yr = [s for s in all_samples if s["horizon"] is not None and s["horizon"] > 1]
        create_steering_bar_plot(
            horizon_gt1yr,
            "Steering Effects: Horizon > 1 Year",
            "steering_effects_horizon_gt1yr.png",
        )

        # 6. Flip direction counts (L→S and S→L)
        fig, ax = plt.subplots(figsize=(10, 6))

        ls_flip_counts = []  # Long → Short flips
        sl_flip_counts = []  # Short → Long flips

        for strength in strengths:
            ls_flips = 0
            sl_flips = 0
            for s_data in all_samples:
                sample = s_data["sample"]
                choice = sample.steered_choices.get(strength, "unknown")
                if choice in ("short_term", "long_term") and choice != sample.baseline_choice:
                    if sample.baseline_choice == "long_term" and choice == "short_term":
                        ls_flips += 1
                    elif sample.baseline_choice == "short_term" and choice == "long_term":
                        sl_flips += 1
            ls_flip_counts.append(ls_flips)
            sl_flip_counts.append(sl_flips)

        x = range(len(strengths))
        width = 0.35

        ax.bar([i - width/2 for i in x], ls_flip_counts, width, label="Long→Short", color="darkorange")
        ax.bar([i + width/2 for i in x], sl_flip_counts, width, label="Short→Long", color="steelblue")

        ax.set_xticks(x)
        ax.set_xticklabels([f"{s:+.0f}" for s in strengths], rotation=45, ha="right")
        ax.set_xlabel("Steering Strength")
        ax.set_ylabel("Flip Count")
        ax.set_title(f"Flip Direction by Steering Magnitude\n{pr.probe_id} (n={len(all_samples)})")
        ax.legend()
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.savefig(probe_viz_dir / "flip_direction_counts.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved: {probe_viz_dir / 'flip_direction_counts.png'}")


def create_all_steering_effect_plots(
    output: "SteeringExperimentOutput",
    output_dir: Path,
    probe_types: Optional[list[str]] = None,
) -> None:
    """Create all steering effect visualization plots."""
    print("\n=== Creating Steering Effect Plots ===")

    # Determine which probe types we have
    if probe_types is None:
        available_types = set(pr.probe_type for pr in output.probe_results)
        probe_types = [t for t in ["choice", "time_horizon_category"] if t in available_types]

    if not probe_types:
        print("  No matching probe types found, skipping plots")
        return

    print(f"  Probe types: {probe_types}")

    # Create per-probe steering effects plots (count-based)
    create_steering_effects_suite(output, output_dir, probe_types)

    # Create aggregate flip rate vs reward ratio plot
    create_flip_rate_vs_reward_ratio(output, output_dir, probe_types)


# =============================================================================
# Output Serialization
# =============================================================================


def create_debug_summary(output: SteeringExperimentOutput, output_dir: Path) -> dict:
    """
    Create debug_steering.json with summary of steering results.

    Shows at a glance whether steering worked, failed samples, invalid choices, etc.
    """
    summary = {
        "steering_id": output.steering_id,
        "dataset_id": output.dataset_id,
        "model_name": output.model_name,
        "n_samples": output.n_samples,
        "sample_indices": output.sample_indices,
        "steering_strengths": output.steering_strengths,
        "probes_tested": len(output.probe_results),
        "probe_summaries": [],
    }
    if output.probes_id:
        summary["probes_id"] = output.probes_id
    if output.contrastive_id:
        summary["contrastive_id"] = output.contrastive_id

    total_flips = 0
    total_failed = 0
    total_invalid_baseline = 0

    for pr in output.probe_results:
        # Count outcomes for this probe
        flips_sl = 0  # short->long
        flips_ls = 0  # long->short
        failed = 0  # degenerate output
        invalid_baseline = 0  # unknown baseline choice
        valid_baseline = 0

        failed_samples = []
        invalid_samples = []
        flipped_samples = []

        for s in pr.samples:
            if s.baseline_choice == "unknown":
                invalid_baseline += 1
                invalid_samples.append(s.sample_id)
            elif s.baseline_choice in ("short_term", "long_term"):
                valid_baseline += 1

            # Check for degenerate outputs
            has_degenerate = any(
                choice == "degenerate" for choice in s.steered_choices.values()
            )
            if has_degenerate:
                failed += 1
                failed_samples.append(s.sample_id)

            # Check for flips
            if s.min_flip_magnitude_to_long is not None:
                flips_sl += 1
                flipped_samples.append(
                    {
                        "sample_id": s.sample_id,
                        "direction": "short->long",
                        "magnitude": s.min_flip_magnitude_to_long,
                    }
                )
            if s.min_flip_magnitude_to_short is not None:
                flips_ls += 1
                flipped_samples.append(
                    {
                        "sample_id": s.sample_id,
                        "direction": "long->short",
                        "magnitude": s.min_flip_magnitude_to_short,
                    }
                )

        probe_summary = {
            "probe_id": pr.probe_id,
            "layer": pr.layer,
            "position": pr.token_position_idx,
            "valid_baseline": valid_baseline,
            "invalid_baseline": invalid_baseline,
            "failed_degenerate": failed,
            "flips_short_to_long": flips_sl,
            "flips_long_to_short": flips_ls,
            "total_flips": flips_sl + flips_ls,
            "steering_worked": (flips_sl + flips_ls) > 0,
        }

        # Include sample IDs for failed/invalid if any
        if failed_samples:
            probe_summary["failed_sample_ids"] = failed_samples
        if invalid_samples:
            probe_summary["invalid_sample_ids"] = invalid_samples
        if flipped_samples:
            probe_summary["flipped_samples"] = flipped_samples

        summary["probe_summaries"].append(probe_summary)
        total_flips += flips_sl + flips_ls
        total_failed += failed
        total_invalid_baseline += invalid_baseline

    # Overall summary
    summary["overall"] = {
        "total_flips": total_flips,
        "total_failed": total_failed,
        "total_invalid_baseline": total_invalid_baseline,
        "any_steering_worked": total_flips > 0,
        "status": "SUCCESS"
        if total_flips > 0
        else (
            "FAILED"
            if total_failed > 0
            else (
                "INVALID" if total_invalid_baseline == output.n_samples else "NO_FLIPS"
            )
        ),
    }

    # Save to file
    debug_path = output_dir / "debug_steering.json"
    save_json(summary, debug_path)
    print(f"\nDebug summary saved to: {debug_path}")
    print(f"  Status: {summary['overall']['status']}")
    print(
        f"  Total flips: {total_flips}, Failed: {total_failed}, Invalid: {total_invalid_baseline}"
    )

    return summary


def serialize_probe_result(pr: ProbeSteeringResult) -> dict:
    """Serialize a single probe result to JSON-compatible format."""
    samples = []
    for s in pr.samples:
        sample_data = {
            "sample_id": s.sample_id,
            "time_horizon": s.time_horizon,
            "baseline_choice": s.baseline_choice,
            "steered_choices": {str(k): v for k, v in s.steered_choices.items()},
            "min_flip_to_long": s.min_flip_magnitude_to_long,
            "min_flip_to_short": s.min_flip_magnitude_to_short,
            "short_reward": s.short_reward,
            "long_reward": s.long_reward,
        }
        # Include failed_responses if present (for unknown/degenerate debugging)
        if s.failed_responses:
            sample_data["failed_responses"] = {
                str(k): v for k, v in s.failed_responses.items()
            }
        samples.append(sample_data)

    return {
        "probe_id": pr.probe_id,
        "probe_type": pr.probe_type,
        "layer": pr.layer,
        "token_position_idx": pr.token_position_idx,
        "after_horizon": pr.after_horizon,
        "samples": samples,
    }


@dataclass
class IncrementalSaver:
    """
    Saves steering results incrementally as probes complete.

    This allows for:
    - Progress persistence if interrupted
    - Lower memory usage (results written to disk)
    - Real-time visibility into experiment progress
    """

    results_dir: Path
    timestamp: str
    steering_id: str
    query_id: str
    dataset_id: str
    model_name: str
    steering_strengths: list[float]
    n_samples: int
    sample_indices: list[int]
    probes_id: Optional[str] = None
    contrastive_id: Optional[str] = None

    def __post_init__(self):
        ensure_dir(self.results_dir)
        self._results_path = self.results_dir / f"steering_{self.timestamp}.json"
        self._probe_results: list[dict] = []
        self._completed_probe_ids: set[str] = set()
        # Write initial partial file
        self._save()

    def _save(self) -> None:
        """Write current state to disk."""
        data = {
            "steering_id": self.steering_id,
            "query_id": self.query_id,
            "dataset_id": self.dataset_id,
            "model_name": self.model_name,
            "steering_strengths": self.steering_strengths,
            "n_samples": self.n_samples,
            "sample_indices": self.sample_indices,
            "timestamp": self.timestamp,
            "probe_results": self._probe_results,
            "_completed_probe_ids": list(self._completed_probe_ids),
        }
        if self.probes_id:
            data["probes_id"] = self.probes_id
        if self.contrastive_id:
            data["contrastive_id"] = self.contrastive_id
        save_json(data, self._results_path)

    def add_probe_result(self, result: ProbeSteeringResult) -> None:
        """Add and save a completed probe result."""
        serialized = serialize_probe_result(result)
        self._probe_results.append(serialized)
        self._completed_probe_ids.add(result.probe_id)
        self._save()

        # Save individual probe JSON file
        probe_dir = self.results_dir / "probes"
        ensure_dir(probe_dir)
        probe_file = probe_dir / f"{result.probe_id}.json"
        probe_data = {
            "probe_id": result.probe_id,
            "probe_type": result.probe_type,
            "layer": result.layer,
            "token_position_idx": result.token_position_idx,
            "after_horizon": result.after_horizon,
            "steering_strengths": self.steering_strengths,
            "steering_id": self.steering_id,
            "model_name": self.model_name,
            "samples": serialized["samples"],
        }
        save_json(probe_data, probe_file)

        # Create per-probe visualization
        viz_dir = self.results_dir / "viz"
        ensure_dir(viz_dir)
        probe_viz_dir = get_probe_viz_dir(viz_dir, result)
        create_probe_analysis_plots(result, self.steering_strengths, probe_viz_dir)

        print(
            f"    Saved result for {result.probe_id} ({len(self._probe_results)} probes complete)"
        )

    def is_completed(self, probe_id: str) -> bool:
        """Check if a probe has already been completed."""
        return probe_id in self._completed_probe_ids

    def get_results_path(self) -> Path:
        """Get the path where results are saved."""
        return self._results_path

    def get_probe_results(self) -> list[dict]:
        """Get all collected probe results."""
        return self._probe_results


def serialize_output(output: SteeringExperimentOutput) -> dict:
    """Serialize output to JSON-compatible format."""
    data = {
        "steering_id": output.steering_id,
        "dataset_id": output.dataset_id,
        "model_name": output.model_name,
        "steering_strengths": output.steering_strengths,
        "n_samples": output.n_samples,
        "sample_indices": output.sample_indices,
        "timestamp": output.timestamp,
        "probe_results": [serialize_probe_result(pr) for pr in output.probe_results],
    }
    if output.probes_id:
        data["probes_id"] = output.probes_id
    if output.contrastive_id:
        data["contrastive_id"] = output.contrastive_id
    return data


def print_summary(output: SteeringExperimentOutput) -> None:
    """Print summary of steering experiment."""
    print("\n" + "=" * 70)
    print("STEERING EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"\n>>> STEERING ID: {output.steering_id} <<<\n")
    print(f"Model: {output.model_name}")
    print(f"Dataset: {output.dataset_id}")
    print(f"Samples: {output.n_samples}")

    for pr in output.probe_results:
        print(f"\n{'-' * 60}")
        print(f"Probe: {pr.probe_id}")
        print(f"{'-' * 60}")

        # Count baselines
        baseline_counts = {"short_term": 0, "long_term": 0, "unknown": 0}
        for s in pr.samples:
            baseline_counts[s.baseline_choice] = (
                baseline_counts.get(s.baseline_choice, 0) + 1
            )

        print(
            f"Baseline: ST={baseline_counts['short_term']} LT={baseline_counts['long_term']}"
        )

        # Count flips
        sl_flips = sum(
            1 for s in pr.samples if s.min_flip_magnitude_to_long is not None
        )
        ls_flips = sum(
            1 for s in pr.samples if s.min_flip_magnitude_to_short is not None
        )
        print(
            f"Flips: Short→Long={sl_flips}/{baseline_counts['short_term']} "
            f"Long→Short={ls_flips}/{baseline_counts['long_term']}"
        )

        # Count unknown/degenerate outcomes across all strengths
        unknown_count = 0
        degenerate_count = 0
        total_steered = 0
        for s in pr.samples:
            for strength, choice in s.steered_choices.items():
                if strength != 0:  # Exclude baseline
                    total_steered += 1
                    if choice == "unknown":
                        unknown_count += 1
                    elif choice == "degenerate":
                        degenerate_count += 1

        if total_steered > 0:
            print(
                f"Failures: unknown={unknown_count} ({100 * unknown_count / total_steered:.1f}%) "
                f"degenerate={degenerate_count} ({100 * degenerate_count / total_steered:.1f}%)"
            )

        # Average flip magnitude
        sl_mags = [
            s.min_flip_magnitude_to_long
            for s in pr.samples
            if s.min_flip_magnitude_to_long is not None
        ]
        ls_mags = [
            s.min_flip_magnitude_to_short
            for s in pr.samples
            if s.min_flip_magnitude_to_short is not None
        ]

        if sl_mags:
            print(f"  Avg S→L magnitude: {np.mean(sl_mags):.2f}")
        if ls_mags:
            print(f"  Avg L→S magnitude: {np.mean(ls_mags):.2f}")


# =============================================================================
# Load Existing Results
# =============================================================================


def load_existing_results(
    results_dir: Path,
) -> Optional[SteeringExperimentOutput]:
    """
    Load existing steering experiment results from JSON.

    Args:
        results_dir: Directory containing steering_*.json files

    Returns:
        SteeringExperimentOutput or None if not found
    """
    if not results_dir.exists():
        print(f"  No results directory found at {results_dir}")
        return None

    # Find most recent results file
    result_files = sorted(results_dir.glob("steering_*.json"), reverse=True)
    if not result_files:
        print(f"  No result files found in {results_dir}")
        return None

    result_path = result_files[0]
    print(f"  Loading existing results from: {result_path}")

    data = load_json(result_path)

    # Reconstruct probe results
    probe_results = []
    for pr_data in data.get("probe_results", []):
        samples = []
        for s_data in pr_data.get("samples", []):
            # Convert string keys back to floats
            steered_choices = {
                float(k): v for k, v in s_data.get("steered_choices", {}).items()
            }
            samples.append(
                SampleSteeringResult(
                    sample_id=s_data["sample_id"],
                    time_horizon=s_data.get("time_horizon"),
                    baseline_choice=s_data["baseline_choice"],
                    steered_choices=steered_choices,
                    min_flip_magnitude_to_long=s_data.get("min_flip_to_long"),
                    min_flip_magnitude_to_short=s_data.get("min_flip_to_short"),
                    short_reward=s_data.get("short_reward"),
                    long_reward=s_data.get("long_reward"),
                )
            )

        probe_results.append(
            ProbeSteeringResult(
                probe_id=pr_data["probe_id"],
                probe_type=pr_data["probe_type"],
                layer=pr_data["layer"],
                token_position_idx=pr_data["token_position_idx"],
                after_horizon=pr_data.get("after_horizon", False),
                samples=samples,
            )
        )

    # Support both old "probe_config_id" and new "probes_id"
    probes_id = data.get("probes_id") or data.get("probe_config_id")
    contrastive_id = data.get("contrastive_id")

    # Support both old "config_id" and new "steering_id"
    steering_id = data.get("steering_id") or data.get("config_id")
    return SteeringExperimentOutput(
        steering_id=steering_id,
        query_id=data.get("query_id", ""),  # Backwards compat
        dataset_id=data["dataset_id"],
        model_name=data["model_name"],
        steering_strengths=data["steering_strengths"],
        n_samples=data["n_samples"],
        sample_indices=data.get("sample_indices", []),  # Backwards compat
        probe_results=probe_results,
        timestamp=data["timestamp"],
        probes_id=probes_id,
        contrastive_id=contrastive_id,
    )


# =============================================================================
# Token Position Info for Visualization
# =============================================================================


def get_token_position_info_from_query_ids(
    query_ids: list[str],
) -> Optional[dict]:
    """
    Get token position info from query IDs.

    Loads the preference data and extracts token positions, actual tokens,
    and token position specs for visualization labels and text-based matching.

    Args:
        query_ids: List of query IDs to load from

    Returns:
        Dict with 'tokens', 'resolved_positions', and 'specs' mappings, or None if not available
    """
    from src.probes.data import (
        extract_token_position_specs,
        find_preference_data_by_query_id,
        load_preference_data_file,
    )

    if not query_ids:
        return None

    # Load first preference data file to get tokens
    path = find_preference_data_by_query_id(query_ids[0])
    if path is None:
        return None

    pref_data = load_preference_data_file(path)

    # Extract token position specs from query config
    query_config = pref_data.get("metadata", {}).get("query_config", {})
    token_specs = extract_token_position_specs(query_config)

    # Find first sample with internals containing tokens
    for pref in pref_data["preferences"]:
        internals = pref.get("internals", {})
        if internals.get("token_positions") and internals.get("tokens"):
            resolved_positions = internals["token_positions"]
            tokens_list = internals["tokens"]

            return {
                "tokens": {i: tok for i, tok in enumerate(tokens_list)},
                "resolved_positions": {
                    i: pos for i, pos in enumerate(resolved_positions)
                },
                "specs": token_specs,
            }

    return None


def get_token_position_info_from_probe_data(
    probes_id: str,
) -> Optional[dict]:
    """
    Get token position info from probe training data.

    Loads the preference data used to train the probes and extracts
    token positions, actual tokens, and token position specs for visualization labels.

    Returns:
        Dict with 'tokens', 'resolved_positions', and 'specs' mappings, or None if not available
    """
    # Load probe index to get train query IDs
    probes_dir = PROJECT_ROOT / "out" / "probes" / probes_id / "probes"
    index_path = probes_dir / "index.json"
    if not index_path.exists():
        return None

    index = load_json(index_path)
    data_info = index.get("data", {})
    query_ids = data_info.get("train_query_ids", [])

    return get_token_position_info_from_query_ids(query_ids)


def get_token_position_info_from_contrastive_data(
    contrastive_id: str,
) -> Optional[dict]:
    """
    Get token position info from contrastive (CAA) data.

    Loads the preference data used for CAA computation and extracts
    token positions, actual tokens, and token position specs for text-based matching.

    Returns:
        Dict with 'tokens', 'resolved_positions', and 'specs' mappings, or None if not available
    """
    # Load CAA results to get query IDs
    results_path = PROJECT_ROOT / "out" / "contrastive" / contrastive_id / "results" / "caa_results.json"
    if not results_path.exists():
        return None

    results = load_json(results_path)
    query_ids = results.get("query_ids", [])

    return get_token_position_info_from_query_ids(query_ids)


# =============================================================================
# CLI
# =============================================================================


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment with activation steering using trained probes"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default_steering",
        help="Steering config name from configs/steering/",
    )
    parser.add_argument(
        "--probes-id",
        type=str,
        default=None,
        help="Override probes config ID",
    )
    parser.add_argument(
        "--contrastive-id",
        type=str,
        default=None,
        help="Override contrastive (CAA) config ID",
    )
    parser.add_argument(
        "--query-id",
        type=str,
        default=None,
        help="Override query ID (references preference_data)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples to use (0 = all)",
    )
    parser.add_argument(
        "--sample-indices",
        type=str,
        default=None,
        help="Comma-separated sample IDs to use (e.g., '123,456,789')",
    )
    parser.add_argument(
        "--all-probes",
        action="store_true",
        help="Test all probes, not just after-horizon",
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse existing results from JSON and regenerate visualizations only",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: print responses that fail to parse",
    )
    parser.add_argument(
        "--starting-strength",
        type=float,
        default=None,
        help="Max steering magnitude (e.g., 500). Overrides config.",
    )
    return parser.parse_args()


def main() -> int:
    args = get_args()

    # Load config
    config_path = SCRIPTS_DIR / "configs" / "steering" / f"{args.config}.json"
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        return 1

    config = load_steering_config(config_path)

    # Apply overrides
    if args.probes_id:
        config.probes_id = args.probes_id
    if args.contrastive_id:
        config.contrastive_id = args.contrastive_id
    if args.query_id:
        config.query_id = args.query_id
    if args.max_samples is not None:
        config.max_samples = args.max_samples
    if args.sample_indices:
        # Parse comma-separated sample IDs
        config.sample_indices = [int(x.strip()) for x in args.sample_indices.split(",")]
    if args.all_probes:
        config.only_after_horizon = False
    if args.starting_strength is not None:
        config.starting_steering_strength = args.starting_strength

    steering_id = config.get_id()
    print("=" * 60)
    print("STEERING EXPERIMENT")
    print("=" * 60)
    print(f"\n>>> STEERING ID: {steering_id} <<<\n")
    if config.probes_id:
        print(f"Probes ID: {config.probes_id}")
    if config.contrastive_id:
        print(f"Contrastive ID: {config.contrastive_id}")
    print(f"Query ID: {config.query_id}")
    print(f"Starting strength: {config.starting_steering_strength}")
    print(f"Max samples: {config.max_samples} (0 = all)")
    if config.specific_probes:
        print(f"Specific probes: {len(config.specific_probes)} specs")
        for sp in config.specific_probes:
            print(f"  - layer={sp.layer}, token_pos={sp.token_pos}")
    if config.specific_contrastive:
        print(f"Specific contrastive: {len(config.specific_contrastive)} specs")
        for sp in config.specific_contrastive:
            print(f"  - layer={sp.layer}, token_pos={sp.token_pos}")
    if not config.specific_probes and not config.specific_contrastive:
        print(f"Only after horizon: {config.only_after_horizon}")
        print(f"Only before choice: {config.only_before_choice}")
    print(f"Reuse existing: {args.reuse}")

    # Setup output directory
    output_base = PROJECT_ROOT / "out" / "steering"
    output_dir = output_base / config.get_id()
    ensure_dir(output_dir)
    results_dir = output_dir / "results"

    # Load probe index for visualizations (only if probes_id is set)
    probe_index = None
    if config.probes_id:
        probes_dir = PROJECT_ROOT / "out" / "probes" / config.probes_id / "probes"
        index_path = probes_dir / "index.json"
        if index_path.exists():
            probe_index = load_json(index_path)

    # Get token info for proper x-axis labels
    token_info = None
    if config.probes_id:
        token_info = get_token_position_info_from_probe_data(config.probes_id)
    if token_info:
        print(f"  Loaded token info: {len(token_info.get('tokens', {}))} positions")

    # Run experiment
    output = None
    if args.reuse:
        print("\nLooking for existing results...")
        output = load_existing_results(results_dir)
        if output:
            print(f"  Loaded {len(output.probe_results)} probe results")
        else:
            print("  No existing results found, will run experiment")

    # Run experiment if not reusing or no results found
    if output is None:
        output = run_steering_experiment(
            config,
            debug=args.debug,
            output_dir=output_dir,
        )

        # Print summary (results already saved incrementally by run_steering_experiment)
        print_summary(output)
    else:
        # Print summary for loaded results
        print_summary(output)

    # Always create debug summary
    create_debug_summary(output, output_dir)

    # Create visualizations
    print("\nCreating visualizations...")
    create_steering_heatmaps(
        output, output_dir, probe_index=probe_index, token_info=token_info
    )
    create_summary_plots(output, output_dir)

    # Create per-probe analysis plots
    viz_dir = output_dir / "viz"
    for pr in output.probe_results:
        probe_viz_dir = get_probe_viz_dir(viz_dir, pr)
        create_probe_analysis_plots(pr, output.steering_strengths, probe_viz_dir)

    create_all_steering_effect_plots(output, output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
