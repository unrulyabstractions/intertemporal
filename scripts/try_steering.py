#!/usr/bin/env python
"""
Experiment with activation steering using trained probes.

Uses probe directions to steer model behavior toward short-term or long-term choices.
Tests steering on dataset samples and measures the minimum steering magnitude
needed to flip choices.

Two modes:
1. Grid search (default when random_search=false): Tests all probes systematically
2. Random search (when random_search=true): Randomly samples (layer, position, sample,
   strength) combinations with adaptive biasing towards configurations that produce flips

Creates heatmaps showing:
- Min steering magnitude to flip choice (per layer/position)
- Separate plots for short->long and long->short flips
- Gray cells for "never flipped", black "FAIL" for degeneration
- White cells for "not tested" (quick mode)

Config file options:
    {
      "probe_config_id": "7c7dfcf3d544239e7badadfd89859c38",
      "dataset_id": "c35e217b8473d84a41b79c38e5b3c059",
      "probe_types": ["choice"],
      "steering_strengths": [-5.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 5.0],
      "n_samples": 10,
      "sample_indices": [123, 456, 789],  // Optional: specific sample IDs
      "only_after_horizon": true,
      "subsample_probes": 0.25,  // Fraction of probes to use (0.0-1.0)
      "random_search": true,  // Enable random parameter search
      "random_search_iterations": 100  // Number of random search iterations
    }

Outputs:
    - results/steering_*.json: Full results with all sample data (grid search)
    - results/random_search_*.json: Results from random search mode
    - debug_steering.json: Quick summary showing if steering worked, failures, etc.
    - viz/random_search_short_to_long.png: Min magnitude to flip short→long (random search)
    - viz/random_search_long_to_short.png: Min magnitude to flip long→short (random search)
    - viz/steering_short_to_long.png: Min magnitude heatmap (grid search)
    - viz/steering_long_to_short.png: Min magnitude heatmap (grid search)

Usage:
    python scripts/try_steering.py
    python scripts/try_steering.py --config default_steering
    python scripts/try_steering.py --all-probes
    python scripts/try_steering.py --quick  # Fast test with best probe only
    python scripts/try_steering.py --random-search --iterations 50  # Random search
    python scripts/try_steering.py --no-random-search  # Force grid search
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Add project root and scripts to path
PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from src.common.io import ensure_dir, get_timestamp, load_json, save_json
from src.common.schema_utils import SchemaClass
from src.model_runner import ModelRunner
from src.probes import (
    LoadedProbe,
    load_probes_from_dir,
)
from src.steering import SteeringConfig as SteeringDirectionConfig, SteeringOption

from common import (
    build_prompt_from_question,
    clear_memory,
    determine_choice,
    load_dataset_output,
    parse_label_from_response,
    DatasetOutput,
    QuestionOutput,
)


# =============================================================================
# Steering Configuration Schema
# =============================================================================


@dataclass
class SteeringConfigSchema(SchemaClass):
    """Schema for steering experiment config - used for deterministic folder IDs."""

    probe_config_id: str
    dataset_id: str
    probe_types: tuple[str, ...]
    steering_strengths: tuple[float, ...]
    n_samples: int
    only_after_horizon: bool
    subsample_probes: float
    random_search: bool
    random_search_iterations: int
    # Note: sample_indices not included in schema - it's for reproducibility, not identity


@dataclass
class SteeringConfig:
    """Configuration for steering experiment.

    Attributes:
        sample_indices: If specified, use these exact sample IDs from the dataset
                       instead of random sampling. Takes precedence over n_samples.
        subsample_probes: Fraction of probes to use (0.0-1.0). Default 1.0 uses all.
                         Probes are sampled randomly but deterministically (seed=42).
        random_search: If True, use random parameter search instead of grid search.
        random_search_iterations: Number of random search iterations to run.
    """

    probe_config_id: str
    dataset_id: str
    probe_types: list[str] = field(default_factory=lambda: ["choice"])
    steering_strengths: list[float] = field(
        default_factory=lambda: [-5.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 5.0]
    )
    n_samples: int = 10
    only_after_horizon: bool = True  # Only test probes after time horizon
    subsample_probes: float = 1.0  # Fraction of probes to use (0.0-1.0)
    sample_indices: Optional[list[int]] = None  # Specific sample IDs to use
    random_search: bool = False  # Enable random parameter search
    random_search_iterations: int = 100  # Number of iterations for random search

    def get_schema(self) -> SteeringConfigSchema:
        """Convert to schema for ID generation."""
        return SteeringConfigSchema(
            probe_config_id=self.probe_config_id,
            dataset_id=self.dataset_id,
            probe_types=tuple(sorted(self.probe_types)),
            steering_strengths=tuple(sorted(self.steering_strengths)),
            n_samples=self.n_samples,
            only_after_horizon=self.only_after_horizon,
            subsample_probes=self.subsample_probes,
            random_search=self.random_search,
            random_search_iterations=self.random_search_iterations,
        )

    def get_id(self) -> str:
        """Get deterministic config ID."""
        return self.get_schema().get_id()


def load_steering_config(path: Path) -> SteeringConfig:
    """Load steering config from JSON file."""
    data = load_json(path)
    return SteeringConfig(
        probe_config_id=data["probe_config_id"],
        dataset_id=data["dataset_id"],
        probe_types=data.get("probe_types", ["choice"]),
        steering_strengths=data.get(
            "steering_strengths", [-5.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 5.0]
        ),
        n_samples=data.get("n_samples", 10),
        only_after_horizon=data.get("only_after_horizon", True),
        subsample_probes=data.get("subsample_probes", 1.0),
        sample_indices=data.get("sample_indices"),  # None = random sampling
        random_search=data.get("random_search", False),
        random_search_iterations=data.get("random_search_iterations", 100),
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

    config_id: str
    probe_config_id: str
    dataset_id: str
    model_name: str
    steering_strengths: list[float]
    n_samples: int
    sample_indices: list[int]  # Actual sample IDs used (for reproducibility)
    probe_results: list[ProbeSteeringResult]
    timestamp: str


# =============================================================================
# Dataset Loading
# =============================================================================


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
        print(f"Using {len(selected)} specified samples: {[q.sample_id for q in selected]}")
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
            short_numbers = [n for n in re.findall(r'[\d,]+', short_value) if len(n) >= 3]
            long_numbers = [n for n in re.findall(r'[\d,]+', long_value) if len(n) >= 3]

            short_found = any(num in response for num in short_numbers)
            long_found = any(num in response for num in long_numbers)

            if short_found and not long_found:
                if debug:
                    print(f"      [FALLBACK] Detected short_term by value: {short_numbers}")
                return "short_term"
            elif long_found and not short_found:
                if debug:
                    print(f"      [FALLBACK] Detected long_term by value: {long_numbers}")
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


def run_steering_for_probe(
    runner: ModelRunner,
    probe: LoadedProbe,
    questions: list[QuestionOutput],
    steering_strengths: list[float],
    max_new_tokens: int = 64,
    debug: bool = False,
) -> ProbeSteeringResult:
    """Run steering experiment for a single probe using ModelRunner."""
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

    samples = []

    for i, question in enumerate(questions):
        if (i + 1) % 5 == 0 or i == 0:
            print(f"    Sample {i + 1}/{len(questions)}")

        prompt = build_prompt_from_question(question)
        pair = question.preference_pair
        short_label = pair.short_term.label
        long_label = pair.long_term.label
        # Get reward values for fallback parsing (format as string with commas)
        short_value = f"{pair.short_term.reward:,.0f}"
        long_value = f"{pair.long_term.reward:,.0f}"

        if debug:
            print(f"      Labels: short={repr(short_label)}, long={repr(long_label)}")

        # Generate baseline
        baseline_response = runner.generate_baseline(prompt, max_new_tokens)
        baseline_choice = parse_choice_from_response(
            baseline_response, short_label, long_label, debug=debug,
            short_value=short_value, long_value=long_value
        )
        if debug:
            print(f"      Baseline: choice={baseline_choice}")
            print(f"        Response: {repr(baseline_response[:100])}")

        # Generate with steering at various strengths
        steered_choices = {}
        for strength in steering_strengths:
            if strength == 0:
                steered_choices[strength] = baseline_choice
            else:
                # Create steering config for this probe/strength
                steering_config = SteeringDirectionConfig(
                    direction=probe.direction,
                    layer=layer,
                    strength=strength,
                    option=SteeringOption.APPLY_TO_ALL,
                )
                response = runner.generate_with_steering(
                    prompt,
                    steering=steering_config,
                    max_new_tokens=max_new_tokens,
                )
                choice = parse_choice_from_response(
                    response, short_label, long_label, debug=debug,
                    short_value=short_value, long_value=long_value
                )
                steered_choices[strength] = choice

                # Always log when debug is enabled
                if debug:
                    print(f"      [strength={strength:+.1f}] choice={choice}")
                    # Always show response snippet in debug mode
                    print(f"        Response: {repr(response[:100])}")

        # Find minimum flip magnitudes
        min_flip_to_long = None
        min_flip_to_short = None

        if baseline_choice == "short_term":
            # Look for flip to long_term (positive strengths typically)
            for s in sorted(steering_strengths, key=abs):
                if steered_choices.get(s) == "long_term":
                    min_flip_to_long = abs(s)
                    break

        elif baseline_choice == "long_term":
            # Look for flip to short_term (negative strengths typically)
            for s in sorted(steering_strengths, key=abs):
                if steered_choices.get(s) == "short_term":
                    min_flip_to_short = abs(s)
                    break

        samples.append(
            SampleSteeringResult(
                sample_id=question.sample_id,
                time_horizon=question.time_horizon,
                baseline_choice=baseline_choice,
                steered_choices=steered_choices,
                min_flip_magnitude_to_long=min_flip_to_long,
                min_flip_magnitude_to_short=min_flip_to_short,
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


def subsample_probes(
    probes: dict[str, LoadedProbe],
    fraction: float,
    seed: int = 42,
) -> dict[str, LoadedProbe]:
    """
    Subsample probes with stratified sampling for diverse layer/position coverage.

    Uses stratified sampling to ensure selected probes are spread across:
    - Different layers (early, middle, late)
    - Different token positions

    Args:
        probes: Dict of probe_id -> LoadedProbe
        fraction: Fraction to keep (0.0-1.0). 1.0 keeps all.
        seed: Random seed for reproducibility

    Returns:
        Subsampled dict of probes with diverse layer/position coverage
    """
    import random

    if fraction >= 1.0:
        return probes

    if fraction <= 0.0:
        print("  Warning: subsample_probes=0 would select no probes, using 1 probe")
        fraction = 1.0 / max(len(probes), 1)

    n_to_keep = max(1, int(len(probes) * fraction))

    # Group probes by (layer, position) for stratified sampling
    # This ensures diverse coverage across the layer x position grid
    layer_pos_groups: dict[tuple[int, int], list[str]] = {}
    for pid, probe in probes.items():
        key = (probe.layer, probe.token_position_idx)
        if key not in layer_pos_groups:
            layer_pos_groups[key] = []
        layer_pos_groups[key].append(pid)

    # Get unique layers and positions
    layers = sorted(set(k[0] for k in layer_pos_groups.keys()))
    positions = sorted(set(k[1] for k in layer_pos_groups.keys()))

    # Stratified selection: pick probes spread across the grid
    random.seed(seed)
    selected_ids = set()

    # Round-robin selection across grid cells until we have enough
    grid_cells = list(layer_pos_groups.keys())
    random.shuffle(grid_cells)

    cell_idx = 0
    while len(selected_ids) < n_to_keep and grid_cells:
        cell = grid_cells[cell_idx % len(grid_cells)]
        available = [p for p in layer_pos_groups[cell] if p not in selected_ids]
        if available:
            selected_ids.add(random.choice(available))
        cell_idx += 1
        # Remove exhausted cells
        if cell_idx % len(grid_cells) == 0:
            grid_cells = [c for c in grid_cells if any(p not in selected_ids for p in layer_pos_groups[c])]
            if not grid_cells:
                break

    # Report coverage
    selected_layers = set(probes[pid].layer for pid in selected_ids)
    selected_positions = set(probes[pid].token_position_idx for pid in selected_ids)
    print(f"  Subsampling probes: {len(selected_ids)}/{len(probes)} ({fraction:.0%})")
    print(f"    Layers covered: {len(selected_layers)}/{len(layers)} ({sorted(selected_layers)})")
    print(f"    Positions covered: {len(selected_positions)}/{len(positions)}")

    return {pid: probes[pid] for pid in selected_ids}


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


def load_quick_mode_samples(
    dataset_id: str,
) -> tuple[DatasetOutput, list[QuestionOutput], list[int]]:
    """
    Load exactly 2 samples for quick mode: one expected short->long, one long->short.

    Tries to find samples with different expected baseline choices for balanced testing.

    Returns:
        Tuple of (dataset, questions, sample_ids_used)
    """
    import random

    dataset_path = find_dataset_by_id(dataset_id)
    dataset = load_dataset_output(dataset_path)

    questions = list(dataset.questions)
    random.seed(42)
    random.shuffle(questions)

    # For quick mode, just take 2 samples
    # In a more sophisticated version, we could try to balance baseline choices
    selected = questions[:2] if len(questions) >= 2 else questions
    sample_ids = [q.sample_id for q in selected]
    print(f"Quick mode: selected {len(selected)} samples: {sample_ids}")

    return dataset, selected, sample_ids


def run_steering_experiment(
    config: SteeringConfig,
    quick_mode: bool = False,
    debug: bool = False,
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
        quick_mode: If True, test only best probe with 2 samples
        debug: If True, print debug output
        output_dir: Directory for output (auto-created if None)

    Returns:
        SteeringExperimentOutput with all results
    """
    # Find probes directory
    probes_base = PROJECT_ROOT / "out" / "probes" / config.probe_config_id
    if not probes_base.exists():
        raise FileNotFoundError(f"Probe config not found: {probes_base}")

    probes_dir = probes_base / "probes"
    if not probes_dir.exists():
        raise FileNotFoundError(f"Probes directory not found: {probes_dir}")

    # Load probe index (new format)
    index_path = probes_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Probe index not found: {index_path}")

    index = load_json(index_path)
    # New format uses "model", old format used "model_name"
    model_name = index.get("model", index.get("model_name"))
    if not model_name:
        raise ValueError("No model name found in probe index")

    print(f"Probe config: {config.probe_config_id}")
    print(f"Model: {model_name}")
    if quick_mode:
        print("Quick mode: enabled")

    # Load dataset
    print(f"\nLoading dataset: {config.dataset_id}")
    if quick_mode:
        dataset, questions, sample_ids = load_quick_mode_samples(config.dataset_id)
    else:
        dataset, questions, sample_ids = load_dataset_samples(
            config.dataset_id, config.n_samples, config.sample_indices
        )
    print(f"  Loaded {len(questions)} samples: {sample_ids}")

    # Load model using ModelRunner
    runner = load_model_for_steering(model_name)

    # Load probes
    all_probes = load_probes_from_dir(probes_dir)

    # Filter by probe type
    probes_by_type = {
        pid: p
        for pid, p in all_probes.items()
        if p.probe_type.value in config.probe_types
    }

    if quick_mode:
        # Quick mode: find best probe before choice selection
        best = find_best_probe_before_choice(all_probes, index, config.probe_types)
        if best is None:
            print("  Warning: No suitable probe found for quick mode, using all probes")
        else:
            probes_by_type = {best[0]: best[1]}
    elif config.only_after_horizon:
        # Filter to after-horizon probes if requested
        probes_by_type = filter_probes_after_horizon(probes_by_type, index)

    # Apply probe subsampling if requested
    if config.subsample_probes < 1.0:
        probes_by_type = subsample_probes(probes_by_type, config.subsample_probes)

    print(f"\nTesting {len(probes_by_type)} probes:")
    for pid in sorted(probes_by_type.keys()):
        print(f"  - {pid}")

    # Setup output directory and incremental saver
    if output_dir is None:
        output_base = PROJECT_ROOT / "out" / "steering"
        output_dir = output_base / config.get_id()
    results_dir = output_dir / "results"
    timestamp = get_timestamp()

    saver = IncrementalSaver(
        results_dir=results_dir,
        timestamp=timestamp,
        config_id=config.get_id(),
        probe_config_id=config.probe_config_id,
        dataset_id=config.dataset_id,
        model_name=model_name,
        steering_strengths=config.steering_strengths,
        n_samples=len(questions),
        sample_indices=sample_ids,
    )
    print(f"\nResults will be saved incrementally to: {saver.get_results_path()}")

    # Run steering for each probe with incremental saving and memory clearing
    # Convert to list to allow deletion during iteration
    probe_items = list(probes_by_type.items())
    probe_results = []

    for i, (probe_id, probe) in enumerate(probe_items):
        print(f"\n[{i+1}/{len(probe_items)}] Processing {probe_id}...")

        # Skip if already completed (allows resumption)
        if saver.is_completed(probe_id):
            print(f"    Skipping (already completed)")
            # Still delete probe from memory
            del probes_by_type[probe_id]
            if probe_id in all_probes:
                del all_probes[probe_id]
            continue

        result = run_steering_for_probe(
            runner, probe, questions, config.steering_strengths, debug=debug
        )
        probe_results.append(result)

        # Save result immediately
        saver.add_probe_result(result)

        # Delete probe from memory - we no longer need it
        del probes_by_type[probe_id]
        if probe_id in all_probes:
            del all_probes[probe_id]

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
            samples.append(
                SampleSteeringResult(
                    sample_id=s_data["sample_id"],
                    time_horizon=s_data.get("time_horizon"),
                    baseline_choice=s_data["baseline_choice"],
                    steered_choices=steered_choices,
                    min_flip_magnitude_to_long=s_data.get("min_flip_to_long"),
                    min_flip_magnitude_to_short=s_data.get("min_flip_to_short"),
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
        config_id=config.get_id(),
        probe_config_id=config.probe_config_id,
        dataset_id=config.dataset_id,
        model_name=model_name,
        steering_strengths=config.steering_strengths,
        n_samples=len(questions),
        sample_indices=sample_ids,
        probe_results=final_probe_results,
        timestamp=timestamp,
    )


# =============================================================================
# Random Search - Cell State Tracking
# =============================================================================


@dataclass
class DirectionState:
    """
    Tracks steering state for one direction (short→long or long→short).

    Binary search via distribution bounds:
    - Initial bounds: [min_mag, max_mag]
    - On flip at X: upper_bound = min(upper_bound, X)
    - On no-flip at Y: lower_bound = max(lower_bound, Y)
    - Always sample uniformly from [lower_bound, upper_bound]
    """

    direction: str  # "short_to_long" or "long_to_short"
    min_mag: float  # Initial minimum magnitude
    max_mag: float  # Initial maximum magnitude
    sign: int  # +1 for positive steering, -1 for negative
    tested_magnitudes: list[float] = field(default_factory=list)
    tested_results: list[bool] = field(default_factory=list)
    min_flip_magnitude: Optional[float] = None  # Best (smallest) flip found
    # Binary search bounds
    lower_bound: Optional[float] = None  # Largest no-flip (lower bound of uncertainty)
    upper_bound: Optional[float] = None  # Smallest flip (upper bound of uncertainty)

    def _get_current_bounds(self) -> tuple[float, float]:
        """Get current search bounds."""
        low = self.lower_bound if self.lower_bound is not None else self.min_mag
        high = self.upper_bound if self.upper_bound is not None else self.max_mag
        return (low, high)

    def sample_magnitude(self, rng) -> float:
        """
        Sample from current bounds [lower_bound, upper_bound].

        Before flip found: beta distribution biased toward upper bound.
        After flip found: uniform over narrowed range.
        """
        low, high = self._get_current_bounds()

        if self.upper_bound is None:
            # No flip yet - bias toward large magnitudes
            alpha, beta = 2.0, 1.0  # Skewed toward high end
            u = rng.betavariate(alpha, beta)
        else:
            # Flip found - uniform in [lower, upper] for binary search
            u = rng.random()

        mag = low + u * (high - low)
        return self.sign * mag

    def record_result(self, magnitude: float, flipped: bool) -> None:
        """Record result and update binary search bounds."""
        self.tested_magnitudes.append(magnitude)
        self.tested_results.append(flipped)

        abs_mag = abs(magnitude)
        if flipped:
            # Flip: tighten upper bound
            if self.upper_bound is None or abs_mag < self.upper_bound:
                self.upper_bound = abs_mag
            if self.min_flip_magnitude is None or abs_mag < abs(self.min_flip_magnitude):
                self.min_flip_magnitude = magnitude
        else:
            # No flip: tighten lower bound
            if self.lower_bound is None or abs_mag > self.lower_bound:
                self.lower_bound = abs_mag

    def get_search_range(self) -> tuple[float, float]:
        """Get current search range."""
        return self._get_current_bounds()

    def is_complete(self, min_tests: int = 5, precision: float = 1.0) -> bool:
        """Check if we've narrowed down enough."""
        if len(self.tested_magnitudes) < min_tests:
            return False
        if self.upper_bound is None:
            # No flip found yet
            return len(self.tested_magnitudes) >= min_tests * 2
        low, high = self._get_current_bounds()
        return (high - low) < precision


@dataclass
class CellState:
    """
    Tracks state for a single (layer, position, sample_id) cell.

    Manages both directions separately and tracks baseline choice.
    """

    layer: int
    position: int
    sample_id: int
    positive_range: tuple[float, float]  # (min, max) for short→long
    negative_range: tuple[float, float]  # (min, max) for long→short

    baseline_choice: Optional[str] = None  # Determined on first test
    short_to_long: DirectionState = field(init=False)
    long_to_short: DirectionState = field(init=False)

    def __post_init__(self):
        self.short_to_long = DirectionState(
            direction="short_to_long",
            min_mag=self.positive_range[0],
            max_mag=self.positive_range[1],
            sign=1,
        )
        self.long_to_short = DirectionState(
            direction="long_to_short",
            min_mag=self.negative_range[0],
            max_mag=self.negative_range[1],
            sign=-1,
        )

    def get_active_direction(self) -> Optional[DirectionState]:
        """
        Get the direction we should be testing based on baseline.

        Returns None if baseline not yet determined.
        """
        if self.baseline_choice is None:
            return None
        if self.baseline_choice == "short_term":
            return self.short_to_long  # Try to flip to long
        elif self.baseline_choice == "long_term":
            return self.long_to_short  # Try to flip to short
        return None  # Unknown baseline

    def needs_baseline(self) -> bool:
        """Check if we still need to determine baseline."""
        return self.baseline_choice is None

    def set_baseline(self, choice: str) -> None:
        """Set baseline choice."""
        self.baseline_choice = choice

    def get_results(self) -> dict:
        """Get results for this cell in a format compatible with steering heatmaps."""
        return {
            "layer": self.layer,
            "position": self.position,
            "sample_id": self.sample_id,
            "baseline_choice": self.baseline_choice,
            "min_flip_to_long": self.short_to_long.min_flip_magnitude,
            "min_flip_to_short": self.long_to_short.min_flip_magnitude,
            "tests_to_long": len(self.short_to_long.tested_magnitudes),
            "tests_to_short": len(self.long_to_short.tested_magnitudes),
        }


class CellTracker:
    """
    Manages all cells for random search.

    Provides cell selection with novelty/exploitation balance and
    magnitude selection within cells.
    """

    def __init__(
        self,
        layers: list[int],
        positions: list[int],
        sample_ids: list[int],
        steering_strengths: list[float],
        probe_index: dict,
        seed: int = 42,
        novelty_weight: float = 2.0,
        min_magnitude: float = 0.5,  # Minimum magnitude to test
    ):
        import random

        self.layers = layers
        self.positions = positions
        self.sample_ids = sample_ids
        self.probe_index = probe_index
        self.novelty_weight = novelty_weight
        self._rng = random.Random(seed)

        # Compute continuous ranges from steering_strengths
        positive_vals = [s for s in steering_strengths if s > 0]
        negative_vals = [abs(s) for s in steering_strengths if s < 0]

        # Use min/max to define the sampling range
        self.positive_range = (
            min_magnitude,
            max(positive_vals) if positive_vals else 50.0,
        )
        self.negative_range = (
            min_magnitude,
            max(negative_vals) if negative_vals else 50.0,
        )

        # Cell states: (layer, position, sample_id) -> CellState
        self.cells: dict[tuple[int, int, int], CellState] = {}

        # Compute base weights for layer/position selection
        self._base_layer_weights = self._compute_layer_weights()
        self._base_position_weights = self._compute_position_weights()

    def _compute_layer_weights(self) -> dict[int, float]:
        """Compute base weights for layers based on probe accuracy."""
        probe_accuracy = {}
        for probe_info in self.probe_index.get("probes", []):
            if probe_info.get("type") != "choice":
                continue
            key = (probe_info["layer"], probe_info["position"])
            probe_accuracy[key] = probe_info.get("accuracy", 0.5)

        weights = {}
        n_layers = len(self.layers)
        for i, layer in enumerate(self.layers):
            # Bell curve centered on middle layers
            center = n_layers / 2
            distance = abs(i - center) / max(center, 1)
            layer_weight = 1.0 - 0.5 * distance

            # Boost by max accuracy
            accuracies = [probe_accuracy.get((layer, p), 0.5) for p in self.positions]
            max_acc = max(accuracies) if accuracies else 0.5
            weights[layer] = layer_weight * max_acc

        return weights

    def _compute_position_weights(self) -> dict[int, float]:
        """Compute base weights for positions based on after_horizon and accuracy."""
        probe_after_horizon = {}
        probe_accuracy = {}
        for probe_info in self.probe_index.get("probes", []):
            if probe_info.get("type") != "choice":
                continue
            key = (probe_info["layer"], probe_info["position"])
            probe_accuracy[key] = probe_info.get("accuracy", 0.5)
            probe_after_horizon[key] = probe_info.get("after_horizon", False)

        weights = {}
        for pos in self.positions:
            is_after = any(probe_after_horizon.get((l, pos), False) for l in self.layers)
            accuracies = [probe_accuracy.get((l, pos), 0.5) for l in self.layers]
            max_acc = max(accuracies) if accuracies else 0.5
            weights[pos] = (2.0 if is_after else 1.0) * max_acc

        return weights

    def _get_or_create_cell(self, layer: int, position: int, sample_id: int) -> CellState:
        """Get existing cell or create new one."""
        key = (layer, position, sample_id)
        if key not in self.cells:
            self.cells[key] = CellState(
                layer=layer,
                position=position,
                sample_id=sample_id,
                positive_range=self.positive_range,
                negative_range=self.negative_range,
            )
        return self.cells[key]

    def _get_cell_novelty(self, layer: int, position: int) -> float:
        """UCB1-style novelty bonus for (layer, position)."""
        import math

        total_cells = len(self.cells)
        if total_cells == 0:
            return 1.0

        # Count cells at this (layer, position)
        cell_count = sum(
            1 for (l, p, _) in self.cells.keys() if l == layer and p == position
        )
        return math.sqrt(math.log(total_cells + 1) / (cell_count + 1))

    def _get_cell_priority(self, layer: int, position: int, sample_id: int) -> float:
        """
        Compute priority for a cell based on:
        - Base weights (layer/position quality)
        - Novelty (unexplored cells)
        - Progress (cells with flips found need refinement)
        """
        base = self._base_layer_weights.get(layer, 1.0)
        base *= self._base_position_weights.get(position, 1.0)

        novelty = self._get_cell_novelty(layer, position)

        # Check if cell exists and has found flips
        key = (layer, position, sample_id)
        if key in self.cells:
            cell = self.cells[key]
            direction = cell.get_active_direction()
            if direction and direction.min_flip_magnitude is not None:
                # Has flip - boost priority if still refining
                if not direction.is_complete():
                    return base * 2.0  # Prioritize refinement
                return base * 0.1  # De-prioritize completed cells

        return base * (1.0 + self.novelty_weight * novelty)

    def sample_cell(self) -> Optional[tuple[int, int, int]]:
        """Sample a (layer, position, sample_id) cell weighted by priority."""
        # Build candidate list with weights
        candidates = []
        weights = []
        for layer in self.layers:
            for position in self.positions:
                for sample_id in self.sample_ids:
                    key = (layer, position, sample_id)
                    # Skip completed cells
                    if key in self.cells:
                        cell = self.cells[key]
                        direction = cell.get_active_direction()
                        if direction and direction.is_complete():
                            continue

                    priority = self._get_cell_priority(layer, position, sample_id)
                    candidates.append(key)
                    weights.append(priority)

        if not candidates:
            return None

        # Weighted random choice
        total = sum(weights)
        if total == 0:
            return self._rng.choice(candidates)

        r = self._rng.random() * total
        cumulative = 0.0
        for candidate, weight in zip(candidates, weights):
            cumulative += weight
            if r <= cumulative:
                return candidate
        return candidates[-1]

    def sample_magnitude(self, cell: CellState) -> Optional[float]:
        """Sample a magnitude for a cell based on its current state."""
        direction = cell.get_active_direction()
        if direction is None:
            return None

        # Check if direction is complete
        if direction.is_complete():
            return None

        return direction.sample_magnitude(self._rng)

    def record_baseline(self, layer: int, position: int, sample_id: int, choice: str) -> None:
        """Record baseline choice for a cell."""
        cell = self._get_or_create_cell(layer, position, sample_id)
        cell.set_baseline(choice)

    def record_steering_result(
        self,
        layer: int,
        position: int,
        sample_id: int,
        magnitude: float,
        steered_choice: str,
    ) -> bool:
        """
        Record steering result. Returns True if a flip occurred.
        """
        cell = self._get_or_create_cell(layer, position, sample_id)
        if cell.baseline_choice is None:
            return False

        flipped = (
            cell.baseline_choice in ("short_term", "long_term")
            and steered_choice in ("short_term", "long_term")
            and cell.baseline_choice != steered_choice
        )

        # Record to appropriate direction
        if magnitude > 0:
            cell.short_to_long.record_result(magnitude, flipped)
        else:
            cell.long_to_short.record_result(magnitude, flipped)

        return flipped

    def get_all_results(self) -> list[dict]:
        """Get results from all cells for heatmap generation."""
        return [cell.get_results() for cell in self.cells.values()]

    def get_stats(self) -> dict:
        """Get summary statistics."""
        total_cells = len(self.cells)
        cells_with_baseline = sum(1 for c in self.cells.values() if c.baseline_choice)
        cells_with_flip_to_long = sum(
            1 for c in self.cells.values()
            if c.short_to_long.min_flip_magnitude is not None
        )
        cells_with_flip_to_short = sum(
            1 for c in self.cells.values()
            if c.long_to_short.min_flip_magnitude is not None
        )

        # Grid coverage
        layer_pos_pairs = set((c.layer, c.position) for c in self.cells.values())
        total_grid = len(self.layers) * len(self.positions)

        # Test counts
        total_tests = sum(
            len(c.short_to_long.tested_magnitudes) + len(c.long_to_short.tested_magnitudes)
            for c in self.cells.values()
        )

        return {
            "total_cells": total_cells,
            "cells_with_baseline": cells_with_baseline,
            "cells_with_flip_to_long": cells_with_flip_to_long,
            "cells_with_flip_to_short": cells_with_flip_to_short,
            "grid_coverage": len(layer_pos_pairs) / total_grid if total_grid > 0 else 0,
            "unique_layer_pos": len(layer_pos_pairs),
            "total_grid_cells": total_grid,
            "total_tests": total_tests,
        }


@dataclass
class RandomSearchResult:
    """Result from a single random search iteration."""

    layer: int
    position: int
    sample_id: int
    strength: float
    baseline_choice: str
    steered_choice: str
    flipped: bool
    flip_direction: Optional[str]  # "short_to_long" or "long_to_short"


@dataclass
class RandomSearchOutput:
    """Full output from random search experiment."""

    config_id: str
    probe_config_id: str
    dataset_id: str
    model_name: str
    n_iterations: int
    timestamp: str
    results: list[RandomSearchResult]
    cell_results: list[dict]  # Per-cell min flip magnitudes
    stats: dict


def run_steering_for_single_sample(
    runner: ModelRunner,
    probe: LoadedProbe,
    question: QuestionOutput,
    strength: float,
    max_new_tokens: int = 64,
    debug: bool = False,
) -> tuple[str, str, bool]:
    """
    Run steering for a single sample at a single strength.

    Returns:
        (baseline_choice, steered_choice, flipped)
    """
    prompt = build_prompt_from_question(question)
    pair = question.preference_pair
    short_label = pair.short_term.label
    long_label = pair.long_term.label
    short_value = f"{pair.short_term.reward:,.0f}"
    long_value = f"{pair.long_term.reward:,.0f}"

    # Generate baseline
    baseline_response = runner.generate_baseline(prompt, max_new_tokens)
    baseline_choice = parse_choice_from_response(
        baseline_response, short_label, long_label, debug=debug,
        short_value=short_value, long_value=long_value
    )

    if strength == 0:
        return baseline_choice, baseline_choice, False

    # Generate with steering
    steering_config = SteeringDirectionConfig(
        direction=probe.direction,
        layer=probe.layer,
        strength=strength,
        option=SteeringOption.APPLY_TO_ALL,
    )
    steered_response = runner.generate_with_steering(
        prompt, steering=steering_config, max_new_tokens=max_new_tokens
    )
    steered_choice = parse_choice_from_response(
        steered_response, short_label, long_label, debug=debug,
        short_value=short_value, long_value=long_value
    )

    # Check if flipped
    flipped = (
        baseline_choice in ("short_term", "long_term")
        and steered_choice in ("short_term", "long_term")
        and baseline_choice != steered_choice
    )

    return baseline_choice, steered_choice, flipped


def run_random_search_experiment(
    config: SteeringConfig,
    debug: bool = False,
    output_dir: Optional[Path] = None,
) -> RandomSearchOutput:
    """
    Run random search steering experiment.

    Uses CellTracker for intelligent sampling:
    - Samples cells (layer, position, sample) with novelty/exploitation balance
    - Within each cell, starts with large magnitudes to find flips
    - After flip found, narrows magnitude range to find minimum
    - Tracks both directions (short→long, long→short) separately

    Args:
        config: Steering experiment configuration
        debug: If True, print debug output
        output_dir: Directory for output (auto-created if None)

    Returns:
        RandomSearchOutput with results and per-cell min flip magnitudes
    """
    # Load probe index and model info
    probes_base = PROJECT_ROOT / "out" / "probes" / config.probe_config_id
    probes_dir = probes_base / "probes"
    index_path = probes_dir / "index.json"

    if not index_path.exists():
        raise FileNotFoundError(f"Probe index not found: {index_path}")

    index = load_json(index_path)
    model_name = index.get("model", index.get("model_name"))
    if not model_name:
        raise ValueError("No model name found in probe index")

    print(f"Random Search Experiment")
    print(f"  Probe config: {config.probe_config_id}")
    print(f"  Model: {model_name}")
    print(f"  Iterations: {config.random_search_iterations}")

    # Load dataset
    print(f"\nLoading dataset: {config.dataset_id}")
    dataset, questions, sample_ids = load_dataset_samples(
        config.dataset_id, config.n_samples, config.sample_indices
    )
    questions_by_id = {q.sample_id: q for q in questions}
    print(f"  Loaded {len(questions)} samples")

    # Load model
    runner = load_model_for_steering(model_name)

    # Load all probes of requested types
    all_probes = load_probes_from_dir(probes_dir)
    probes_by_type = {
        pid: p for pid, p in all_probes.items()
        if p.probe_type.value in config.probe_types
    }

    # Filter to after-horizon if requested
    if config.only_after_horizon:
        probes_by_type = filter_probes_after_horizon(probes_by_type, index)

    # Get available layers and positions
    layers = sorted(set(p.layer for p in probes_by_type.values()))
    positions = sorted(set(p.token_position_idx for p in probes_by_type.values()))

    # Build probe lookup by (layer, position)
    probe_lookup: dict[tuple[int, int], LoadedProbe] = {}
    for probe in probes_by_type.values():
        probe_lookup[(probe.layer, probe.token_position_idx)] = probe

    print(f"\nParameter space:")
    print(f"  Layers: {len(layers)} ({min(layers)}-{max(layers)})")
    print(f"  Positions: {len(positions)}")
    print(f"  Samples: {len(sample_ids)}")
    print(f"  Magnitudes: {config.steering_strengths}")

    # Initialize cell tracker
    tracker = CellTracker(
        layers=layers,
        positions=positions,
        sample_ids=sample_ids,
        steering_strengths=config.steering_strengths,
        probe_index=index,
    )

    # Setup output
    if output_dir is None:
        output_base = PROJECT_ROOT / "out" / "steering"
        output_dir = output_base / config.get_id()
    results_dir = output_dir / "results"
    ensure_dir(results_dir)
    timestamp = get_timestamp()

    # Run random search
    results: list[RandomSearchResult] = []
    print(f"\nRunning random search ({config.random_search_iterations} iterations)...")

    for i in range(config.random_search_iterations):
        # Sample a cell
        cell_key = tracker.sample_cell()
        if cell_key is None:
            print(f"\n  All cells complete at iteration {i}")
            break

        layer, position, sample_id = cell_key
        cell = tracker._get_or_create_cell(layer, position, sample_id)

        # Get probe and question
        probe = probe_lookup.get((layer, position))
        question = questions_by_id.get(sample_id)
        if probe is None or question is None:
            continue

        # Build prompt and labels
        prompt = build_prompt_from_question(question)
        pair = question.preference_pair
        short_label = pair.short_term.label
        long_label = pair.long_term.label
        short_value = f"{pair.short_term.reward:,.0f}"
        long_value = f"{pair.long_term.reward:,.0f}"

        # If cell needs baseline, run baseline first
        if cell.needs_baseline():
            baseline_response = runner.generate_baseline(prompt, max_new_tokens=64)
            baseline_choice = parse_choice_from_response(
                baseline_response, short_label, long_label, debug=debug,
                short_value=short_value, long_value=long_value
            )
            tracker.record_baseline(layer, position, sample_id, baseline_choice)

            # Record baseline test
            results.append(RandomSearchResult(
                layer=layer,
                position=position,
                sample_id=sample_id,
                strength=0.0,
                baseline_choice=baseline_choice,
                steered_choice=baseline_choice,
                flipped=False,
                flip_direction=None,
            ))

            # Skip to next iteration if baseline is unknown
            if baseline_choice not in ("short_term", "long_term"):
                continue

        # Sample magnitude for steering
        magnitude = tracker.sample_magnitude(cell)
        if magnitude is None:
            continue  # Cell is complete

        # Run steering
        steering_config_obj = SteeringDirectionConfig(
            direction=probe.direction,
            layer=probe.layer,
            strength=magnitude,
            option=SteeringOption.APPLY_TO_ALL,
        )
        steered_response = runner.generate_with_steering(
            prompt, steering=steering_config_obj, max_new_tokens=64
        )
        steered_choice = parse_choice_from_response(
            steered_response, short_label, long_label, debug=debug,
            short_value=short_value, long_value=long_value
        )

        # Record result
        flipped = tracker.record_steering_result(
            layer, position, sample_id, magnitude, steered_choice
        )

        # Determine flip direction
        flip_direction = None
        if flipped:
            if cell.baseline_choice == "short_term":
                flip_direction = "short_to_long"
            elif cell.baseline_choice == "long_term":
                flip_direction = "long_to_short"

        results.append(RandomSearchResult(
            layer=layer,
            position=position,
            sample_id=sample_id,
            strength=magnitude,
            baseline_choice=cell.baseline_choice or "unknown",
            steered_choice=steered_choice,
            flipped=flipped,
            flip_direction=flip_direction,
        ))

        # Progress report
        if (i + 1) % 10 == 0 or i == 0:
            stats = tracker.get_stats()
            print(f"  [{i+1}/{config.random_search_iterations}] "
                  f"Cells: {stats['total_cells']} | "
                  f"Flips: S→L={stats['cells_with_flip_to_long']}, "
                  f"L→S={stats['cells_with_flip_to_short']} | "
                  f"Coverage: {stats['grid_coverage']:.0%}")

        # Save incrementally
        if (i + 1) % 20 == 0:
            _save_random_search_results(
                results_dir, timestamp, config, model_name, results, tracker
            )

    # Final save
    final_stats = tracker.get_stats()
    cell_results = tracker.get_all_results()
    _save_random_search_results(
        results_dir, timestamp, config, model_name, results, tracker
    )

    # Print summary
    print(f"\n{'='*60}")
    print("RANDOM SEARCH COMPLETE")
    print(f"{'='*60}")
    print(f"Total iterations: {len(results)}")
    print(f"Cells explored: {final_stats['total_cells']}")
    print(f"Flips found: short→long={final_stats['cells_with_flip_to_long']}, "
          f"long→short={final_stats['cells_with_flip_to_short']}")
    print(f"Grid coverage: {final_stats['unique_layer_pos']}/{final_stats['total_grid_cells']} "
          f"({final_stats['grid_coverage']:.0%})")

    return RandomSearchOutput(
        config_id=config.get_id(),
        probe_config_id=config.probe_config_id,
        dataset_id=config.dataset_id,
        model_name=model_name,
        n_iterations=len(results),
        timestamp=timestamp,
        results=results,
        cell_results=cell_results,
        stats=final_stats,
    )


def _save_random_search_results(
    results_dir: Path,
    timestamp: str,
    config: SteeringConfig,
    model_name: str,
    results: list[RandomSearchResult],
    tracker: CellTracker,
) -> None:
    """Save random search results to JSON."""
    data = {
        "config_id": config.get_id(),
        "probe_config_id": config.probe_config_id,
        "dataset_id": config.dataset_id,
        "model_name": model_name,
        "n_iterations": len(results),
        "timestamp": timestamp,
        "stats": tracker.get_stats(),
        "cell_results": tracker.get_all_results(),
        "results": [
            {
                "layer": r.layer,
                "position": r.position,
                "sample_id": r.sample_id,
                "strength": r.strength,
                "baseline_choice": r.baseline_choice,
                "steered_choice": r.steered_choice,
                "flipped": r.flipped,
                "flip_direction": r.flip_direction,
            }
            for r in results
        ],
    }
    save_json(data, results_dir / f"random_search_{timestamp}.json")


def create_random_search_heatmaps(
    output: RandomSearchOutput,
    output_dir: Path,
    probe_index: Optional[dict] = None,
    token_info: Optional[dict] = None,
) -> None:
    """
    Create heatmaps showing minimum steering magnitude to flip choice.

    Creates separate heatmaps for:
    - short->long flips (positive steering)
    - long->short flips (negative steering)

    Uses cell_results from CellTracker which contains min_flip_to_long and
    min_flip_to_short for each (layer, position, sample) cell. Aggregates
    across samples to get per-(layer, position) values.

    Cell values:
    - Number: mean min |strength| that caused flip across samples
    - Gray "N/A": not tested
    - "NEVER": tested but never flipped

    Args:
        output: Random search experiment results
        output_dir: Directory to save visualizations
        probe_index: Full probe index (to show all positions)
        token_info: Dict with 'tokens' and 'resolved_positions' for x-axis labels
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    viz_dir = output_dir / "viz"
    ensure_dir(viz_dir)

    if not output.cell_results:
        print("  No cell results to visualize")
        return

    # Get all layers and positions
    if probe_index is not None:
        all_probes = probe_index.get("probes", [])
        choice_probes = [p for p in all_probes if p.get("type") == "choice"]
        layers = sorted(set(p["layer"] for p in choice_probes))
        positions = sorted(set(p["position"] for p in choice_probes))
    else:
        layers = sorted(set(c["layer"] for c in output.cell_results))
        positions = sorted(set(c["position"] for c in output.cell_results))

    if not layers or not positions:
        print("  No layers/positions to visualize")
        return

    # Build position labels
    def build_position_labels():
        from src.plotting.common import format_token_position_label

        labels = []
        specs = token_info.get("specs", []) if token_info else []
        for pos_idx in positions:
            if specs and pos_idx < len(specs):
                spec_label = format_token_position_label(specs[pos_idx])
            else:
                spec_label = f"pos_{pos_idx}"

            if token_info:
                token = token_info.get("tokens", {}).get(pos_idx, "")
                resolved_pos = token_info.get("resolved_positions", {}).get(pos_idx, "?")
                token_display = repr(token) if token else ""
                labels.append(f"{spec_label}\n[{resolved_pos}] {token_display}")
            else:
                labels.append(spec_label)
        return labels

    position_labels = build_position_labels()

    # Aggregate cell results by (layer, position)
    # Collect all min flip magnitudes for each direction
    sl_flips: dict[tuple[int, int], list[float]] = {}  # short->long
    ls_flips: dict[tuple[int, int], list[float]] = {}  # long->short
    tested_cells: dict[tuple[int, int], int] = {}  # count of cells tested

    for cell in output.cell_results:
        key = (cell["layer"], cell["position"])
        tested_cells[key] = tested_cells.get(key, 0) + 1

        if cell.get("min_flip_to_long") is not None:
            if key not in sl_flips:
                sl_flips[key] = []
            sl_flips[key].append(abs(cell["min_flip_to_long"]))

        if cell.get("min_flip_to_short") is not None:
            if key not in ls_flips:
                ls_flips[key] = []
            ls_flips[key].append(abs(cell["min_flip_to_short"]))

    # Build matrices for each flip direction
    sl_matrix = np.full((len(layers), len(positions)), np.nan)
    sl_never = np.zeros((len(layers), len(positions)), dtype=bool)
    sl_not_tested = np.ones((len(layers), len(positions)), dtype=bool)

    ls_matrix = np.full((len(layers), len(positions)), np.nan)
    ls_never = np.zeros((len(layers), len(positions)), dtype=bool)
    ls_not_tested = np.ones((len(layers), len(positions)), dtype=bool)

    for key, count in tested_cells.items():
        layer, pos = key
        if layer not in layers or pos not in positions:
            continue

        li = layers.index(layer)
        pi = positions.index(pos)

        # Short->Long
        sl_not_tested[li, pi] = False
        if key in sl_flips and sl_flips[key]:
            sl_matrix[li, pi] = np.mean(sl_flips[key])
        else:
            sl_never[li, pi] = True

        # Long->Short
        ls_not_tested[li, pi] = False
        if key in ls_flips and ls_flips[key]:
            ls_matrix[li, pi] = np.mean(ls_flips[key])
        else:
            ls_never[li, pi] = True

    # Determine max strength for colorbar from actual data
    all_vals = [v for vals in sl_flips.values() for v in vals]
    all_vals += [v for vals in ls_flips.values() for v in vals]
    max_strength = max(all_vals) if all_vals else 50.0

    # Create heatmaps
    for matrix, never, not_tested, flips_dict, title, filename in [
        (sl_matrix, sl_never, sl_not_tested, sl_flips,
         "Short→Long Flip", "random_search_short_to_long.png"),
        (ls_matrix, ls_never, ls_not_tested, ls_flips,
         "Long→Short Flip", "random_search_long_to_short.png"),
    ]:
        fig_height = max(6, len(layers) * 0.5 + 1)
        fig_width = max(12, len(positions) * 1.5)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Colormap: Red=high magnitude (hard to flip), Green=low magnitude (easy)
        cmap = plt.cm.RdYlGn_r.copy()
        cmap.set_bad(color="white")
        norm = Normalize(vmin=0, vmax=max_strength)

        # Create display matrix
        display_matrix = matrix.copy()
        display_matrix = np.ma.array(display_matrix)
        display_matrix[not_tested] = np.ma.masked
        display_matrix[never] = np.ma.masked  # Show as white (never flipped)

        im = ax.imshow(display_matrix, cmap=cmap, norm=norm, aspect="auto", origin="lower")

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Min |Steering| to Flip", fontsize=10)

        # Add text annotations
        for i in range(len(layers)):
            for j in range(len(positions)):
                key = (layers[i], positions[j])

                if not_tested[i, j]:
                    ax.text(j, i, "N/A", ha="center", va="center", fontsize=7,
                            color="#cccccc", style="italic")
                elif never[i, j]:
                    # Tested but never flipped - show count
                    n_samples = tested_cells.get(key, 0)
                    ax.text(j, i, f"NEVER\n({n_samples})", ha="center", va="center",
                            fontsize=7, color="#888888")
                else:
                    val = matrix[i, j]
                    n_flips = len(flips_dict.get(key, []))
                    color = "white" if val > max_strength / 2 else "black"
                    ax.text(j, i, f"{val:.1f}\n({n_flips})", ha="center", va="center",
                            fontsize=7, color=color)

        ax.set_xticks(range(len(positions)))
        ax.set_xticklabels(position_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([f"Layer {l}" for l in layers])
        ax.set_xlabel("Token Position (sequence order)", fontsize=11)
        ax.set_ylabel("Layer", fontsize=11)
        ax.set_title(
            f"Random Search: Min Steering to {title}\n"
            f"Model: {output.model_name} | Iterations: {output.n_iterations}",
            fontsize=12,
        )

        plt.tight_layout()
        plt.savefig(viz_dir / filename, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved: {viz_dir / filename}")


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

    Creates separate heatmaps for:
    - short->long flips (positive steering)
    - long->short flips (negative steering)

    Cell values:
    - Number: min |strength| that caused flip
    - Gray "N/A": never flipped
    - White "N/A": not tested (quick mode)
    - Black "FAIL": degeneration (can't parse)

    Args:
        output: Steering experiment results
        output_dir: Directory to save visualizations
        probe_index: Full probe index (to show all positions in quick mode)
        token_info: Dict with 'tokens' and 'resolved_positions' for x-axis labels
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    viz_dir = output_dir / "viz"
    ensure_dir(viz_dir)

    # Get all layers and positions from probe index if available (for quick mode)
    # Otherwise use only tested positions
    if probe_index is not None:
        # Get all unique layers and positions from the probe index
        all_probes = probe_index.get("probes", [])
        # Filter to the probe types we tested
        # probe_type may be string or enum, handle both
        tested_types = set(
            pr.probe_type if isinstance(pr.probe_type, str) else pr.probe_type.value
            for pr in output.probe_results
        )
        relevant_probes = [p for p in all_probes if p.get("type") in tested_types]
        layers = sorted(set(p["layer"] for p in relevant_probes))
        positions = sorted(set(p["position"] for p in relevant_probes))
    else:
        layers = sorted(set(pr.layer for pr in output.probe_results))
        positions = sorted(set(pr.token_position_idx for pr in output.probe_results))

    if not layers or not positions:
        print("  No probes to visualize")
        return

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

    for pr in output.probe_results:
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
        raise RuntimeError(
            f"No probe results written to matrix! "
            f"probe_results={[(pr.layer, pr.token_position_idx) for pr in output.probe_results]}, "
            f"layers={layers}, positions={positions}"
        )

    # Create heatmaps
    max_strength = max(abs(s) for s in output.steering_strengths)

    for matrix, fail, never, not_tested, title, filename in [
        (
            sl_matrix,
            sl_fail,
            sl_never,
            sl_not_tested,
            "Short→Long Flip",
            "steering_short_to_long.png",
        ),
        (
            ls_matrix,
            ls_fail,
            ls_never,
            ls_not_tested,
            "Long→Short Flip",
            "steering_long_to_short.png",
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

        # Mask cells that are not tested (quick mode shows them as white)
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
                    # Not tested (quick mode) - light gray text on white
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
                    # Tested but never flipped - gray text on colored background
                    ax.text(
                        j,
                        i,
                        "N/A",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="#666666",
                        fontweight="bold",
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
            f"Min Steering Magnitude for {title}\n"
            f"Model: {output.model_name} | n={output.n_samples}",
            fontsize=12,
        )

        plt.tight_layout()
        plt.savefig(viz_dir / filename, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved: {viz_dir / filename}")


def create_summary_plots(output: SteeringExperimentOutput, output_dir: Path) -> None:
    """Create summary bar plots of flip rates."""
    import matplotlib.pyplot as plt

    viz_dir = output_dir / "viz"
    ensure_dir(viz_dir)

    for pr in output.probe_results:
        fig, ax = plt.subplots(figsize=(10, 6))

        strengths = sorted(output.steering_strengths)

        # Count flips and degenerations
        flip_counts = []
        degen_counts = []

        for s in strengths:
            flips = 0
            degens = 0
            for sample in pr.samples:
                choice = sample.steered_choices.get(s, "unknown")
                if choice == "degenerate":
                    degens += 1
                elif choice != sample.baseline_choice and choice in (
                    "short_term",
                    "long_term",
                ):
                    flips += 1
            flip_counts.append(flips)
            degen_counts.append(degens)

        x = range(len(strengths))
        width = 0.35

        ax.bar(
            [i - width / 2 for i in x],
            flip_counts,
            width,
            label="Flipped",
            color="steelblue",
        )
        ax.bar(
            [i + width / 2 for i in x],
            degen_counts,
            width,
            label="Degenerate",
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
        filename = f"{pr.probe_id}_summary.png"
        plt.savefig(viz_dir / filename, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved: {viz_dir / filename}")


# =============================================================================
# Output Serialization
# =============================================================================


def create_debug_summary(output: SteeringExperimentOutput, output_dir: Path) -> dict:
    """
    Create debug_steering.json with summary of steering results.

    Shows at a glance whether steering worked, failed samples, invalid choices, etc.
    """
    summary = {
        "config_id": output.config_id,
        "probe_config_id": output.probe_config_id,
        "dataset_id": output.dataset_id,
        "model_name": output.model_name,
        "n_samples": output.n_samples,
        "sample_indices": output.sample_indices,
        "steering_strengths": output.steering_strengths,
        "probes_tested": len(output.probe_results),
        "probe_summaries": [],
    }

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
                flipped_samples.append({
                    "sample_id": s.sample_id,
                    "direction": "short->long",
                    "magnitude": s.min_flip_magnitude_to_long,
                })
            if s.min_flip_magnitude_to_short is not None:
                flips_ls += 1
                flipped_samples.append({
                    "sample_id": s.sample_id,
                    "direction": "long->short",
                    "magnitude": s.min_flip_magnitude_to_short,
                })

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
        "status": "SUCCESS" if total_flips > 0 else (
            "FAILED" if total_failed > 0 else (
                "INVALID" if total_invalid_baseline == output.n_samples else "NO_FLIPS"
            )
        ),
    }

    # Save to file
    debug_path = output_dir / "debug_steering.json"
    save_json(summary, debug_path)
    print(f"\nDebug summary saved to: {debug_path}")
    print(f"  Status: {summary['overall']['status']}")
    print(f"  Total flips: {total_flips}, Failed: {total_failed}, Invalid: {total_invalid_baseline}")

    return summary


def serialize_probe_result(pr: ProbeSteeringResult) -> dict:
    """Serialize a single probe result to JSON-compatible format."""
    return {
        "probe_id": pr.probe_id,
        "probe_type": pr.probe_type,
        "layer": pr.layer,
        "token_position_idx": pr.token_position_idx,
        "after_horizon": pr.after_horizon,
        "samples": [
            {
                "sample_id": s.sample_id,
                "time_horizon": s.time_horizon,
                "baseline_choice": s.baseline_choice,
                "steered_choices": {
                    str(k): v for k, v in s.steered_choices.items()
                },
                "min_flip_to_long": s.min_flip_magnitude_to_long,
                "min_flip_to_short": s.min_flip_magnitude_to_short,
            }
            for s in pr.samples
        ],
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
    config_id: str
    probe_config_id: str
    dataset_id: str
    model_name: str
    steering_strengths: list[float]
    n_samples: int
    sample_indices: list[int]

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
            "config_id": self.config_id,
            "probe_config_id": self.probe_config_id,
            "dataset_id": self.dataset_id,
            "model_name": self.model_name,
            "steering_strengths": self.steering_strengths,
            "n_samples": self.n_samples,
            "sample_indices": self.sample_indices,
            "timestamp": self.timestamp,
            "probe_results": self._probe_results,
            "_completed_probe_ids": list(self._completed_probe_ids),
        }
        save_json(data, self._results_path)

    def add_probe_result(self, result: ProbeSteeringResult) -> None:
        """Add and save a completed probe result."""
        serialized = serialize_probe_result(result)
        self._probe_results.append(serialized)
        self._completed_probe_ids.add(result.probe_id)
        self._save()
        print(f"    Saved result for {result.probe_id} ({len(self._probe_results)} probes complete)")

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
    return {
        "config_id": output.config_id,
        "probe_config_id": output.probe_config_id,
        "dataset_id": output.dataset_id,
        "model_name": output.model_name,
        "steering_strengths": output.steering_strengths,
        "n_samples": output.n_samples,
        "sample_indices": output.sample_indices,
        "timestamp": output.timestamp,
        "probe_results": [serialize_probe_result(pr) for pr in output.probe_results],
    }


def print_summary(output: SteeringExperimentOutput) -> None:
    """Print summary of steering experiment."""
    print("\n" + "=" * 70)
    print("STEERING EXPERIMENT SUMMARY")
    print("=" * 70)
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

    return SteeringExperimentOutput(
        config_id=data["config_id"],
        probe_config_id=data["probe_config_id"],
        dataset_id=data["dataset_id"],
        model_name=data["model_name"],
        steering_strengths=data["steering_strengths"],
        n_samples=data["n_samples"],
        sample_indices=data.get("sample_indices", []),  # Backwards compat
        probe_results=probe_results,
        timestamp=data["timestamp"],
    )


# =============================================================================
# Token Position Info for Visualization
# =============================================================================


def get_token_position_info_from_probe_data(
    probe_config_id: str,
) -> Optional[dict]:
    """
    Get token position info from probe training data.

    Loads the preference data used to train the probes and extracts
    token positions, actual tokens, and token position specs for visualization labels.

    Returns:
        Dict with 'tokens', 'resolved_positions', and 'specs' mappings, or None if not available
    """
    from src.probes.data import (
        extract_token_position_specs,
        find_preference_data_by_query_id,
        load_preference_data_file,
    )

    # Load probe index to get train query IDs
    probes_dir = PROJECT_ROOT / "out" / "probes" / probe_config_id / "probes"
    index_path = probes_dir / "index.json"
    if not index_path.exists():
        return None

    index = load_json(index_path)
    data_info = index.get("data", {})
    query_ids = data_info.get("train_query_ids", [])

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
        "--probe-config-id",
        type=str,
        default=None,
        help="Override probe config ID",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default=None,
        help="Override dataset ID",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Override number of samples",
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
        "--quick",
        action="store_true",
        help="Quick mode: test only best probe (before choice), 1 sample per direction",
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
        "--subsample-probes",
        type=float,
        default=None,
        help="Fraction of probes to use (0.0-1.0). Overrides config value.",
    )
    parser.add_argument(
        "--steering-strengths",
        type=str,
        default=None,
        help="Comma-separated steering strengths (e.g., '-50,0,50'). Overrides config.",
    )
    parser.add_argument(
        "--random-search",
        action="store_true",
        help="Enable random parameter search mode. Overrides config.",
    )
    parser.add_argument(
        "--no-random-search",
        action="store_true",
        help="Disable random parameter search mode. Overrides config.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of random search iterations. Overrides config.",
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
    if args.probe_config_id:
        config.probe_config_id = args.probe_config_id
    if args.dataset_id:
        config.dataset_id = args.dataset_id
    if args.n_samples:
        config.n_samples = args.n_samples
    if args.sample_indices:
        # Parse comma-separated sample IDs
        config.sample_indices = [int(x.strip()) for x in args.sample_indices.split(",")]
    if args.all_probes:
        config.only_after_horizon = False
    if args.subsample_probes is not None:
        config.subsample_probes = args.subsample_probes
    if args.steering_strengths is not None:
        config.steering_strengths = [float(x.strip()) for x in args.steering_strengths.split(",")]
    if args.random_search:
        config.random_search = True
    if args.no_random_search:
        config.random_search = False
    if args.iterations is not None:
        config.random_search_iterations = args.iterations

    print("=" * 60)
    print("STEERING EXPERIMENT")
    print("=" * 60)
    print(f"Probe config: {config.probe_config_id}")
    print(f"Dataset: {config.dataset_id}")
    print(f"Strengths: {config.steering_strengths}")
    print(f"N samples: {config.n_samples}")
    print(f"Only after horizon: {config.only_after_horizon}")
    print(f"Subsample probes: {config.subsample_probes:.0%}")
    print(f"Random search: {config.random_search}")
    if config.random_search:
        print(f"Random search iterations: {config.random_search_iterations}")
    print(f"Quick mode: {args.quick}")
    print(f"Reuse existing: {args.reuse}")

    # Setup output directory
    output_base = PROJECT_ROOT / "out" / "steering"
    output_dir = output_base / config.get_id()
    ensure_dir(output_dir)
    results_dir = output_dir / "results"

    # Load probe index for visualizations
    probe_index = None
    probes_dir = PROJECT_ROOT / "out" / "probes" / config.probe_config_id / "probes"
    index_path = probes_dir / "index.json"
    if index_path.exists():
        probe_index = load_json(index_path)

    # Get token info for proper x-axis labels
    token_info = get_token_position_info_from_probe_data(config.probe_config_id)
    if token_info:
        print(f"  Loaded token info: {len(token_info.get('tokens', {}))} positions")

    # Branch based on random search mode
    if config.random_search:
        # Random search mode
        print("\n" + "=" * 60)
        print("RANDOM SEARCH MODE")
        print("=" * 60)

        random_output = run_random_search_experiment(
            config,
            debug=args.debug,
            output_dir=output_dir,
        )

        # Create random search visualization
        print("\nCreating visualizations...")
        create_random_search_heatmaps(
            random_output, output_dir, probe_index=probe_index, token_info=token_info
        )
    else:
        # Grid search mode (original behavior)
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
                quick_mode=args.quick,
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
