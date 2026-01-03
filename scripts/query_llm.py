#!/usr/bin/env python
"""
Query a language model with intertemporal preference dataset.

Saves results including model responses and optionally internal activations.

Usage:
    python scripts/query_llm.py
    python scripts/query_llm.py --config default_query
    python scripts/query_llm.py --config my_experiment
"""

from __future__ import annotations

import argparse
import gc
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch

# Add project root and scripts to path
PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from src.dataset_generator import DatasetGenerator
from src.common.io import ensure_dir, load_json
from src.model_runner import ModelRunner, CapturedInternals, LabelProbsOutput, RunOutput
from src.common.profiling import get_profiler
from src.common.schemas import SCHEMA_VERSION, DatasetSpec, DecodingConfig, InternalsConfig, QueryConfig, FormattingConfig, SingleQuerySpec

from common import (
    determine_choice,
    format_response_format,
    load_dataset_output,
    parse_label_from_response,
    save_preference_data,
    DatasetOutput,
    InternalsReference,
    OptionOutput,
    PreferenceDataMetadata,
    PreferenceDataOutput,
    PreferenceDebugInfo,
    PreferenceItem,
    PreferencePairOutput,
    QuestionOutput,
)


# =============================================================================
# Formatting ID Verification
# =============================================================================

# Expected formatting_ids for known configs
# Update these when intentionally changing a formatting config
EXPECTED_FORMATTING_IDS = {
    "default_formatting": "0c3aae84eef4bcf3a62cb05ec7439583",
}


def verify_formatting_id(formatting_name: str, formatting_config: FormattingConfig) -> None:
    """
    Verify that formatting_id matches expected value.

    Prints a big warning if there's a mismatch, which helps detect
    unintentional changes to formatting configs.
    """
    actual_id = formatting_config.get_id()
    expected_id = EXPECTED_FORMATTING_IDS.get(formatting_name)

    if expected_id is None:
        # No expected ID registered - print info
        print(f"\n{'='*70}")
        print(f"INFO: No expected formatting_id for '{formatting_name}'")
        print(f"Computed ID: {actual_id}")
        print(f"To register, add to EXPECTED_FORMATTING_IDS in query_llm.py:")
        print(f'    "{formatting_name}": "{actual_id}",')
        print(f"{'='*70}\n")
        return

    if actual_id != expected_id:
        print(f"\n{'!'*70}")
        print(f"{'!'*70}")
        print(f"WARNING: FORMATTING CONFIG MISMATCH")
        print(f"{'!'*70}")
        print(f"Config name: {formatting_name}")
        print(f"Expected ID: {expected_id}")
        print(f"Actual ID:   {actual_id}")
        print(f"")
        print(f"The formatting config has changed from the expected version!")
        print(f"If this change is intentional, update EXPECTED_FORMATTING_IDS:")
        print(f'    "{formatting_name}": "{actual_id}",')
        print(f"")
        print(f"Current config values:")
        print(f"  question_template: {formatting_config.question_template[:100]}...")
        print(f"  response_format: {formatting_config.response_format[:100]}...")
        print(f"  choice_prefix: {formatting_config.choice_prefix}")
        print(f"  reasoning_prefix: {formatting_config.reasoning_prefix}")
        print(f"  time_horizon_spec: {formatting_config.time_horizon_spec}")
        print(f"  max_reasoning_length: {formatting_config.max_reasoning_length}")
        print(f"{'!'*70}")
        print(f"{'!'*70}\n")


# =============================================================================
# Query Config Loading
# =============================================================================


def load_query_config(path: Path) -> QueryConfig:
    """Load query config from JSON file."""
    data = load_json(path)

    decoding = DecodingConfig(
        max_new_tokens=data.get("decoding", {}).get("max_new_tokens", 256),
        temperature=data.get("decoding", {}).get("temperature", 0.0),
        top_k=data.get("decoding", {}).get("top_k", 0),
        top_p=data.get("decoding", {}).get("top_p", 1.0),
    )

    internals = InternalsConfig(
        activations=data.get("internals", {}),
        token_positions=data.get("token_positions", []),
    )

    # Load formatting config to get its ID
    formatting_name = data.get("formatting", {}).get("name", "default_formatting")
    formatting_config_path = SCRIPTS_DIR / "configs" / "formatting" / f"{formatting_name}.json"
    formatting_config = DatasetGenerator.load_formatting_config(formatting_config_path)

    # Verify formatting_id matches expected value
    verify_formatting_id(formatting_name, formatting_config)

    # Parse datasets list
    datasets = []
    for ds in data["datasets"]:
        datasets.append(DatasetSpec(
            name=ds["name"],
            dataset_id=ds["dataset_id"],
        ))

    return QueryConfig(
        models=data["models"],
        datasets=datasets,
        formatting_name=formatting_name,
        formatting_id=formatting_config.get_id(),
        decoding=decoding,
        internals=internals,
        device=data.get("device"),
        limit=data.get("limit", 0),
        subsample=data.get("subsample", 1.0),
        batch_size=data.get("batch_size", 8),
    )


# =============================================================================
# Dataset Loading
# =============================================================================


@dataclass
class DatasetInfo:
    """Loaded dataset with associated config info."""
    dataset: DatasetOutput
    questions: list[QuestionOutput]
    formatting_config: FormattingConfig
    add_variations: bool
    base_labels: tuple[str, str]


def load_dataset_for_query(config: QueryConfig, dataset_spec: DatasetSpec) -> DatasetInfo:
    """Load dataset and prepare questions for querying."""
    datasets_dir = PROJECT_ROOT / "out" / "datasets"

    # Try to find dataset by ID first
    dataset_pattern = f"*_{dataset_spec.dataset_id}.json"
    matches = list(datasets_dir.glob(dataset_pattern))

    dataset_path = None
    if matches:
        dataset_path = matches[0]
    else:
        # Fallback: try to find by name
        name_pattern = f"{dataset_spec.name}_*.json"
        name_matches = list(datasets_dir.glob(name_pattern))

        if len(name_matches) == 1:
            dataset_path = name_matches[0]
            print("\n" + "!" * 80)
            print("!" * 80)
            print("!!! GIGANTIC WARNING !!!")
            print(f"!!! Dataset ID '{dataset_spec.dataset_id}' NOT FOUND")
            print(f"!!! Falling back to dataset by name: {dataset_spec.name}")
            print(f"!!! Using: {dataset_path.name}")
            print("!!! The dataset_id in your config may be outdated!")
            print("!" * 80)
            print("!" * 80 + "\n")
        elif len(name_matches) > 1:
            raise FileNotFoundError(
                f"Dataset ID '{dataset_spec.dataset_id}' not found, and multiple datasets "
                f"match name '{dataset_spec.name}': {[m.name for m in name_matches]}"
            )
        else:
            raise FileNotFoundError(
                f"Dataset with id '{dataset_spec.dataset_id}' or name '{dataset_spec.name}' "
                f"not found in out/datasets/"
            )

    dataset = load_dataset_output(dataset_path)

    # Warn if expected name doesn't match actual name
    actual_name = dataset.metadata.config.get("name", "")
    if actual_name and actual_name != dataset_spec.name:
        print(f"WARNING: Expected dataset name '{dataset_spec.name}' but found '{actual_name}'")

    questions = list(dataset.questions)

    # Apply limit
    if config.limit > 0:
        questions = questions[:config.limit]

    # Apply subsampling
    if config.subsample < 1.0:
        n_samples = max(1, int(len(questions) * config.subsample))
        questions = random.sample(questions, n_samples)
        print(f"Subsampled to {n_samples} questions ({config.subsample:.0%} of {len(dataset.questions)})")

    # Load formatting config
    formatting_path = SCRIPTS_DIR / "configs" / "formatting" / f"{config.formatting_name}.json"
    formatting_config = DatasetGenerator.load_formatting_config(formatting_path)

    # Check for variations
    add_variations = dataset.metadata.config.get("add_formatting_variations", False)
    context = dataset.metadata.config.get("context", {})
    base_labels = tuple(context.get("labels", ["a)", "b)"]))

    return DatasetInfo(
        dataset=dataset,
        questions=questions,
        formatting_config=formatting_config,
        add_variations=add_variations,
        base_labels=base_labels,
    )


# =============================================================================
# Internals Handling
# =============================================================================


@dataclass
class InternalsSetup:
    """Setup for capturing internals."""
    config: Optional[InternalsConfig]
    output_dir: Optional[Path]
    base_filename: Optional[str]


def setup_internals_capture(
    config: QueryConfig,
    dataset_name: str,
    model_name: str,
    query_id: str,
) -> InternalsSetup:
    """Setup internals capture if configured."""
    if not config.internals.activations:
        return InternalsSetup(config=None, output_dir=None, base_filename=None)

    safe_model_name = model_name.split("/")[-1]
    return InternalsSetup(
        config=config.internals,
        output_dir=PROJECT_ROOT / "out" / "internals",
        base_filename=f"{dataset_name}_{safe_model_name}_{query_id}",
    )


def save_internals(
    captured: Optional[CapturedInternals],
    setup: InternalsSetup,
    sample_id: int,
) -> Optional[InternalsReference]:
    """Save captured internals to file and return reference."""
    if captured is None or not captured.activations or setup.output_dir is None:
        return None

    ensure_dir(setup.output_dir)

    filename = f"{setup.base_filename}_sample_{sample_id}.pt"
    file_path = setup.output_dir / filename

    torch.save(captured.activations, file_path)

    relative_path = file_path.relative_to(PROJECT_ROOT)
    return InternalsReference(
        file_path=str(relative_path),
        activations=list(captured.activations.keys()),
        token_positions=captured.token_positions,
        tokens=captured.tokens,
    )


# =============================================================================
# Question Processing
# =============================================================================


def build_prompt(
    question: QuestionOutput,
    response_format_str: str,
) -> str:
    """Build full prompt from question and response format."""
    return question.question_text + response_format_str


def build_time_horizon_spec_text(
    time_horizon: list | None,
    formatting_config: FormattingConfig,
) -> str | None:
    """
    Build the actual time_horizon_spec text for a sample.

    Args:
        time_horizon: [value, unit] or None
        formatting_config: Formatting configuration with time_horizon_spec template

    Returns:
        Actual time_horizon_spec text, or None if no time_horizon
    """
    if time_horizon is None or not formatting_config.time_horizon_spec:
        return None

    value, unit = time_horizon
    time_str = f"{value} {unit}"
    return formatting_config.time_horizon_spec.replace("[TIME_HORIZON]", time_str)


def process_response(
    response_text: str,
    question: QuestionOutput,
    choice_prefix: str,
    model_name: str,
) -> tuple[Optional[str], str]:
    """Parse response and determine choice."""
    pair = question.preference_pair
    labels = [pair.short_term.label, pair.long_term.label]

    chosen_label = parse_label_from_response(response_text, labels, choice_prefix, model_name)
    choice = determine_choice(chosen_label, pair.short_term.label, pair.long_term.label)

    return chosen_label, choice


def build_preference_item(
    question: QuestionOutput,
    choice: str,
    choice_probability: Optional[float],
    alternative_probability: Optional[float],
    internals_ref: Optional[InternalsReference],
    debug_info: Optional[PreferenceDebugInfo],
) -> PreferenceItem:
    """Build preference item from question and response data."""
    pair = question.preference_pair
    return PreferenceItem(
        sample_id=question.sample_id,
        time_horizon=question.time_horizon,
        preference_pair=PreferencePairOutput(
            short_term=OptionOutput(
                label=pair.short_term.label,
                time=pair.short_term.time,
                reward=pair.short_term.reward,
            ),
            long_term=OptionOutput(
                label=pair.long_term.label,
                time=pair.long_term.time,
                reward=pair.long_term.reward,
            ),
        ),
        choice=choice,
        choice_probability=choice_probability,
        alternative_probability=alternative_probability,
        internals=internals_ref,
        debug=debug_info,
    )


# =============================================================================
# Memory Management
# =============================================================================


def clear_memory() -> None:
    """Clear GPU/CPU memory after model run."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# =============================================================================
# Main Query Function
# =============================================================================


def query_llm(
    config: QueryConfig,
    dataset_spec: DatasetSpec,
    model_name: str,
    output_dir: Path,
    query_run_id: str,
) -> PreferenceDataOutput:
    """
    Query LLM on dataset samples.

    Args:
        config: Query configuration
        dataset_spec: Dataset specification (name and ID)
        model_name: Name of the model to query
        output_dir: Output directory
        query_run_id: Shared ID for all queries from same config

    Returns:
        PreferenceDataOutput with results
    """
    # Load dataset and prepare questions
    dataset_info = load_dataset_for_query(config, dataset_spec)

    # Compute query_id as proper schema ID (single dataset + model + all query params)
    query_spec = SingleQuerySpec(
        dataset_id=dataset_spec.dataset_id,
        model=model_name,
        formatting_id=config.formatting_id,
        decoding=config.decoding,
        internals=config.internals,
        subsample=config.subsample,
    )
    query_id = query_spec.get_id()

    # Pre-format response_format if variations disabled
    base_response_format = None
    if not dataset_info.add_variations:
        base_response_format = format_response_format(
            dataset_info.formatting_config,
            dataset_info.base_labels,
            model_name,
        )

    # Load model
    print(f"Loading model: {model_name}")
    model = ModelRunner(model_name=model_name, device=config.device)

    # Setup internals capture
    dataset_name = dataset_info.dataset.metadata.config["name"]
    internals_setup = setup_internals_capture(config, dataset_name, model_name, query_id)

    # Process questions
    preferences = []
    sample_prompt = ""
    sample_continuation = ""
    print(f"\nQuerying {len(dataset_info.questions)} samples...")

    for i, question in enumerate(dataset_info.questions):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processing sample {i + 1}/{len(dataset_info.questions)}")

        # Get response format (per-question if variations enabled)
        if dataset_info.add_variations:
            pair = question.preference_pair
            labels = (pair.short_term.label, pair.long_term.label)
            response_format_str = format_response_format(
                dataset_info.formatting_config, labels, model_name
            )
        else:
            response_format_str = base_response_format

        # Build prompt and run inference
        full_prompt = build_prompt(question, response_format_str)
        time_horizon_spec_text = build_time_horizon_spec_text(
            question.time_horizon, dataset_info.formatting_config
        )
        run_output = model.run(
            full_prompt,
            decoding=config.decoding,
            internals_config=internals_setup.config,
            marker_text=time_horizon_spec_text,
        )
        response_text = run_output.response
        captured = run_output.internals

        # Process response
        chosen_label, choice = process_response(
            response_text,
            question,
            dataset_info.formatting_config.choice_prefix,
            model_name,
        )

        # Get label probabilities
        pair = question.preference_pair
        labels = (pair.short_term.label, pair.long_term.label)
        label_probs = model.get_label_probs(full_prompt, labels)
        short_prob, long_prob = label_probs.prob1, label_probs.prob2

        # Determine choice and alternative probabilities
        if choice == "short_term":
            choice_probability = short_prob
            alternative_probability = long_prob
        elif choice == "long_term":
            choice_probability = long_prob
            alternative_probability = short_prob
        else:  # unknown
            # For unknown, use max as "choice" prob, min as alternative
            choice_probability = max(short_prob, long_prob)
            alternative_probability = min(short_prob, long_prob)

        # Save internals
        internals_ref = save_internals(captured, internals_setup, question.sample_id)

        # Build debug info - always save prompt and continuation for reproducibility
        debug_info = PreferenceDebugInfo(
            raw_prompt=full_prompt,
            raw_continuation=response_text,
            parsed_label=chosen_label,
        )

        # Build preference item
        preferences.append(build_preference_item(
            question, choice, choice_probability, alternative_probability,
            internals_ref, debug_info,
        ))

        # Capture sample prompt/continuation (prefer one with time_horizon)
        has_horizon_sample = sample_prompt and "concerned about outcome" in sample_prompt
        if question.time_horizon is not None and not has_horizon_sample:
            # Found a sample with time_horizon - use it
            sample_prompt = full_prompt
            sample_continuation = response_text
        elif not sample_prompt:
            # Fallback: use first sample if no time_horizon samples exist yet
            sample_prompt = full_prompt
            sample_continuation = response_text

    # Build output - query_config matches SingleQuerySpec fields for ID reconstruction
    # Use full model name (e.g., "Qwen/Qwen2.5-7B-Instruct") for consistency
    single_query_config = {
        "dataset_id": dataset_spec.dataset_id,
        "model": model_name,
        "formatting_id": config.formatting_id,
        "decoding": asdict(config.decoding),
        "internals": asdict(config.internals),
        "subsample": config.subsample,
    }
    output = PreferenceDataOutput(
        metadata=PreferenceDataMetadata(
            version=SCHEMA_VERSION,
            dataset_id=dataset_spec.dataset_id,
            formatting_id=config.formatting_id,
            query_id=query_id,
            model=model_name,  # Full model name, matches query_config.model
            query_config=single_query_config,
            query_run_id=query_run_id,
            sample_prompt=sample_prompt,
            sample_continuation=sample_continuation,
        ),
        preferences=preferences,
    )

    # Save output
    output_path = save_preference_data(output, output_dir, dataset_name, model_name, query_id)
    print(f"\nResults saved to: {output_path}")

    # Cleanup
    del model
    clear_memory()

    return output


# =============================================================================
# CLI
# =============================================================================


def get_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Query a language model with intertemporal preference dataset"
    )
    parser.add_argument(
        "--config", type=str, nargs="*", default=["default_query"],
        help="Query config name(s) from configs/query/ (default: default_query)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output directory (default: out/preference_data/)",
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="Show profiling summary",
    )
    return parser.parse_args()


def print_output_summary(output: PreferenceDataOutput, dataset_name: str) -> None:
    """Print output summary for a single query."""
    print()
    print("=" * 60)
    print("QUERY COMPLETE")
    print("=" * 60)
    print(f"Dataset:  {dataset_name}")
    print(f"Model:    {output.metadata.model}")
    print(f"Query ID: {output.metadata.query_id}")
    print(f"Samples:  {len(output.preferences)}")

    choices = {"short_term": 0, "long_term": 0, "unknown": 0}
    for p in output.preferences:
        choices[p.choice] = choices.get(p.choice, 0) + 1

    print("\nChoice distribution:")
    print(f"  Short-term: {choices['short_term']}")
    print(f"  Long-term:  {choices['long_term']}")
    print(f"  Unknown:    {choices['unknown']}")


@dataclass
class QueryResult:
    """Result from a single query run."""
    config_name: str
    dataset_name: str
    model: str
    query_id: str
    num_samples: int
    short_term: int
    long_term: int
    unknown: int


def print_final_summary(results: list[QueryResult]) -> None:
    """Print final summary of all queries."""
    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print()

    # Table header
    print(f"{'Dataset':<20} {'Model':<30} {'Query ID':<34} {'S/L/U':<10}")
    print("-" * 80)

    for r in results:
        choices = f"{r.short_term}/{r.long_term}/{r.unknown}"
        print(f"{r.dataset_name:<20} {r.model:<30} {r.query_id:<34} {choices:<10}")

    print("-" * 80)
    print(f"Total queries: {len(results)}")


def main() -> int:
    args = get_args()

    # Determine output directory
    output_dir = args.output or (PROJECT_ROOT / "out" / "preference_data")

    # Collect results for final summary
    all_results: list[QueryResult] = []

    # Run for each config
    for config_name in args.config:
        config_path = SCRIPTS_DIR / "configs" / "query" / f"{config_name}.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Query config not found: {config_path}")

        config = load_query_config(config_path)

        # Compute query_run_id once per config (shared across all models/datasets)
        query_run_id = config.get_id()

        # Run query for each dataset and model
        for dataset_spec in config.datasets:
            for model_name in config.models:
                print(f"\n{'=' * 60}")
                print(f"Config: {config_name} | Dataset: {dataset_spec.name} | Model: {model_name}")
                print(f"{'=' * 60}")
                output = query_llm(config, dataset_spec, model_name, output_dir, query_run_id)
                print_output_summary(output, dataset_spec.name)

                # Collect choice counts for final summary
                choices = {"short_term": 0, "long_term": 0, "unknown": 0}
                for p in output.preferences:
                    choices[p.choice] = choices.get(p.choice, 0) + 1

                all_results.append(QueryResult(
                    config_name=config_name,
                    dataset_name=dataset_spec.name,
                    model=output.metadata.model,
                    query_id=output.metadata.query_id,
                    num_samples=len(output.preferences),
                    short_term=choices["short_term"],
                    long_term=choices["long_term"],
                    unknown=choices["unknown"],
                ))

    # Print final summary if multiple queries
    if len(all_results) > 1:
        print_final_summary(all_results)

    if args.profile:
        profiler = get_profiler()
        print()
        print(profiler.summary())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
