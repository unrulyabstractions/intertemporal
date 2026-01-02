#!/usr/bin/env python
"""
Generate intertemporal preference dataset.

Usage:
    python scripts/generate_dataset.py --config cityhousing
    python scripts/generate_dataset.py --config cityhousing --formatting default_formatting
    python scripts/generate_dataset.py --config cityhousing --dry-run
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

# Add project root and scripts to path
PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from src.dataset_generator import DatasetGenerator
from src.common.profiling import get_profiler
from src.common.schemas import DatasetRunSpec

from common import save_dataset, GenerateDatasetOutput


# -----------------------------------------------------------------------------
# Input/Output
# -----------------------------------------------------------------------------


@dataclass
class GenerateInput:
    """Input for dataset generation."""

    dataset_config_path: Path
    formatting_config_path: Path
    output_path: Path
    dry_run: bool
    show_profiling: bool


# -----------------------------------------------------------------------------
# Core Logic
# -----------------------------------------------------------------------------


def print_dry_run(samples: list, dataset_id: str) -> None:
    """Print dry run output."""
    print(f"DRY RUN: Would generate {len(samples)} questions")
    print(f"Dataset ID: {dataset_id}")
    for sample in samples[:3]:
        horizon = sample.prompt.question.time_horizon
        print(f"\n{'=' * 60}")
        print(f"Question {sample.id} | Horizon: {horizon}")
        print(f"{'=' * 60}")
        print(sample.prompt.text)
        print(f"\nExpected response format: {sample.prompt.response_format}")
    if len(samples) > 3:
        print(f"\n... and {len(samples) - 3} more questions")


def generate_dataset(inp: GenerateInput, dataset_run_id: str) -> GenerateDatasetOutput:
    """
    Generate dataset from config.

    Args:
        inp: Generation input
        dataset_run_id: Shared ID for all datasets from same --config invocation

    Returns:
        GenerateDatasetOutput with generation info
    """
    generator = DatasetGenerator.from_config_files(
        inp.dataset_config_path,
        inp.formatting_config_path,
    )

    dataset_id = generator.dataset_config.get_id()
    config_dict = generator.dataset_config.to_dict()
    samples, metadata = generator.generate()

    if inp.dry_run:
        print_dry_run(samples, dataset_id)
        return GenerateDatasetOutput(
            dataset_id=dataset_id,
            num_samples=len(samples),
            num_horizons=len(metadata.time_horizons),
            output_path=str(inp.output_path),
        )

    output = save_dataset(samples, dataset_id, dataset_run_id, config_dict, inp.output_path)
    print(f"Dataset saved to {inp.output_path}")
    print(f"  - {len(samples)} questions")
    print(f"  - Dataset ID: {dataset_id}")

    return output


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def get_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate intertemporal preference dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        nargs="*",
        default=["default_dataset"],
        help="Dataset config name(s) (without .json) from configs/dataset/",
    )
    parser.add_argument(
        "--formatting",
        type=str,
        default="default_formatting",
        help="Formatting config name (without .json) from configs/formatting/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file path (default: out/datasets/<name>_<id>.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print samples without saving",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Show profiling summary",
    )
    return parser.parse_args()


def input_from_args(args: argparse.Namespace, config_name: str) -> GenerateInput:
    """Create input from command line arguments."""
    dataset_config_path = SCRIPTS_DIR / "configs" / "dataset" / f"{config_name}.json"
    formatting_config_path = SCRIPTS_DIR / "configs" / "formatting" / f"{args.formatting}.json"

    if not dataset_config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {dataset_config_path}")
    if not formatting_config_path.exists():
        raise FileNotFoundError(f"Formatting config not found: {formatting_config_path}")

    output_path = args.output
    if output_path is None:
        config = DatasetGenerator.load_dataset_config(dataset_config_path)
        config_id = config.get_id()
        output_path = PROJECT_ROOT / "out" / "datasets" / f"{config.name}_{config_id}.json"

    return GenerateInput(
        dataset_config_path=dataset_config_path,
        formatting_config_path=formatting_config_path,
        output_path=output_path,
        dry_run=args.dry_run,
        show_profiling=args.profile,
    )


def print_output(output: GenerateDatasetOutput) -> None:
    """Print output summary."""
    print()
    print("=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Dataset ID: {output.dataset_id}")
    print(f"Samples: {output.num_samples}")
    print(f"Time horizons: {output.num_horizons}")
    print(f"Output: {output.output_path}")


def main() -> int:
    args = get_args()

    # Compute dataset_run_id once for all configs (shared across all datasets)
    run_spec = DatasetRunSpec(
        config_names=tuple(sorted(args.config)),
        formatting_name=args.formatting,
    )
    dataset_run_id = run_spec.get_id()

    for config_name in args.config:
        inp = input_from_args(args, config_name)
        output = generate_dataset(inp, dataset_run_id)

        if not args.dry_run:
            print_output(output)

    if args.profile:
        profiler = get_profiler()
        print()
        print(profiler.summary())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
