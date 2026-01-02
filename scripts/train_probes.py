#!/usr/bin/env python
"""
Train probes to analyze model internals for intertemporal preference choices.

Supports three probe types:
- choice: Binary classification (short_term vs long_term)
- time_horizon_category: Binary classification (<=1yr vs >1yr)
- time_horizon_value: Regression (predict time horizon in months)

Trains probes for all (layer, token_position) combinations and reports results.

Config file options:
    {
      "query_ids": {
        "train_test": ["query_id1", ...]  // Single dataset with auto split
        // OR
        "train_data": ["query_id1", ...],  // Separate train/test
        "test_data": ["query_id2", ...]
      },
      "train_test_split": 0.7,  // Train ratio (default: 0.7)
      "subsample": 0.1          // Optional: subsample ratio for faster iteration (0-1]
    }

Usage:
    python scripts/train_probes.py
    python scripts/train_probes.py --config default_probes
    python scripts/train_probes.py --config my_experiment --probe-types choice time_horizon_category
    python scripts/train_probes.py --camera-ready  # Creates final/ with publication-ready plots
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Add project root and scripts to path
PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from src.common.io import ensure_dir, get_timestamp, load_json, save_json
from src.common.schema_utils import deterministic_id_from_dataclass, SchemaClass
from src.probes import (
    ClassificationMetrics,
    CombinedPreferenceData,
    NoHorizonResults,
    ProbeType,
    ProbeResult,
    ProbeTrainingConfig,
    ProbeTrainingOutput,
    RegressionMetrics,
    TokenPositionSpec,
    class_balanced_train_test_split,
    evaluate_probes_on_no_horizon,
    load_combined_preference_data,
    save_probes,
    train_all_probes,
)
from src.plotting import (
    TokenPositionInfo,
    create_accuracy_heatmap,
    create_regression_heatmap,
    create_regression_metric_heatmap,
    create_text_only_visualization,
    create_unit_comparison_plot,
    format_token_position_label,
    get_camera_ready_filename,
    get_probe_type_info,
    render_sample_text_with_accuracy,
)


# =============================================================================
# Formatting-Dependent Boundary Markers (shared schema)
# =============================================================================

from common.formatting import (
    BoundaryMarkers,
    get_boundary_markers,
    DEFAULT_BOUNDARY_MARKERS,
)


# =============================================================================
# Config Loading
# =============================================================================


def load_probe_config(path: Path) -> dict:
    """Load probe config from JSON file."""
    return load_json(path)


def parse_probe_types(probe_type_strs: list[str]) -> list[ProbeType]:
    """Parse probe type strings to ProbeType enum."""
    type_map = {
        "choice": ProbeType.CHOICE,
        "time_horizon_category": ProbeType.TIME_HORIZON_CATEGORY,
        "time_horizon_value": ProbeType.TIME_HORIZON_VALUE,
    }
    return [type_map[s] for s in probe_type_strs]


@dataclass
class ProbeConfigSchema(SchemaClass):
    """Schema for probe training config - used to generate deterministic folder IDs."""
    train_data: tuple[str, ...]
    test_data: tuple[str, ...]
    train_test: tuple[str, ...]
    train_test_split: float
    subsample: float
    random_seed: int


def get_config_id(
    config_dict: dict,
    subsample_override: float | None = None,
    random_seed: int = 42,
) -> str:
    """Get a deterministic ID for the config using SchemaClass."""
    query_ids = config_dict.get("query_ids", {})

    # Use override if provided, otherwise use config value (default 1.0 = no subsampling)
    subsample = subsample_override if subsample_override is not None else config_dict.get("subsample", 1.0)

    schema = ProbeConfigSchema(
        train_data=tuple(sorted(query_ids.get("train_data", []))),
        test_data=tuple(sorted(query_ids.get("test_data", []))),
        train_test=tuple(sorted(query_ids.get("train_test", []))),
        train_test_split=config_dict.get("train_test_split", 0.7),
        subsample=subsample,
        random_seed=random_seed,
    )

    return schema.get_id()


# =============================================================================
# Output Formatting
# =============================================================================


def serialize_probe_result(result: ProbeResult) -> dict:
    """Serialize a ProbeResult to dict."""
    output = {
        "layer": result.layer,
        "token_position_idx": result.token_position_idx,
        "probe_type": result.probe_type.value,
        "n_train": result.n_train,
        "n_test": result.n_test,
        "n_features": result.n_features,
    }

    # Add classification-specific fields
    if result.probe_type in (ProbeType.CHOICE, ProbeType.TIME_HORIZON_CATEGORY):
        output["cv_accuracy_mean"] = result.cv_accuracy_mean
        output["cv_accuracy_std"] = result.cv_accuracy_std
        if result.train_metrics:
            output["train_accuracy"] = result.train_metrics.accuracy
        if result.test_metrics:
            output["test_accuracy"] = result.test_metrics.accuracy
            output["test_precision"] = result.test_metrics.precision
            output["test_recall"] = result.test_metrics.recall
            output["test_f1"] = result.test_metrics.f1
        if result.confusion_matrix:
            output["confusion_matrix"] = result.confusion_matrix
    else:  # Regression
        if result.train_metrics:
            output["train_mae"] = result.train_metrics.mae
            output["train_rmse"] = result.train_metrics.rmse
            output["train_mse"] = result.train_metrics.mse
            output["train_r2"] = result.train_metrics.r2
            output["train_normalized_mae"] = result.train_metrics.normalized_mae
        if result.test_metrics:
            output["test_mae"] = result.test_metrics.mae
            output["test_rmse"] = result.test_metrics.rmse
            output["test_mse"] = result.test_metrics.mse
            output["test_r2"] = result.test_metrics.r2
            output["test_normalized_mae"] = result.test_metrics.normalized_mae

    return output


def serialize_training_output(output: ProbeTrainingOutput) -> dict:
    """Serialize full training output to dict."""
    results = [serialize_probe_result(r) for r in output.results]

    best_by_type = {}
    for probe_type, result in output.best_by_type.items():
        best_by_type[probe_type.value] = serialize_probe_result(result)

    config_dict = {
        "probe_types": [pt.value for pt in output.config.probe_types],
        "n_cv_folds": output.config.n_cv_folds,
        "random_state": output.config.random_state,
        "regularization_C": output.config.regularization_C,
        "regularization_alpha": output.config.regularization_alpha,
    }

    return {
        "train_data_query_ids": output.train_data_query_ids,
        "test_data_query_ids": output.test_data_query_ids,
        "model": output.model_name,
        "config": config_dict,
        "best_by_type": best_by_type,
        "results": results,
    }


# =============================================================================
# Printing Results
# =============================================================================


def print_results(output: ProbeTrainingOutput) -> None:
    """Print training results summary."""
    print("\n" + "=" * 70)
    print("PROBE TRAINING RESULTS")
    print("=" * 70)
    print(f"Model: {output.model_name}")
    print(f"Train data: {output.train_data_query_ids}")
    print(f"Test data: {output.test_data_query_ids}")
    print()

    # Print results by probe type
    for probe_type in output.config.probe_types:
        type_results = [r for r in output.results if r.probe_type == probe_type]
        if not type_results:
            continue

        print(f"\n{'-' * 70}")
        print(f"Probe Type: {probe_type.value}")
        print(f"{'-' * 70}")

        if probe_type in (ProbeType.CHOICE, ProbeType.TIME_HORIZON_CATEGORY):
            print(f"{'Layer':>6} {'TP Idx':>7} {'CV Acc':>12} {'Test Acc':>10} {'F1':>8}")
            print("-" * 50)
            for r in sorted(type_results, key=lambda x: (x.layer, x.token_position_idx)):
                cv_str = f"{r.cv_accuracy_mean:.3f}+/-{r.cv_accuracy_std:.3f}"
                test_acc = r.test_metrics.accuracy if r.test_metrics else 0.0
                test_f1 = r.test_metrics.f1 if r.test_metrics else 0.0
                print(f"{r.layer:>6} {r.token_position_idx:>7} {cv_str:>12} {test_acc:>10.3f} {test_f1:>8.3f}")
        else:  # Regression
            print(f"{'Layer':>6} {'TP Idx':>7} {'Train MAE':>12} {'Test MAE':>10} {'Test RMSE':>10}")
            print("-" * 55)
            for r in sorted(type_results, key=lambda x: (x.layer, x.token_position_idx)):
                train_mae = r.train_metrics.mae if r.train_metrics else 0.0
                test_mae = r.test_metrics.mae if r.test_metrics else 0.0
                test_rmse = r.test_metrics.rmse if r.test_metrics else 0.0
                print(f"{r.layer:>6} {r.token_position_idx:>7} {train_mae:>12.2f} {test_mae:>10.2f} {test_rmse:>10.2f}")

        # Print best result for this type
        if probe_type in output.best_by_type:
            best = output.best_by_type[probe_type]
            print()
            print(f"  BEST: Layer {best.layer}, Token Position {best.token_position_idx}")
            if probe_type in (ProbeType.CHOICE, ProbeType.TIME_HORIZON_CATEGORY):
                test_acc = best.test_metrics.accuracy if best.test_metrics else best.cv_accuracy_mean
                print(f"        Accuracy: {test_acc:.3f}")
            else:
                test_mae = best.test_metrics.mae if best.test_metrics else best.train_metrics.mae
                print(f"        MAE: {test_mae:.2f} months")

    print("\n" + "=" * 70)


# =============================================================================
# Visualization
# =============================================================================


def get_token_position_info(
    train_data: CombinedPreferenceData,
    query_ids: list[str],
) -> TokenPositionInfo:
    """
    Get comprehensive token position information for visualization.

    Args:
        train_data: Combined preference data
        query_ids: Query IDs used to load the data

    Returns:
        TokenPositionInfo with ordering, tokens, and prompt/continuation info
    """
    from src.probes.data import find_preference_data_by_query_id, load_preference_data_file

    n_positions = len(train_data.token_position_specs)
    default_order = list(range(n_positions))

    if not query_ids:
        return TokenPositionInfo(
            order=default_order,
            resolved_positions={},
            tokens={},
            is_prompt={},
            prompt_end_idx=-1,
            time_horizon_spec_idx=-1,
            choices_presented_idx=-1,
            choice_made_idx=-1,
            sample_prompt="",
            sample_continuation="",
        )

    # Load the first preference file to get resolved positions and tokens
    path = find_preference_data_by_query_id(query_ids[0])
    if path is None:
        return TokenPositionInfo(
            order=default_order,
            resolved_positions={},
            tokens={},
            is_prompt={},
            prompt_end_idx=-1,
            time_horizon_spec_idx=-1,
            choices_presented_idx=-1,
            choice_made_idx=-1,
            sample_prompt="",
            sample_continuation="",
        )

    data = load_preference_data_file(path)
    metadata = data.get("metadata", {})
    sample_prompt = metadata.get("sample_prompt", "")
    sample_continuation = metadata.get("sample_continuation", "")

    # Get boundary markers based on formatting_id
    formatting_id = metadata.get("formatting_id", "")
    markers = get_boundary_markers(formatting_id)
    choices_marker = markers.choices_presented
    choice_made_marker = markers.choice_made

    # Get token_positions and tokens from first sample with internals
    for pref in data["preferences"]:
        internals = pref.get("internals", {})
        if internals.get("token_positions") and internals.get("tokens"):
            resolved_positions = internals["token_positions"]
            tokens_list = internals["tokens"]

            # Build dictionaries
            resolved_dict = {i: pos for i, pos in enumerate(resolved_positions)}
            tokens_dict = {i: tok for i, tok in enumerate(tokens_list)}

            # Determine which positions are prompt vs continuation
            # Also track special marker positions
            is_prompt = {}
            time_horizon_spec_idx = -1
            choices_presented_idx = -1
            choice_made_idx = -1
            for i, spec in enumerate(train_data.token_position_specs):
                s = spec.spec
                if isinstance(s, dict):
                    if s.get("location") == "prompt":
                        is_prompt[i] = True
                    elif s.get("location") == "continuation":
                        is_prompt[i] = False
                    elif "prompt_index" in s:
                        is_prompt[i] = True
                    elif "continuation_index" in s or "index" in s:
                        is_prompt[i] = False
                    else:
                        # Default: assume continuation
                        is_prompt[i] = False

                    # Check for after_time_horizon_spec marker
                    if s.get("after_time_horizon_spec", False):
                        # This position is right after time_horizon_spec injection
                        time_horizon_spec_idx = i

                    # Check for keyword markers (from formatting_id config)
                    text = s.get("text", "")
                    if choices_marker and choices_marker in text:
                        choices_presented_idx = i
                    if choice_made_marker and choice_made_marker.lower() in text.lower():
                        choice_made_idx = i
                elif isinstance(s, int):
                    # Plain int is continuation index
                    is_prompt[i] = False
                else:
                    is_prompt[i] = False

            # Sort by actual sequence position (resolved position in text)
            order = sorted(range(len(resolved_positions)), key=lambda i: resolved_dict[i])

            # Find where prompt ends in the sorted order
            prompt_end_idx = -1
            for i, tp_idx in enumerate(order):
                if is_prompt.get(tp_idx, False):
                    prompt_end_idx = i

            return TokenPositionInfo(
                order=order,
                resolved_positions=resolved_dict,
                tokens=tokens_dict,
                is_prompt=is_prompt,
                prompt_end_idx=prompt_end_idx,
                time_horizon_spec_idx=time_horizon_spec_idx,
                choices_presented_idx=choices_presented_idx,
                choice_made_idx=choice_made_idx,
                sample_prompt=sample_prompt,
                sample_continuation=sample_continuation,
            )

    return TokenPositionInfo(
        order=default_order,
        resolved_positions={},
        tokens={},
        is_prompt={},
        prompt_end_idx=-1,
        time_horizon_spec_idx=-1,
        choices_presented_idx=-1,
        choice_made_idx=-1,
        sample_prompt=sample_prompt,
        sample_continuation=sample_continuation,
    )


def create_visualizations(
    output: ProbeTrainingOutput,
    token_position_specs: list[TokenPositionSpec],
    tp_info: TokenPositionInfo,
    viz_dir: Path,
) -> list[tuple[Path, ProbeType, str]]:
    """
    Create all visualization plots.

    Args:
        output: Training output with results
        token_position_specs: Token position specs for labels
        tp_info: Token position information (order, tokens, etc.)
        viz_dir: Directory to save plots

    Returns:
        List of (path, probe_type, metric) for each created plot
    """
    ensure_dir(viz_dir)

    print("\nCreating visualizations...")
    created_plots = []

    # Time units to generate for regression probes
    time_units = ["months", "days", "weeks", "years"]

    for probe_type in output.config.probe_types:
        type_results = [r for r in output.results if r.probe_type == probe_type]
        if not type_results:
            continue

        if probe_type in (ProbeType.CHOICE, ProbeType.TIME_HORIZON_CATEGORY):
            # Classification probes: CV and test accuracy heatmaps
            cv_path = viz_dir / f"{probe_type.value}_cv_accuracy.png"
            create_accuracy_heatmap(
                output, probe_type, "cv", token_position_specs, tp_info,
                cv_path,
            )
            created_plots.append((cv_path, probe_type, "cv"))

            # Only create test plot if we have test data
            has_test = any(r.test_metrics for r in type_results)
            if has_test:
                test_path = viz_dir / f"{probe_type.value}_test_accuracy.png"
                create_accuracy_heatmap(
                    output, probe_type, "test", token_position_specs, tp_info,
                    test_path,
                )
                created_plots.append((test_path, probe_type, "test"))
        else:
            # Regression probes: MAE heatmaps in multiple units + R², normalized MAE, MSE
            has_test = any(r.test_metrics for r in type_results)

            for unit in time_units:
                # Train MAE
                train_path = viz_dir / f"{probe_type.value}_train_mae_{unit}.png"
                create_regression_heatmap(
                    output, probe_type, "train", token_position_specs, tp_info,
                    train_path, unit=unit,
                )
                created_plots.append((train_path, probe_type, f"train_{unit}"))

                # Test MAE
                if has_test:
                    test_path = viz_dir / f"{probe_type.value}_test_mae_{unit}.png"
                    create_regression_heatmap(
                        output, probe_type, "test", token_position_specs, tp_info,
                        test_path, unit=unit,
                    )
                    created_plots.append((test_path, probe_type, f"test_{unit}"))

            # R², normalized MAE, and MSE heatmaps
            for metric_type in ["r2", "normalized_mae", "mse"]:
                # Train
                train_path = viz_dir / f"{probe_type.value}_train_{metric_type}.png"
                create_regression_metric_heatmap(
                    output, probe_type, metric_type, "train",
                    token_position_specs, tp_info, train_path,
                )
                created_plots.append((train_path, probe_type, f"train_{metric_type}"))

                # Test
                if has_test:
                    test_path = viz_dir / f"{probe_type.value}_test_{metric_type}.png"
                    create_regression_metric_heatmap(
                        output, probe_type, metric_type, "test",
                        token_position_specs, tp_info, test_path,
                    )
                    created_plots.append((test_path, probe_type, f"test_{metric_type}"))

            # Unit comparison bar chart
            unit_comparison_path = viz_dir / f"{probe_type.value}_unit_comparison_train.png"
            create_unit_comparison_plot(output, probe_type, "train", unit_comparison_path)
            created_plots.append((unit_comparison_path, probe_type, "unit_comparison_train"))

            if has_test:
                unit_comparison_path = viz_dir / f"{probe_type.value}_unit_comparison_test.png"
                create_unit_comparison_plot(output, probe_type, "test", unit_comparison_path)
                created_plots.append((unit_comparison_path, probe_type, "unit_comparison_test"))

    # Create text-only visualization for debugging (using choice probe accuracy)
    if tp_info.sample_prompt or tp_info.sample_continuation:
        # Find choice results for coloring
        choice_results = [r for r in output.results if r.probe_type == ProbeType.CHOICE]
        if choice_results:
            layers = sorted(set(r.layer for r in choice_results))
            available_tp = set(r.token_position_idx for r in choice_results)
            tp_indices = [tp for tp in tp_info.order if tp in available_tp]

            if layers and tp_indices:
                # Build accuracy matrix
                acc_matrix = np.full((len(layers), len(tp_indices)), np.nan)
                for r in choice_results:
                    layer_idx = layers.index(r.layer)
                    if r.token_position_idx in tp_indices:
                        tp_idx = tp_indices.index(r.token_position_idx)
                        acc_matrix[layer_idx, tp_idx] = r.cv_accuracy_mean

                text_viz_path = viz_dir / "sample_text_colored.png"
                create_text_only_visualization(
                    tp_info.sample_prompt,
                    tp_info.sample_continuation,
                    token_position_specs,
                    tp_info,
                    acc_matrix,
                    tp_indices,
                    text_viz_path,
                    title="Sample Text with CHOICE Probe Accuracy Colors",
                )
                created_plots.append((text_viz_path, ProbeType.CHOICE, "text_viz"))

    return created_plots


def create_no_horizon_visualizations(
    no_horizon_results: NoHorizonResults,
    output: ProbeTrainingOutput,
    token_position_specs: list[TokenPositionSpec],
    tp_info: TokenPositionInfo,
    viz_dir: Path,
) -> list[Path]:
    """
    Create heatmaps showing probe performance on no-horizon samples only.

    Args:
        no_horizon_results: Results from evaluating probes on no-horizon samples
        output: Training output (for model name and other metadata)
        token_position_specs: Token position specs for labels
        tp_info: Token position information
        viz_dir: Directory to save plots (will create no_horizon/ subfolder)

    Returns:
        List of paths to created plots
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    no_horizon_dir = viz_dir / "no_horizon"
    ensure_dir(no_horizon_dir)

    if not no_horizon_results.results:
        print("  No no-horizon results to visualize")
        return []

    created_plots = []

    # Get layers and token positions from results
    layers = sorted(set(r.layer for r in no_horizon_results.results.values()))
    available_tp = set(r.token_position_idx for r in no_horizon_results.results.values())
    tp_indices = [tp for tp in tp_info.order if tp in available_tp]

    if not layers or not tp_indices:
        return []

    # Build accuracy matrices
    train_acc_matrix = np.full((len(layers), len(tp_indices)), np.nan)
    test_acc_matrix = np.full((len(layers), len(tp_indices)), np.nan)

    for result in no_horizon_results.results.values():
        layer_idx = layers.index(result.layer)
        if result.token_position_idx in tp_indices:
            tp_idx = tp_indices.index(result.token_position_idx)
            train_acc_matrix[layer_idx, tp_idx] = result.train_accuracy
            if result.test_accuracy is not None:
                test_acc_matrix[layer_idx, tp_idx] = result.test_accuracy

    # Helper to build token position labels
    def build_tp_labels():
        labels = []
        for tp_idx in tp_indices:
            spec_label = (
                format_token_position_label(token_position_specs[tp_idx])
                if tp_idx < len(token_position_specs)
                else f"pos_{tp_idx}"
            )
            token = tp_info.tokens.get(tp_idx, "")
            pos = tp_info.resolved_positions.get(tp_idx, "?")
            token_display = repr(token) if token else ""
            labels.append(f"{spec_label}\n[{pos}] {token_display}")
        return labels

    tp_labels = build_tp_labels()

    # Create train accuracy heatmap
    fig_height = max(6, len(layers) * 0.5 + 1)
    fig_width = max(12, len(tp_indices) * 1.5)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(
        train_acc_matrix, aspect="auto", cmap="RdYlGn", vmin=0.4, vmax=1.0, origin="lower"
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Accuracy", rotation=270, labelpad=15)

    ax.set_xticks(range(len(tp_indices)))
    ax.set_yticks(range(len(layers)))
    ax.set_xticklabels(tp_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([f"Layer {l}" for l in layers])
    ax.set_xlabel("Token Position (sequence order)", fontsize=10)
    ax.set_ylabel("Layer", fontsize=10)
    ax.set_title(
        f"CHOICE Probe Accuracy on $\\bf{{No-Horizon}}$ Samples (Train, n={no_horizon_results.n_train})",
        fontsize=12,
    )

    # Annotations
    for i in range(len(layers)):
        for j in range(len(tp_indices)):
            val = train_acc_matrix[i, j]
            if not np.isnan(val):
                color = "white" if val < 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9, fontweight="bold")

    # Model info
    fig.text(0.99, 0.99, f"Model: {output.model_name}", ha="right", va="top",
             fontsize=8, style="italic", color="gray", transform=fig.transFigure)
    fig.text(0.99, 0.96, "Train Accuracy (No-Horizon)", ha="right", va="top",
             fontsize=9, fontweight="bold", color="#333333", transform=fig.transFigure)

    plt.tight_layout()
    train_path = no_horizon_dir / "choice_no_horizon_train_accuracy.png"
    plt.savefig(train_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"  Saved: {train_path}")
    created_plots.append(train_path)

    # Create test accuracy heatmap if we have test data
    has_test = any(r.test_accuracy is not None for r in no_horizon_results.results.values())
    if has_test and no_horizon_results.n_test > 0:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        im = ax.imshow(
            test_acc_matrix, aspect="auto", cmap="RdYlGn", vmin=0.4, vmax=1.0, origin="lower"
        )
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Accuracy", rotation=270, labelpad=15)

        ax.set_xticks(range(len(tp_indices)))
        ax.set_yticks(range(len(layers)))
        ax.set_xticklabels(tp_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels([f"Layer {l}" for l in layers])
        ax.set_xlabel("Token Position (sequence order)", fontsize=10)
        ax.set_ylabel("Layer", fontsize=10)
        ax.set_title(
            f"CHOICE Probe Accuracy on $\\bf{{No-Horizon}}$ Samples (Test, n={no_horizon_results.n_test})",
            fontsize=12,
        )

        # Annotations
        for i in range(len(layers)):
            for j in range(len(tp_indices)):
                val = test_acc_matrix[i, j]
                if not np.isnan(val):
                    color = "white" if val < 0.6 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9, fontweight="bold")

        # Model info
        fig.text(0.99, 0.99, f"Model: {output.model_name}", ha="right", va="top",
                 fontsize=8, style="italic", color="gray", transform=fig.transFigure)
        fig.text(0.99, 0.96, "Test Accuracy (No-Horizon)", ha="right", va="top",
                 fontsize=9, fontweight="bold", color="#333333", transform=fig.transFigure)

        plt.tight_layout()
        test_path = no_horizon_dir / "choice_no_horizon_test_accuracy.png"
        plt.savefig(test_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close()
        print(f"  Saved: {test_path}")
        created_plots.append(test_path)

    return created_plots


def create_camera_ready_plots(
    output: ProbeTrainingOutput,
    token_position_specs: list[TokenPositionSpec],
    tp_info: TokenPositionInfo,
    final_dir: Path,
) -> None:
    """
    Create ALL camera-ready plots with short descriptive filenames.

    Creates the same plots as create_visualizations() but with shorter,
    claim-focused filenames suitable for publication.

    Args:
        output: Training output with results
        token_position_specs: Token position specs for labels
        tp_info: Token position information
        final_dir: Directory for camera-ready plots
    """
    ensure_dir(final_dir)

    print("\nCreating camera-ready plots in final/...")

    # Time units for regression (same as create_visualizations)
    time_units = ["months", "days", "weeks", "years"]

    for probe_type in output.config.probe_types:
        type_results = [r for r in output.results if r.probe_type == probe_type]
        if not type_results:
            continue

        if probe_type in (ProbeType.CHOICE, ProbeType.TIME_HORIZON_CATEGORY):
            # CV accuracy
            filename = get_camera_ready_filename(output, probe_type, "cv")
            create_accuracy_heatmap(
                output, probe_type, "cv", token_position_specs, tp_info,
                final_dir / f"{filename}.png",
            )

            # Test accuracy if available
            has_test = any(r.test_metrics for r in type_results)
            if has_test:
                filename = get_camera_ready_filename(output, probe_type, "test")
                create_accuracy_heatmap(
                    output, probe_type, "test", token_position_specs, tp_info,
                    final_dir / f"{filename}.png",
                )
        else:
            # Regression probes: create plots for all time units
            has_test = any(r.test_metrics for r in type_results)

            for unit in time_units:
                # Train MAE
                filename = get_camera_ready_filename(output, probe_type, "train", unit=unit)
                create_regression_heatmap(
                    output, probe_type, "train", token_position_specs, tp_info,
                    final_dir / f"{filename}.png",
                    unit=unit,
                )

                # Test MAE if available
                if has_test:
                    filename = get_camera_ready_filename(output, probe_type, "test", unit=unit)
                    create_regression_heatmap(
                        output, probe_type, "test", token_position_specs, tp_info,
                        final_dir / f"{filename}.png",
                        unit=unit,
                    )

            # R², normalized MAE, and MSE heatmaps
            for metric_type in ["r2", "normalized_mae", "mse"]:
                # Train
                create_regression_metric_heatmap(
                    output, probe_type, metric_type, "train",
                    token_position_specs, tp_info,
                    final_dir / f"horizon_val_train_{metric_type}.png",
                )
                # Test
                if has_test:
                    create_regression_metric_heatmap(
                        output, probe_type, metric_type, "test",
                        token_position_specs, tp_info,
                        final_dir / f"horizon_val_test_{metric_type}.png",
                    )

            # Unit comparison bar chart
            create_unit_comparison_plot(
                output, probe_type, "train",
                final_dir / "horizon_val_unit_comparison_train.png"
            )
            if has_test:
                create_unit_comparison_plot(
                    output, probe_type, "test",
                    final_dir / "horizon_val_unit_comparison_test.png"
                )


# =============================================================================
# Subsampling
# =============================================================================


def subsample_data(
    data: CombinedPreferenceData,
    ratio: float,
    random_seed: int = 42,
) -> CombinedPreferenceData:
    """
    Subsample preference data for faster iteration.

    Args:
        data: Combined preference data
        ratio: Fraction of samples to keep (0, 1]
        random_seed: Random seed for reproducibility

    Returns:
        Subsampled CombinedPreferenceData
    """
    import random

    if ratio >= 1.0:
        return data

    random.seed(random_seed)
    n_keep = max(1, int(len(data.samples) * ratio))
    sampled = random.sample(data.samples, n_keep)

    return CombinedPreferenceData(
        samples=sampled,
        layers=data.layers,
        token_position_specs=data.token_position_specs,
        model=data.model,
        d_model=data.d_model,
    )


# =============================================================================
# Main Pipeline
# =============================================================================


def run_probe_training(
    config_path: Path,
    probe_types: list[ProbeType],
    output_dir: Path,
    random_seed: int = 42,
    subsample_override: float | None = None,
) -> tuple[ProbeTrainingOutput, CombinedPreferenceData, Optional[CombinedPreferenceData], dict, TokenPositionInfo]:
    """
    Run full probe training pipeline.

    Args:
        config_path: Path to probe config JSON
        probe_types: List of probe types to train
        output_dir: Directory for output files
        random_seed: Random seed for reproducibility
        subsample_override: If provided, overrides config subsample value

    Returns:
        Tuple of (ProbeTrainingOutput, train_data, test_data, config_dict, tp_info)
    """
    # Load config
    config_dict = load_probe_config(config_path)
    query_ids = config_dict.get("query_ids", {})
    train_query_ids = query_ids.get("train_data", [])
    test_query_ids = query_ids.get("test_data", [])
    train_test_query_ids = query_ids.get("train_test", [])
    train_test_split_ratio = config_dict.get("train_test_split", 0.7)
    subsample_ratio = subsample_override if subsample_override is not None else config_dict.get("subsample", 1.0)

    # Validate config - must have either train_data or train_test
    if not train_query_ids and not train_test_query_ids:
        raise ValueError("Config must specify either 'train_data' or 'train_test' query_ids")

    print(f"Config: {config_path}")
    if subsample_ratio < 1.0:
        print(f"Subsampling: {subsample_ratio:.0%} of data (for faster iteration)")

    # Handle train_test mode (class-balanced split from single dataset)
    if train_test_query_ids:
        print(f"Train/test query IDs: {train_test_query_ids}")
        print(f"Train/test split ratio: {train_test_split_ratio}")
        print()

        # Load all data
        print("Loading data for train/test split...")
        all_data = load_combined_preference_data(train_test_query_ids)
        print(f"  Loaded {len(all_data.samples)} samples from {all_data.model}")
        print(f"  Layers: {all_data.layers}")
        print(f"  Token positions: {len(all_data.token_position_specs)}")
        print(f"  Hidden dim: {all_data.d_model}")

        # Apply subsampling before split (for faster iteration)
        if subsample_ratio < 1.0:
            all_data = subsample_data(all_data, subsample_ratio, random_seed)
            print(f"  After subsampling: {len(all_data.samples)} samples")

        # Class-balanced split
        print(f"\nSplitting data ({train_test_split_ratio:.0%} train, {1-train_test_split_ratio:.0%} test)...")
        train_data, test_data = class_balanced_train_test_split(
            all_data,
            train_ratio=train_test_split_ratio,
            random_seed=random_seed,
        )
        print(f"  Train: {len(train_data.samples)} samples")
        print(f"  Test: {len(test_data.samples)} samples")

        # Print class distribution for choice probe
        train_choices = {"short_term": 0, "long_term": 0, "unknown": 0}
        test_choices = {"short_term": 0, "long_term": 0, "unknown": 0}
        for s in train_data.samples:
            train_choices[s.choice] = train_choices.get(s.choice, 0) + 1
        for s in test_data.samples:
            test_choices[s.choice] = test_choices.get(s.choice, 0) + 1
        print(f"  Train choices: {train_choices}")
        print(f"  Test choices: {test_choices}")

        # Print class distribution for time horizon category probe
        from src.probes.data import categorize_time_horizon
        train_th_cat = {"short(<=1yr)": 0, "long(>1yr)": 0, "null": 0}
        test_th_cat = {"short(<=1yr)": 0, "long(>1yr)": 0, "null": 0}
        for s in train_data.samples:
            if s.time_horizon is None:
                train_th_cat["null"] += 1
            elif categorize_time_horizon(s.time_horizon) == 0:
                train_th_cat["short(<=1yr)"] += 1
            else:
                train_th_cat["long(>1yr)"] += 1
        for s in test_data.samples:
            if s.time_horizon is None:
                test_th_cat["null"] += 1
            elif categorize_time_horizon(s.time_horizon) == 0:
                test_th_cat["short(<=1yr)"] += 1
            else:
                test_th_cat["long(>1yr)"] += 1
        print(f"  Train time horizon: {train_th_cat}")
        print(f"  Test time horizon: {test_th_cat}")

        # Use train_test_query_ids for both
        train_query_ids = train_test_query_ids
        test_query_ids = train_test_query_ids

    else:
        # Standard mode: separate train and test datasets
        print(f"Train query IDs: {train_query_ids}")
        print(f"Test query IDs: {test_query_ids}")
        print()

        # Load data
        print("Loading training data...")
        train_data = load_combined_preference_data(train_query_ids)
        print(f"  Loaded {len(train_data.samples)} samples from {train_data.model}")
        print(f"  Layers: {train_data.layers}")
        print(f"  Token positions: {len(train_data.token_position_specs)}")
        print(f"  Hidden dim: {train_data.d_model}")

        # Apply subsampling (for faster iteration)
        if subsample_ratio < 1.0:
            train_data = subsample_data(train_data, subsample_ratio, random_seed)
            print(f"  After subsampling: {len(train_data.samples)} samples")

        test_data = None
        if test_query_ids:
            print("Loading test data...")
            test_data = load_combined_preference_data(test_query_ids)
            print(f"  Loaded {len(test_data.samples)} samples")
            if subsample_ratio < 1.0:
                test_data = subsample_data(test_data, subsample_ratio, random_seed)
                print(f"  After subsampling: {len(test_data.samples)} samples")

    # Create training config
    training_config = ProbeTrainingConfig(
        probe_types=probe_types,
        n_cv_folds=5,
        random_state=random_seed,
    )

    # Train all probes
    print("\nTraining probes...")
    output = train_all_probes(train_data, test_data, training_config)

    # Fill in query IDs
    output.train_data_query_ids = train_query_ids
    output.test_data_query_ids = test_query_ids

    # Print results
    print_results(output)

    # Get token position info for visualizations
    tp_info = get_token_position_info(train_data, train_query_ids)

    return output, train_data, test_data, config_dict, tp_info


# =============================================================================
# CLI
# =============================================================================


def get_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train probes to analyze model internals for intertemporal choices"
    )
    parser.add_argument(
        "--config",
        type=str,
        nargs="*",
        default=["default_probes"],
        help="Probe config name(s) from configs/probes/ (default: default_probes)",
    )
    parser.add_argument(
        "--probe-types",
        type=str,
        nargs="*",
        default=["choice", "time_horizon_category", "time_horizon_value"],
        choices=["choice", "time_horizon_category", "time_horizon_value"],
        help="Probe types to train (default: all)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: out/probes/)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--camera-ready",
        action="store_true",
        help="Create final/ folder with publication-ready plots with descriptive filenames",
    )
    parser.add_argument(
        "--save-best-only",
        action="store_true",
        help="Only save best probe per type (default: save all probes)",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=None,
        help="Subsample ratio (0-1] for faster iteration. Overrides config value.",
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse existing trained probes and results instead of retraining. "
             "Looks for existing data in the config ID folder.",
    )
    return parser.parse_args()


def load_existing_results(
    run_dir: Path,
    config_path: Path,
) -> tuple[ProbeTrainingOutput, CombinedPreferenceData, TokenPositionInfo] | None:
    """
    Load existing probe training results from a run directory.

    Args:
        run_dir: Directory containing results/, viz/, probes/ folders
        config_path: Path to config file (to reload data for tp_info)

    Returns:
        Tuple of (output, train_data, tp_info) or None if not found
    """
    results_dir = run_dir / "results"
    if not results_dir.exists():
        print(f"No results directory found at {results_dir}")
        return None

    # Find the most recent results file
    result_files = sorted(results_dir.glob("probe_results_*.json"), reverse=True)
    if not result_files:
        print(f"No result files found in {results_dir}")
        return None

    result_path = result_files[0]
    print(f"Loading existing results from: {result_path}")

    result_dict = load_json(result_path)

    # Reconstruct ProbeTrainingOutput from saved results
    # We need to reload the training data to get token_position_specs and tp_info
    config_dict = load_json(config_path)

    query_ids = config_dict.get("query_ids", {})
    train_query_ids = query_ids.get("train", [])
    test_query_ids = query_ids.get("test", [])
    train_test_query_ids = query_ids.get("train_test", [])

    # Load data to get token_position_specs
    if train_test_query_ids:
        print("Loading data for token position info...")
        full_data = load_combined_preference_data(train_test_query_ids)
        train_data = full_data  # Use full data for specs
        actual_train_ids = train_test_query_ids
    else:
        print("Loading training data for token position info...")
        train_data = load_combined_preference_data(train_query_ids)
        actual_train_ids = train_query_ids

    # Get tp_info
    tp_info = get_token_position_info(train_data, actual_train_ids)

    # Reconstruct ProbeTrainingOutput
    # probe_types is stored in config.probe_types in the JSON
    config_data = result_dict.get("config", {})
    probe_types = [ProbeType(pt) for pt in config_data.get("probe_types", [])]
    training_config = ProbeTrainingConfig(
        probe_types=probe_types,
        n_cv_folds=5,
        random_state=42,
    )

    # Reconstruct results
    results = []
    for r in result_dict.get("results", []):
        probe_type = ProbeType(r["probe_type"])

        # Reconstruct metrics
        train_metrics = None
        test_metrics = None

        if probe_type in (ProbeType.CHOICE, ProbeType.TIME_HORIZON_CATEGORY):
            if "train_accuracy" in r:
                train_metrics = ClassificationMetrics(
                    accuracy=r.get("train_accuracy", 0),
                    precision=r.get("train_precision", 0),
                    recall=r.get("train_recall", 0),
                    f1=r.get("train_f1", 0),
                    n_samples=r.get("n_train", 0),
                )
            if "test_accuracy" in r:
                test_metrics = ClassificationMetrics(
                    accuracy=r.get("test_accuracy", 0),
                    precision=r.get("test_precision", 0),
                    recall=r.get("test_recall", 0),
                    f1=r.get("test_f1", 0),
                    n_samples=r.get("n_test", 0),
                )
        else:
            if "train_mae" in r:
                train_metrics = RegressionMetrics(
                    mae=r.get("train_mae", 0),
                    rmse=r.get("train_rmse", 0),
                    mse=r.get("train_mse", 0),
                    r2=r.get("train_r2", 0),
                    normalized_mae=r.get("train_normalized_mae", 0),
                    n_samples=r.get("n_train", 0),
                )
            if "test_mae" in r:
                test_metrics = RegressionMetrics(
                    mae=r.get("test_mae", 0),
                    rmse=r.get("test_rmse", 0),
                    mse=r.get("test_mse", 0),
                    r2=r.get("test_r2", 0),
                    normalized_mae=r.get("test_normalized_mae", 0),
                    n_samples=r.get("n_test", 0),
                )

        result = ProbeResult(
            layer=r["layer"],
            token_position_idx=r["token_position_idx"],
            probe_type=probe_type,
            cv_accuracy_mean=r.get("cv_accuracy_mean", 0),
            cv_accuracy_std=r.get("cv_accuracy_std", 0),
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            confusion_matrix=r.get("confusion_matrix"),
            n_train=r.get("n_train", 0),
            n_test=r.get("n_test", 0),
            n_features=r.get("n_features", 0),
        )
        results.append(result)

    # Build best_by_type
    best_by_type = {}
    for pt in probe_types:
        type_results = [r for r in results if r.probe_type == pt]
        if type_results:
            if pt in (ProbeType.CHOICE, ProbeType.TIME_HORIZON_CATEGORY):
                best = max(type_results, key=lambda r: r.cv_accuracy_mean)
            else:
                best = min(type_results, key=lambda r: r.train_metrics.mae if r.train_metrics else float('inf'))
            best_by_type[pt] = best

    output = ProbeTrainingOutput(
        model_name=result_dict.get("model_name", "Unknown"),
        results=results,
        trained_probes=[],  # Not available when loading from JSON
        best_by_type=best_by_type,
        config=training_config,
        train_data_query_ids=result_dict.get("train_data_query_ids", []),
        test_data_query_ids=result_dict.get("test_data_query_ids", []),
    )

    print(f"  Loaded {len(results)} probe results")
    return output, train_data, tp_info


def main() -> int:
    args = get_args()

    # Output directory
    output_dir = args.output
    if output_dir is None:
        output_dir = PROJECT_ROOT / "out" / "probes"
    ensure_dir(output_dir)

    # Parse probe types
    probe_types = parse_probe_types(args.probe_types)

    for config_name in args.config:
        config_path = SCRIPTS_DIR / "configs" / "probes" / f"{config_name}.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        # Load config to compute config_id
        config_dict = load_json(config_path)
        config_id = get_config_id(
            config_dict,
            subsample_override=args.subsample,
            random_seed=args.seed,
        )
        run_dir = output_dir / config_id

        # Check if we should reuse existing results
        reuse_success = False
        test_data = None
        if args.reuse:
            print(f"Config: {config_path}")
            print(f"Looking for existing results in: {run_dir}")
            loaded = load_existing_results(run_dir, config_path)
            if loaded:
                output, train_data, tp_info = loaded
                reuse_success = True
                print(f"Successfully loaded existing results (regenerate=False)")
            else:
                print("No existing results found, will train from scratch")

        # Train if not reusing or reuse failed
        if not reuse_success:
            output, train_data, test_data, config_dict, tp_info = run_probe_training(
                config_path=config_path,
                probe_types=probe_types,
                output_dir=output_dir,
                random_seed=args.seed,
                subsample_override=args.subsample,
            )

        ensure_dir(run_dir)

        # Create subfolders for organization
        results_dir = run_dir / "results"
        viz_dir = run_dir / "viz"
        ensure_dir(results_dir)
        ensure_dir(viz_dir)

        # Only save results if we trained (not reusing)
        if not reuse_success:
            timestamp = get_timestamp()
            output_dict = serialize_training_output(output)
            output_dict["config_path"] = str(config_path.relative_to(PROJECT_ROOT))
            output_dict["timestamp"] = timestamp

            # Add train_test mode info if applicable
            query_ids = config_dict.get("query_ids", {})
            if query_ids.get("train_test"):
                output_dict["train_test_mode"] = True
                output_dict["train_test_split"] = config_dict.get("train_test_split", 0.7)

            output_path = results_dir / f"probe_results_{timestamp}.json"
            save_json(output_dict, output_path)
            print(f"\nResults saved to: {output_path}")

            # Save trained probes for future experiments (e.g., steering)
            probes_model_dir = run_dir / "probes"
            print("\nSaving trained probes...")
            save_probes(output, probes_model_dir, save_all=not args.save_best_only, tp_info=tp_info)
            print(f"Probes saved to: {probes_model_dir}")

        # Always create visualizations (whether training or reusing)
        print("\nCreating visualizations...")
        create_visualizations(output, train_data.token_position_specs, tp_info, viz_dir)

        # Create no-horizon visualizations (evaluate probes on samples without time horizon)
        if not reuse_success and output.trained_probes:
            print("\nEvaluating probes on no-horizon samples...")
            no_horizon_results = evaluate_probes_on_no_horizon(output, train_data, test_data)
            if no_horizon_results.n_train > 0:
                print(f"  Found {no_horizon_results.n_train} train / {no_horizon_results.n_test} test no-horizon samples")
                create_no_horizon_visualizations(
                    no_horizon_results, output, train_data.token_position_specs, tp_info, viz_dir
                )
            else:
                print("  No no-horizon samples found in data")

        # Create camera-ready plots if requested
        if args.camera_ready:
            final_dir = run_dir / "final"
            create_camera_ready_plots(
                output, train_data.token_position_specs, tp_info, final_dir
            )

            # Copy final/ to paper/ folder at project root
            paper_dir = PROJECT_ROOT / "paper" / "probes"
            ensure_dir(paper_dir)
            import shutil
            for f in final_dir.glob("*.png"):
                shutil.copy2(f, paper_dir / f.name)
            print(f"\nCopied camera-ready plots to: {paper_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
