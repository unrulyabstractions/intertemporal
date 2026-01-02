"""
Common utilities for probe visualization.

Provides:
- TokenPositionInfo dataclass for storing token position metadata
- Formatting utilities for labels
- Colormap and normalization helpers
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.probes import ProbeType, TokenPositionSpec, ProbeTrainingOutput


@dataclass
class TokenPositionInfo:
    """
    Information about token positions for visualization.

    Attributes:
        order: List of token position indices in sequence order
        tokens: Mapping from token position index to actual token string
        resolved_positions: Mapping from token position index to absolute position
        is_prompt: Mapping from token position index to whether it's in prompt
        prompt_end_idx: Token position index of last prompt token (-1 if unknown)
        time_horizon_spec_idx: Token position index of time horizon specification (-1 if unknown)
        choices_presented_idx: Token position index of "CONSIDER:" keyword (-1 if unknown)
        choice_made_idx: Token position index of "My reasoning:" keyword (-1 if unknown)
        sample_prompt: Sample prompt text for display
        sample_continuation: Sample continuation text for display
    """
    order: list[int]
    tokens: dict[int, str]
    resolved_positions: dict[int, int]
    is_prompt: dict[int, bool]
    prompt_end_idx: int = -1
    time_horizon_spec_idx: int = -1
    choices_presented_idx: int = -1
    choice_made_idx: int = -1
    sample_prompt: str = ""
    sample_continuation: str = ""


def format_token_position_label(spec: TokenPositionSpec) -> str:
    """Format a token position spec as a readable label."""
    s = spec.spec

    if isinstance(s, dict):
        if "text" in s:
            # Truncate long text patterns
            text = s["text"]
            if len(text) > 15:
                text = text[:12] + "..."
            return f'"{text}"'
        elif "relative_to" in s:
            rel = s["relative_to"]
            offset = s.get("offset", 0)
            sign = "+" if offset >= 0 else ""
            return f'{rel}{sign}{offset}'
    elif isinstance(s, int):
        return f'pos_{s}'
    elif isinstance(s, str):
        return s

    return str(s)[:15]


def get_probe_type_info(probe_type: ProbeType) -> tuple[str, str, str]:
    """
    Get display info for a probe type.

    Returns:
        Tuple of (title_prefix, emphasized_word, title_suffix)
    """
    if probe_type == ProbeType.CHOICE:
        return ("Can probe predict", "CHOICE", "(short-term vs long-term)?")
    elif probe_type == ProbeType.TIME_HORIZON_CATEGORY:
        return ("Can probe predict", "TIME\\ HORIZON", "category (<=1yr vs >1yr)?")
    elif probe_type == ProbeType.TIME_HORIZON_VALUE:
        return ("Can probe predict", "TIME\\ HORIZON", "value?")
    else:
        return ("Probe:", probe_type.value, "")


def get_camera_ready_filename(
    output: ProbeTrainingOutput,
    probe_type: ProbeType,
    metric: str,
    unit: str = "",
) -> str:
    """
    Generate a short, descriptive filename for camera-ready plots.

    The filename is short but conveys what the plot supports/proves.

    Args:
        output: Training output with results
        probe_type: Type of probe
        metric: "cv", "test", "train", etc.
        unit: Time unit for regression (e.g., "months", "days")

    Returns:
        Short descriptive filename (without extension)
    """
    # Find best result for this probe type
    best = output.best_by_type.get(probe_type)

    # Short probe type names
    type_short = {
        ProbeType.CHOICE: "choice",
        ProbeType.TIME_HORIZON_CATEGORY: "horizon_cat",
        ProbeType.TIME_HORIZON_VALUE: "horizon_val",
    }
    probe_name = type_short.get(probe_type, probe_type.value)

    # Build filename parts
    parts = [probe_name, metric]

    if probe_type in (ProbeType.CHOICE, ProbeType.TIME_HORIZON_CATEGORY):
        # Add accuracy to filename
        if best:
            if "test" in metric and best.test_metrics:
                acc = best.test_metrics.accuracy
            else:
                acc = best.cv_accuracy_mean
            parts.append(f"{int(acc*100)}pct")
    else:
        # Regression - add MAE and unit
        if unit:
            parts.append(unit)
        if best:
            if "test" in metric and best.test_metrics:
                mae = best.test_metrics.mae
            elif best.train_metrics:
                mae = best.train_metrics.mae
            else:
                mae = 0
            # Convert to unit if needed (base is months)
            unit_factors = {"days": 30, "weeks": 30/7, "months": 1, "years": 1/12}
            factor = unit_factors.get(unit, 1)
            mae_converted = mae * factor
            if mae_converted >= 10:
                parts.append(f"{mae_converted:.0f}err")
            else:
                parts.append(f"{mae_converted:.1f}err")

    return "_".join(parts)


def build_accuracy_matrix(
    results: list,
    probe_type: ProbeType,
    tp_info: TokenPositionInfo,
    metric: str = "cv",
) -> tuple[np.ndarray, list[int], list[int]]:
    """
    Build accuracy matrix for heatmap visualization.

    Args:
        results: List of ProbeResult objects
        probe_type: Type of probe to filter for
        tp_info: Token position information
        metric: "cv" for CV accuracy, "test" for test accuracy

    Returns:
        Tuple of (accuracy_matrix, layers, tp_indices)
    """
    type_results = [r for r in results if r.probe_type == probe_type]
    if not type_results:
        return np.array([[]]), [], []

    layers = sorted(set(r.layer for r in type_results))
    available_tp = set(r.token_position_idx for r in type_results)
    tp_indices = [tp for tp in tp_info.order if tp in available_tp]

    if not layers or not tp_indices:
        return np.array([[]]), [], []

    acc_matrix = np.full((len(layers), len(tp_indices)), np.nan)

    for r in type_results:
        layer_idx = layers.index(r.layer)
        if r.token_position_idx in tp_indices:
            tp_idx = tp_indices.index(r.token_position_idx)
            if metric == "cv":
                acc_matrix[layer_idx, tp_idx] = r.cv_accuracy_mean
            elif metric == "test" and r.test_metrics:
                acc_matrix[layer_idx, tp_idx] = r.test_metrics.accuracy

    return acc_matrix, layers, tp_indices


def build_regression_matrix(
    results: list,
    probe_type: ProbeType,
    tp_info: TokenPositionInfo,
    metric_type: str = "mae",
    data_split: str = "train",
) -> tuple[np.ndarray, list[int], list[int]]:
    """
    Build regression metric matrix for heatmap visualization.

    Args:
        results: List of ProbeResult objects
        probe_type: Type of probe to filter for
        tp_info: Token position information
        metric_type: "mae", "rmse", "mse", "r2", or "normalized_mae"
        data_split: "train" or "test"

    Returns:
        Tuple of (metric_matrix, layers, tp_indices)
    """
    type_results = [r for r in results if r.probe_type == probe_type]
    if not type_results:
        return np.array([[]]), [], []

    layers = sorted(set(r.layer for r in type_results))
    available_tp = set(r.token_position_idx for r in type_results)
    tp_indices = [tp for tp in tp_info.order if tp in available_tp]

    if not layers or not tp_indices:
        return np.array([[]]), [], []

    metric_matrix = np.full((len(layers), len(tp_indices)), np.nan)

    for r in type_results:
        layer_idx = layers.index(r.layer)
        if r.token_position_idx in tp_indices:
            tp_idx = tp_indices.index(r.token_position_idx)
            metrics = r.test_metrics if data_split == "test" else r.train_metrics
            if metrics:
                if metric_type == "mae":
                    metric_matrix[layer_idx, tp_idx] = metrics.mae
                elif metric_type == "rmse":
                    metric_matrix[layer_idx, tp_idx] = metrics.rmse
                elif metric_type == "mse":
                    metric_matrix[layer_idx, tp_idx] = metrics.mse
                elif metric_type == "r2":
                    metric_matrix[layer_idx, tp_idx] = metrics.r2
                elif metric_type == "normalized_mae":
                    metric_matrix[layer_idx, tp_idx] = metrics.normalized_mae

    return metric_matrix, layers, tp_indices
