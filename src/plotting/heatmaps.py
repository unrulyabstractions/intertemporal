"""
Heatmap visualizations for probe analysis.

Provides functions to create:
- Classification accuracy heatmaps (CV and test)
- Regression MAE heatmaps with unit conversion
- Regression metric heatmaps (R², normalized MAE, MSE)
- Unit comparison bar charts
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import numpy as np

from src.probes import ProbeType, TokenPositionSpec
from .common import TokenPositionInfo, format_token_position_label, get_probe_type_info

if TYPE_CHECKING:
    from src.probes import ProbeTrainingOutput


# =============================================================================
# Helper Functions
# =============================================================================


@dataclass
class HeatmapFigure:
    """Container for heatmap figure components."""

    fig: plt.Figure
    ax: plt.Axes
    text_ax: Optional[plt.Axes]


def _create_heatmap_figure(
    n_layers: int,
    n_positions: int,
) -> HeatmapFigure:
    """
    Create a figure for heatmap visualization.

    Args:
        n_layers: Number of layers (determines height)
        n_positions: Number of token positions (determines width)

    Returns:
        HeatmapFigure with fig and ax
    """
    fig_height = max(6, n_layers * 0.5 + 1)
    fig_width = max(12, n_positions * 1.5)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    return HeatmapFigure(fig=fig, ax=ax, text_ax=None)


def _build_tp_labels(
    tp_indices: list[int],
    token_position_specs: list[TokenPositionSpec],
    tp_info: TokenPositionInfo,
) -> list[str]:
    """Build token position labels for x-axis."""
    tp_labels = []
    for tp_idx in tp_indices:
        spec_label = (
            format_token_position_label(token_position_specs[tp_idx])
            if tp_idx < len(token_position_specs)
            else f"pos_{tp_idx}"
        )
        token = tp_info.tokens.get(tp_idx, "")
        pos = tp_info.resolved_positions.get(tp_idx, "?")
        token_display = repr(token) if token else ""
        tp_labels.append(f"{spec_label}\n[{pos}] {token_display}")
    return tp_labels


def _add_boundary_lines(
    ax: plt.Axes,
    tp_indices: list[int],
    tp_info: TokenPositionInfo,
) -> None:
    """Add vertical boundary lines to heatmap."""
    # Prompt/continuation boundary (cyan)
    if tp_info.prompt_end_idx >= 0:
        boundary_x = None
        for i, tp_idx in enumerate(tp_indices):
            if tp_info.is_prompt.get(tp_idx, False):
                boundary_x = i
        if boundary_x is not None and boundary_x < len(tp_indices) - 1:
            ax.axvline(
                x=boundary_x + 0.5,
                color="#00CED1",
                linestyle="-",
                linewidth=3,
                label="Prompt | Response",
                alpha=0.9,
            )

    # Time horizon injection point (magenta) - line BEFORE this position
    if (
        tp_info.time_horizon_spec_idx >= 0
        and tp_info.time_horizon_spec_idx in tp_indices
    ):
        th_x = tp_indices.index(tp_info.time_horizon_spec_idx)
        ax.axvline(
            x=th_x - 0.5,
            color="#FF1493",
            linestyle="-",
            linewidth=3,
            label="Before | After time horizon",
            alpha=0.9,
        )

    # Choices presented boundary (green) - line BEFORE CONSIDER position
    if (
        tp_info.choices_presented_idx >= 0
        and tp_info.choices_presented_idx in tp_indices
    ):
        cp_x = tp_indices.index(tp_info.choices_presented_idx)
        ax.axvline(
            x=cp_x - 0.5,
            color="#2F8F7A",
            linestyle="-",
            linewidth=3,
            label="Before | After choices presented",
            alpha=0.9,
        )

    # Choice made boundary (orange) - line BEFORE "My reasoning" position
    if (
        tp_info.choice_made_idx >= 0
        and tp_info.choice_made_idx in tp_indices
    ):
        cm_x = tp_indices.index(tp_info.choice_made_idx)
        ax.axvline(
            x=cm_x - 0.5,
            color="#FF8C00",
            linestyle="-",
            linewidth=3,
            label="Before | After choice made",
            alpha=0.9,
        )

    ax.legend(
        loc="lower left", fontsize=7, framealpha=0.9, edgecolor="gray", fancybox=True
    )


def _add_model_info(
    fig: plt.Figure,
    model_name: str,
    metric_label: str,
) -> None:
    """Add model name and metric label to figure."""
    fig.text(
        0.99,
        0.96,
        metric_label,
        ha="right",
        va="top",
        fontsize=9,
        fontweight="bold",
        color="#333333",
        transform=fig.transFigure,
    )
    fig.text(
        0.99,
        0.99,
        f"Model: {model_name}",
        ha="right",
        va="top",
        fontsize=8,
        style="italic",
        color="gray",
        transform=fig.transFigure,
    )


def create_sample_text_figure(
    tp_info: TokenPositionInfo,
    token_position_specs: list[TokenPositionSpec],
    matrix: np.ndarray,
    tp_indices: list[int],
    save_path: Path,
    model_name: str = "",
    title: str = "Sample Text with Probe Accuracy Highlighting",
) -> Optional[Path]:
    """
    Create a separate figure showing sample text with keyword highlighting.

    Keywords from token position specs are colored by their best probe accuracy
    (max across all layers).

    Args:
        tp_info: Token position info containing sample prompt/continuation
        token_position_specs: Token position specifications with keywords
        matrix: Accuracy/metric matrix (layers x positions)
        tp_indices: Token position indices
        save_path: Path to save the figure
        model_name: Model name for annotation
        title: Figure title

    Returns:
        Path to saved figure, or None if no text to visualize
    """
    if not (tp_info.sample_prompt or tp_info.sample_continuation):
        return None

    from matplotlib.colors import Normalize
    import re

    # Colormap matching heatmap
    cmap = plt.cm.RdYlGn
    norm = Normalize(vmin=0.4, vmax=1.0)

    # Build mapping: keyword -> best accuracy (max across layers)
    keyword_accuracies = {}
    for tp_idx in tp_indices:
        col_idx = tp_indices.index(tp_idx)
        col_vals = matrix[:, col_idx]
        valid_vals = col_vals[~np.isnan(col_vals)]
        if len(valid_vals) == 0:
            continue
        best_acc = np.max(valid_vals)
        if tp_idx < len(token_position_specs):
            spec = token_position_specs[tp_idx]
            s = spec.spec
            if isinstance(s, dict) and "text" in s:
                keyword_accuracies[s["text"]] = best_acc

    if not keyword_accuracies:
        return None

    # Build full text - replace newlines and collapse whitespace
    full_text = (tp_info.sample_prompt or "")[:1200]
    full_text = " ".join(full_text.split())  # Collapse all whitespace
    if tp_info.sample_continuation:
        cont = " ".join(tp_info.sample_continuation[:400].split())
        full_text += " >>> " + cont

    # Sort keywords and split text
    keywords_sorted = sorted(keyword_accuracies.keys(), key=len, reverse=True)
    escaped = [re.escape(k) for k in keywords_sorted]
    pattern = "(" + "|".join(escaped) + ")"
    parts = re.split(pattern, full_text)

    # Build lines with word wrapping
    fontsize = 10
    chars_per_line = 100
    lines = []
    current_line = []
    current_len = 0

    for part in parts:
        if not part:
            continue
        # Split on spaces for better word wrapping
        words = part.split(' ')
        for j, word in enumerate(words):
            if j > 0:
                word = ' ' + word
            if current_len + len(word) > chars_per_line and current_line:
                lines.append(current_line)
                current_line = []
                current_len = 0
                word = word.lstrip()
            is_keyword = part in keyword_accuracies
            current_line.append((word, is_keyword, keyword_accuracies.get(part, 0.5)))
            current_len += len(word)
    if current_line:
        lines.append(current_line)

    # Create figure - size based on content
    n_lines = len(lines)
    fig_height = max(4, min(12, 1.5 + n_lines * 0.35))
    fig_width = 14

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Title
    ax.text(0.5, 0.98, title, ha='center', va='top', fontsize=14, fontweight='bold',
            transform=ax.transAxes)

    # Render text lines
    line_height = 0.08
    start_y = 0.90
    char_width = 0.0085

    for i, line_parts in enumerate(lines):
        y_pos = start_y - i * line_height
        if y_pos < 0.05:
            break
        x_pos = 0.02

        for word, is_keyword, acc in line_parts:
            if is_keyword:
                color = cmap(norm(acc))
                weight = 'bold'
                ax.text(
                    x_pos, y_pos, word,
                    ha='left', va='top',
                    fontsize=fontsize, family='monospace',
                    color=color, fontweight=weight,
                    transform=ax.transAxes,
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='#f8f8f8',
                              edgecolor=color, linewidth=1.5, alpha=0.9),
                )
            else:
                ax.text(
                    x_pos, y_pos, word,
                    ha='left', va='top',
                    fontsize=fontsize, family='monospace',
                    color='#333333',
                    transform=ax.transAxes,
                )
            x_pos += len(word) * char_width

    # Add legend for color scale
    legend_y = 0.02
    ax.text(0.02, legend_y, "Accuracy:", ha='left', va='bottom', fontsize=9,
            fontweight='bold', transform=ax.transAxes)
    for acc, label in [(0.5, "50%"), (0.7, "70%"), (0.9, "90%")]:
        color = cmap(norm(acc))
        ax.text(0.12 + (acc - 0.5) * 0.4, legend_y, f"■ {label}", ha='left', va='bottom',
                fontsize=9, color=color, fontweight='bold', transform=ax.transAxes)

    # Model info
    if model_name:
        fig.text(0.99, 0.01, f"Model: {model_name}", ha='right', va='bottom',
                 fontsize=8, style='italic', color='gray')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"  Saved: {save_path}")
    return save_path


def _save_figure(fig: plt.Figure, save_path: Path) -> None:
    """Save figure and close."""
    plt.tight_layout()
    # bbox_inches='tight' will include any text rendered outside the axes
    plt.savefig(
        save_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    plt.close()
    print(f"  Saved: {save_path}")


def _get_layers_and_positions(
    results: list,
    probe_type: ProbeType,
    tp_info: TokenPositionInfo,
) -> tuple[list[int], list[int], list]:
    """
    Extract layers and token positions from results.

    Returns:
        Tuple of (layers, tp_indices, type_results) or empty lists if no results.
    """
    type_results = [r for r in results if r.probe_type == probe_type]
    if not type_results:
        return [], [], []

    layers = sorted(set(r.layer for r in type_results))
    available_tp = set(r.token_position_idx for r in type_results)
    tp_indices = [tp for tp in tp_info.order if tp in available_tp]

    if not layers or not tp_indices:
        return [], [], []

    return layers, tp_indices, type_results


# =============================================================================
# Public API
# =============================================================================


def create_accuracy_heatmap(
    output: "ProbeTrainingOutput",
    probe_type: ProbeType,
    metric: str,  # "cv" or "test"
    token_position_specs: list[TokenPositionSpec],
    tp_info: TokenPositionInfo,
    save_path: Path,
) -> None:
    """Create and save an accuracy heatmap for classification probes."""
    layers, tp_indices, type_results = _get_layers_and_positions(
        output.results, probe_type, tp_info
    )
    if not layers:
        return

    # Build accuracy matrix
    acc_matrix = np.full((len(layers), len(tp_indices)), np.nan)
    for r in type_results:
        layer_idx = layers.index(r.layer)
        if r.token_position_idx in tp_indices:
            tp_idx = tp_indices.index(r.token_position_idx)
            if metric == "cv":
                acc_matrix[layer_idx, tp_idx] = r.cv_accuracy_mean
            elif metric == "test" and r.test_metrics:
                acc_matrix[layer_idx, tp_idx] = r.test_metrics.accuracy

    # Create figure
    hf = _create_heatmap_figure(len(layers), len(tp_indices))

    # Create heatmap
    im = hf.ax.imshow(
        acc_matrix, aspect="auto", cmap="RdYlGn", vmin=0.4, vmax=1.0, origin="lower"
    )
    cbar = plt.colorbar(im, ax=hf.ax)
    cbar.set_label("Accuracy", rotation=270, labelpad=15)

    # Labels
    hf.ax.set_xticks(range(len(tp_indices)))
    hf.ax.set_yticks(range(len(layers)))
    hf.ax.set_xticklabels(
        _build_tp_labels(tp_indices, token_position_specs, tp_info),
        rotation=45,
        ha="right",
        fontsize=8,
    )
    hf.ax.set_yticklabels([f"Layer {l}" for l in layers])
    hf.ax.set_xlabel("Token Position (sequence order)", fontsize=10)
    hf.ax.set_ylabel("Layer", fontsize=10)

    # Title
    title_prefix, emphasized, title_suffix = get_probe_type_info(probe_type)
    hf.ax.set_title(
        f"{title_prefix} $\\bf{{{emphasized}}}$ {title_suffix}", fontsize=12
    )

    # Annotations
    for i in range(len(layers)):
        for j in range(len(tp_indices)):
            val = acc_matrix[i, j]
            if not np.isnan(val):
                color = "white" if val < 0.6 else "black"
                hf.ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=9,
                    fontweight="bold",
                )

    _add_boundary_lines(hf.ax, tp_indices, tp_info)
    _add_model_info(
        hf.fig, output.model_name, "CV Accuracy" if metric == "cv" else "Test Accuracy"
    )
    _save_figure(hf.fig, save_path)

    # Create separate sample text figure with keyword highlighting
    if probe_type != ProbeType.TIME_HORIZON_VALUE:
        # Build intuitive filename: choice_cv_sample_text.png
        base_name = save_path.stem  # e.g., "choice_cv_accuracy"
        text_filename = base_name.replace("_accuracy", "") + "_sample_text.png"
        text_save_path = save_path.parent / text_filename

        probe_name = probe_type.value.replace("_", " ").title()
        metric_name = "CV" if metric == "cv" else "Test"
        title = f"{probe_name} Probe ({metric_name}) - Sample Text Highlighting"

        create_sample_text_figure(
            tp_info=tp_info,
            token_position_specs=token_position_specs,
            matrix=acc_matrix,
            tp_indices=tp_indices,
            save_path=text_save_path,
            model_name=output.model_name,
            title=title,
        )


def create_regression_heatmap(
    output: "ProbeTrainingOutput",
    probe_type: ProbeType,
    metric: str,  # "train" or "test"
    token_position_specs: list[TokenPositionSpec],
    tp_info: TokenPositionInfo,
    save_path: Path,
    unit: str = "months",
) -> None:
    """Create and save a MAE heatmap for regression probes."""
    # Unit conversion factors (from months)
    unit_factors = {
        "hours": 30 * 24,
        "days": 30,
        "weeks": 30 / 7,
        "months": 1,
        "years": 1 / 12,
        "decades": 1 / 120,
    }
    factor = unit_factors.get(unit, 1)

    layers, tp_indices, type_results = _get_layers_and_positions(
        output.results, probe_type, tp_info
    )
    if not layers:
        return

    # Build MAE matrix
    mae_matrix = np.full((len(layers), len(tp_indices)), np.nan)
    for r in type_results:
        layer_idx = layers.index(r.layer)
        if r.token_position_idx in tp_indices:
            tp_idx = tp_indices.index(r.token_position_idx)
            if metric == "train" and r.train_metrics:
                mae_matrix[layer_idx, tp_idx] = r.train_metrics.mae * factor
            elif metric == "test" and r.test_metrics:
                mae_matrix[layer_idx, tp_idx] = r.test_metrics.mae * factor

    # Create figure
    hf = _create_heatmap_figure(len(layers), len(tp_indices))

    # Create heatmap (reversed colormap - lower is better)
    vmax = np.nanmax(mae_matrix) if not np.all(np.isnan(mae_matrix)) else 100
    im = hf.ax.imshow(
        mae_matrix, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=vmax, origin="lower"
    )
    cbar = plt.colorbar(im, ax=hf.ax)
    cbar.set_label(f"MAE ({unit})", rotation=270, labelpad=15)

    # Labels
    hf.ax.set_xticks(range(len(tp_indices)))
    hf.ax.set_yticks(range(len(layers)))
    hf.ax.set_xticklabels(
        _build_tp_labels(tp_indices, token_position_specs, tp_info),
        rotation=45,
        ha="right",
        fontsize=8,
    )
    hf.ax.set_yticklabels([f"Layer {l}" for l in layers])
    hf.ax.set_xlabel("Token Position (sequence order)", fontsize=10)
    hf.ax.set_ylabel("Layer", fontsize=10)

    # Title
    title_prefix, emphasized, title_suffix = get_probe_type_info(probe_type)
    hf.ax.set_title(
        f"{title_prefix} $\\bf{{{emphasized}}}$ {title_suffix} ({unit})", fontsize=12
    )

    # Annotations
    for i in range(len(layers)):
        for j in range(len(tp_indices)):
            val = mae_matrix[i, j]
            if not np.isnan(val):
                color = "white" if val > vmax * 0.6 else "black"
                val_str = (
                    f"{val:.0f}"
                    if val >= 100
                    else (f"{val:.1f}" if val >= 10 else f"{val:.2f}")
                )
                hf.ax.text(
                    j,
                    i,
                    val_str,
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=9,
                    fontweight="bold",
                )

    _add_boundary_lines(hf.ax, tp_indices, tp_info)
    _add_model_info(
        hf.fig, output.model_name, "Train MAE" if metric == "train" else "Test MAE"
    )
    _save_figure(hf.fig, save_path)


def create_regression_metric_heatmap(
    output: "ProbeTrainingOutput",
    probe_type: ProbeType,
    metric_type: str,  # "r2", "normalized_mae", "mse"
    data_split: str,  # "train" or "test"
    token_position_specs: list[TokenPositionSpec],
    tp_info: TokenPositionInfo,
    save_path: Path,
) -> None:
    """Create and save a heatmap for regression metrics (R², normalized MAE, MSE)."""
    layers, tp_indices, type_results = _get_layers_and_positions(
        output.results, probe_type, tp_info
    )
    if not layers:
        return

    # Build metric matrix
    metric_matrix = np.full((len(layers), len(tp_indices)), np.nan)
    for r in type_results:
        layer_idx = layers.index(r.layer)
        if r.token_position_idx in tp_indices:
            tp_idx = tp_indices.index(r.token_position_idx)
            metrics = r.test_metrics if data_split == "test" else r.train_metrics
            if metrics:
                if metric_type == "r2":
                    metric_matrix[layer_idx, tp_idx] = metrics.r2
                elif metric_type == "normalized_mae":
                    metric_matrix[layer_idx, tp_idx] = metrics.normalized_mae
                elif metric_type == "mse":
                    metric_matrix[layer_idx, tp_idx] = metrics.mse

    # Configure colormap based on metric type
    if metric_type == "r2":
        cmap, vmin, vmax = "RdYlGn", -0.5, 1.0
        cbar_label, metric_label = "R² (0 = random baseline)", "R²"
    elif metric_type == "normalized_mae":
        cmap, vmin = "RdYlGn_r", 0
        vmax = np.nanmax(metric_matrix) if not np.all(np.isnan(metric_matrix)) else 2.0
        cbar_label, metric_label = (
            "Normalized MAE (MAE/σ, <1 = better than mean)",
            "Normalized MAE",
        )
    else:  # mse
        cmap, vmin = "RdYlGn_r", 0
        vmax = np.nanmax(metric_matrix) if not np.all(np.isnan(metric_matrix)) else 100
        cbar_label, metric_label = "MSE (L² loss)", "MSE"

    # Create figure
    hf = _create_heatmap_figure(len(layers), len(tp_indices))

    # Create heatmap
    im = hf.ax.imshow(
        metric_matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, origin="lower"
    )
    cbar = plt.colorbar(im, ax=hf.ax)
    cbar.set_label(cbar_label, rotation=270, labelpad=15)

    # Labels
    hf.ax.set_xticks(range(len(tp_indices)))
    hf.ax.set_yticks(range(len(layers)))
    hf.ax.set_xticklabels(
        _build_tp_labels(tp_indices, token_position_specs, tp_info),
        rotation=45,
        ha="right",
        fontsize=8,
    )
    hf.ax.set_yticklabels([f"Layer {l}" for l in layers])
    hf.ax.set_xlabel("Token Position (sequence order)", fontsize=10)
    hf.ax.set_ylabel("Layer", fontsize=10)

    # Title
    title_prefix, emphasized, title_suffix = get_probe_type_info(probe_type)
    hf.ax.set_title(
        f"{title_prefix} $\\bf{{{emphasized}}}$ {title_suffix}", fontsize=12
    )

    # Annotations
    for i in range(len(layers)):
        for j in range(len(tp_indices)):
            val = metric_matrix[i, j]
            if not np.isnan(val):
                if metric_type == "r2":
                    color = "white" if val < 0.3 else "black"
                    val_str = f"{val:.2f}"
                else:
                    color = "white" if val > vmax * 0.6 else "black"
                    val_str = (
                        f"{val:.0f}"
                        if val >= 100
                        else (f"{val:.1f}" if val >= 10 else f"{val:.2f}")
                    )
                hf.ax.text(
                    j,
                    i,
                    val_str,
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=9,
                    fontweight="bold",
                )

    _add_boundary_lines(hf.ax, tp_indices, tp_info)
    split_label = "Train" if data_split == "train" else "Test"
    _add_model_info(hf.fig, output.model_name, f"{split_label} {metric_label}")
    _save_figure(hf.fig, save_path)


def create_unit_comparison_plot(
    output: "ProbeTrainingOutput",
    probe_type: ProbeType,
    metric: str,  # "train" or "test"
    save_path: Path,
) -> None:
    """Create a bar chart comparing MAE converted to days for each original unit."""
    best = output.best_by_type.get(probe_type)
    if not best:
        return

    # Get MAE in months (base unit)
    if metric == "test" and best.test_metrics:
        mae_months = best.test_metrics.mae
    elif best.train_metrics:
        mae_months = best.train_metrics.mae
    else:
        return

    mae_days = mae_months * 30

    units_display = [
        ("Stored as\ndays", mae_days, "#3498DB"),
        ("Stored as\nweeks", mae_days, "#3498DB"),
        ("Stored as\nmonths", mae_days, "#3498DB"),
        ("Stored as\nyears", mae_days, "#3498DB"),
    ]

    unit_names = [u[0] for u in units_display]
    mae_values = [u[1] for u in units_display]
    colors = [u[2] for u in units_display]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        unit_names, mae_values, color=colors, edgecolor="black", linewidth=1.2
    )

    for bar, val in zip(bars, mae_values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + max(mae_values) * 0.02,
            f"{val:.1f} days",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xlabel("Original Storage Unit (all converted to days)", fontsize=12)
    ax.set_ylabel("MAE (days)", fontsize=12)
    metric_name = "Train" if metric == "train" else "Test"
    ax.set_title(
        f"Prediction Error Consistency Across Time Units ({metric_name})\n"
        f"Best probe: Layer {best.layer}, Position {best.token_position_idx}",
        fontsize=14,
    )

    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_ylim(0, mae_days * 1.3)

    ax.axhline(y=mae_days, color="#E74C3C", linestyle="--", linewidth=2, alpha=0.7)
    ax.text(
        0.98,
        mae_days + mae_days * 0.02,
        f"≈ {mae_months:.1f} months",
        ha="right",
        va="bottom",
        fontsize=10,
        color="#E74C3C",
        fontweight="bold",
    )

    fig.text(
        0.99,
        0.02,
        f"Model: {output.model_name}",
        ha="right",
        va="bottom",
        fontsize=8,
        style="italic",
        color="gray",
        transform=fig.transFigure,
    )

    _save_figure(fig, save_path)
