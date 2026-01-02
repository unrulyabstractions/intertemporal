"""
Text visualization for probe analysis.

Provides functions to render sample text with accuracy-based coloring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from src.probes import TokenPositionSpec
from .common import TokenPositionInfo


def render_sample_text_with_accuracy(
    ax,
    fig,
    prompt: str,
    continuation: str,
    token_position_specs: list[TokenPositionSpec],
    tp_info: TokenPositionInfo,
    acc_matrix: np.ndarray,
    tp_indices: list[int],
    y_position: float = -0.15,
) -> None:
    """
    Render sample prompt and continuation below heatmap with tokens colored by accuracy.

    Tokens matching x-axis positions are rendered BOLD with colors from RdYlGn colormap.

    Args:
        ax: Matplotlib axes
        fig: Matplotlib figure
        prompt: Sample prompt text
        continuation: Sample continuation text
        token_position_specs: Token position specifications
        tp_info: Token position info with tokens mapping
        acc_matrix: Accuracy matrix (layers x token_positions)
        tp_indices: Token position indices used in the heatmap
        y_position: Y position in axes coordinates (negative = below plot)
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.colors import Normalize
    import re

    if not prompt and not continuation:
        return

    # Colormap matching heatmap
    cmap = plt.cm.RdYlGn
    norm = Normalize(vmin=0.4, vmax=1.0)

    # Build mapping: keyword -> best accuracy (max across layers)
    keyword_accuracies = {}
    for tp_idx in tp_indices:
        col_idx = tp_indices.index(tp_idx)
        col_vals = acc_matrix[:, col_idx]
        valid_vals = col_vals[~np.isnan(col_vals)]
        if len(valid_vals) == 0:
            continue
        best_acc = np.max(valid_vals)

        # Get text pattern from spec (e.g., "SITUATION:", "TASK:", etc.)
        if tp_idx < len(token_position_specs):
            spec = token_position_specs[tp_idx]
            s = spec.spec
            if isinstance(s, dict) and "text" in s:
                keyword_accuracies[s["text"]] = best_acc

    # Build full text
    full_text = (prompt or "")[:500]
    if continuation:
        full_text += "  >>>  " + continuation[:200]

    if not keyword_accuracies:
        # No keywords - render plain text
        ax.text(0.5, y_position, full_text[:400], ha='center', va='top',
                fontsize=7, family='monospace', color='#555555',
                transform=ax.transAxes, wrap=True,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f8f8',
                         edgecolor='#cccccc', alpha=0.95))
        return

    # Sort keywords by length (longest first) to match longer patterns first
    keywords_sorted = sorted(keyword_accuracies.keys(), key=len, reverse=True)

    # Build regex to split on keywords
    escaped = [re.escape(k) for k in keywords_sorted]
    pattern = '(' + '|'.join(escaped) + ')'
    parts = re.split(pattern, full_text[:600])

    # Layout parameters
    fontsize = 8
    char_width = 0.0052  # Width per character in axes coords
    line_height = 0.055  # Height per line - generous spacing
    max_width = 0.90  # Max line width
    left_margin = 0.05
    padding = 0.035  # Padding inside box

    # First pass: compute total text bounds for background box
    lines = []
    current_line = []
    current_width = 0

    for part in parts:
        if not part:
            continue
        part_width = len(part) * char_width
        if current_width + part_width > max_width and current_line:
            lines.append(current_line)
            current_line = []
            current_width = 0
        current_line.append(part)
        current_width += part_width
    if current_line:
        lines.append(current_line)

    # Draw background box with clean styling
    num_lines = len(lines)
    box_height = num_lines * line_height + padding * 2
    box_x = left_margin - padding
    box_y = y_position - box_height + padding * 0.5
    box_width = max_width + padding * 2

    # Main box with subtle shadow effect
    shadow_box = FancyBboxPatch(
        (box_x + 0.003, box_y - 0.003),
        box_width, box_height,
        boxstyle='round,pad=0.01,rounding_size=0.02',
        facecolor='#e0e0e0', edgecolor='none', alpha=0.4,
        transform=ax.transAxes, zorder=0
    )
    ax.add_patch(shadow_box)

    bg_box = FancyBboxPatch(
        (box_x, box_y),
        box_width, box_height,
        boxstyle='round,pad=0.01,rounding_size=0.02',
        facecolor='#fafafa', edgecolor='#d0d0d0', alpha=0.98,
        transform=ax.transAxes, zorder=1, linewidth=1.2
    )
    ax.add_patch(bg_box)

    # Second pass: render text with colors
    current_y = y_position - padding
    for line_parts in lines:
        current_x = left_margin
        for part in line_parts:
            if not part:
                continue

            is_keyword = part in keyword_accuracies
            if is_keyword:
                color = cmap(norm(keyword_accuracies[part]))
                weight = 'bold'
            else:
                color = '#444444'
                weight = 'normal'

            ax.text(current_x, current_y, part, ha='left', va='top',
                    fontsize=fontsize, family='monospace', color=color,
                    fontweight=weight, transform=ax.transAxes, zorder=2)
            current_x += len(part) * char_width

        current_y -= line_height


def create_text_only_visualization(
    prompt: str,
    continuation: str,
    token_position_specs: list[TokenPositionSpec],
    tp_info: TokenPositionInfo,
    acc_matrix: np.ndarray,
    tp_indices: list[int],
    save_path: Path,
    title: str = "Sample Text with Accuracy Colors",
) -> None:
    """
    Create a standalone visualization showing just the colored sample text.

    This is useful for debugging the text rendering.

    Args:
        prompt: Sample prompt text
        continuation: Sample continuation text
        token_position_specs: Token position specifications
        tp_info: Token position info
        acc_matrix: Accuracy matrix (layers x token_positions)
        tp_indices: Token position indices
        save_path: Path to save the plot
        title: Title for the plot
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.colors import Normalize
    import re

    # Colormap matching heatmap
    cmap = plt.cm.RdYlGn
    norm = Normalize(vmin=0.4, vmax=1.0)

    # Build mapping: keyword -> best accuracy (max across layers)
    keyword_accuracies = {}
    for tp_idx in tp_indices:
        col_idx = tp_indices.index(tp_idx)
        col_vals = acc_matrix[:, col_idx]
        valid_vals = col_vals[~np.isnan(col_vals)]
        if len(valid_vals) == 0:
            continue
        best_acc = np.max(valid_vals)

        # Get text pattern from spec
        if tp_idx < len(token_position_specs):
            spec = token_position_specs[tp_idx]
            s = spec.spec
            if isinstance(s, dict) and "text" in s:
                keyword_accuracies[s["text"]] = best_acc

    # Build full text
    full_text = (prompt or "")[:800]
    if continuation:
        full_text += "\n\n>>> CONTINUATION >>>\n\n" + continuation[:400]

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Title
    ax.text(0.5, 0.98, title, ha='center', va='top', fontsize=14, fontweight='bold',
            transform=ax.transAxes)

    # Legend showing keyword -> accuracy mapping
    legend_y = 0.92
    ax.text(0.02, legend_y, "Keyword Accuracies (max across layers):", ha='left', va='top',
            fontsize=10, fontweight='bold', transform=ax.transAxes)
    legend_y -= 0.03

    for keyword, acc in sorted(keyword_accuracies.items(), key=lambda x: -x[1]):
        color = cmap(norm(acc))
        ax.text(0.04, legend_y, f"* {keyword}: {acc:.2f}", ha='left', va='top',
                fontsize=9, fontweight='bold', color=color, transform=ax.transAxes)
        legend_y -= 0.025

    # Separator
    legend_y -= 0.02
    ax.plot([0.02, 0.98], [legend_y, legend_y], color='#cccccc', linewidth=1,
            transform=ax.transAxes)
    legend_y -= 0.03

    # Now render the full text with colors
    if not keyword_accuracies:
        ax.text(0.5, legend_y, full_text, ha='center', va='top',
                fontsize=9, family='monospace', color='#555555',
                transform=ax.transAxes, wrap=True)
    else:
        # Sort keywords by length (longest first)
        keywords_sorted = sorted(keyword_accuracies.keys(), key=len, reverse=True)
        escaped = [re.escape(k) for k in keywords_sorted]
        pattern = '(' + '|'.join(escaped) + ')'
        parts = re.split(pattern, full_text)

        # Layout parameters
        fontsize = 9
        char_width = 0.0052
        line_height = 0.025
        max_width = 0.94
        left_margin = 0.03

        # Compute lines
        lines = []
        current_line = []
        current_width = 0

        for part in parts:
            if not part:
                continue
            # Handle newlines
            if '\n' in part:
                subparts = part.split('\n')
                for i, subpart in enumerate(subparts):
                    if subpart:
                        part_width = len(subpart) * char_width
                        if current_width + part_width > max_width and current_line:
                            lines.append(current_line)
                            current_line = []
                            current_width = 0
                        current_line.append(subpart)
                        current_width += part_width
                    if i < len(subparts) - 1:  # Not the last subpart
                        if current_line:
                            lines.append(current_line)
                        current_line = []
                        current_width = 0
            else:
                part_width = len(part) * char_width
                if current_width + part_width > max_width and current_line:
                    lines.append(current_line)
                    current_line = []
                    current_width = 0
                current_line.append(part)
                current_width += part_width

        if current_line:
            lines.append(current_line)

        # Draw background
        num_lines = len(lines)
        box_height = num_lines * line_height + 0.03
        bg_box = FancyBboxPatch(
            (left_margin - 0.01, legend_y - box_height - 0.01),
            max_width + 0.02, box_height + 0.02,
            boxstyle='round,pad=0.01,rounding_size=0.02',
            facecolor='#f8f8f8', edgecolor='#cccccc', alpha=0.95,
            transform=ax.transAxes, zorder=0, linewidth=1
        )
        ax.add_patch(bg_box)

        # Render text
        current_y = legend_y - 0.015
        for line_parts in lines:
            current_x = left_margin
            for part in line_parts:
                if not part:
                    continue
                is_keyword = part in keyword_accuracies
                if is_keyword:
                    color = cmap(norm(keyword_accuracies[part]))
                    weight = 'bold'
                else:
                    color = '#555555'
                    weight = 'normal'

                ax.text(current_x, current_y, part, ha='left', va='top',
                        fontsize=fontsize, family='monospace', color=color,
                        fontweight=weight, transform=ax.transAxes, zorder=1)
                current_x += len(part) * char_width
            current_y -= line_height

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"  Saved text visualization: {save_path}")
