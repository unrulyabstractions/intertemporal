#!/usr/bin/env python
"""
Analyze LLM choices by training and testing choice models.

Trains value function models on preference data, stratified by time horizon,
and evaluates predictions across different horizon buckets.

Usage:
    python scripts/analyze_choices.py
    python scripts/analyze_choices.py --config default_analysis
    python scripts/analyze_choices.py --config default_analysis --camera-ready
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add project root and scripts to path
PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from src.common.io import ensure_dir, get_timestamp, save_json
from src.common.profiling import get_profiler

# Import from refactored src/choice module
from src.choice import (
    ALL_MODEL_SPECS,
    AnalysisConfig,
    AnalysisResult,
    HorizonModelResult,
    LoadedSamples,
    SampleWithHorizon,
    bucket_samples_by_horizon,
    class_balanced_train_test_split,
    filter_problematic_samples,
    find_preference_data_by_query_id,
    load_analysis_config,
    print_summary,
    serialize_analysis_result,
    serialize_problematic_samples,
    train_and_test_by_horizon,
)
from src.choice.consistency import (
    analyze_consistency as _analyze_consistency,
    filter_consistent_samples as _filter_consistent_samples,
)
from src.plotting import export_camera_ready

# Import from scripts/common (still needed for some utilities)
from common import (
    ConsistencyAnalysis,
    ThetaConstraint,
    create_alignment_plot,
    create_conflict_distribution_plot,
    generate_alignment_analysis,
    generate_conflicts_analysis,
    load_preference_data,
    print_consistency_analysis,
    save_analysis_output,
)
from common.consistency import (
    analyze_conflicts_deeply,
    format_horizon_label,
    generate_deep_conflict_analysis,
    horizon_sort_key,
)


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

PREFERENCE_DATA_DIR = PROJECT_ROOT / "out" / "preference_data"
PAPER_DIR = PROJECT_ROOT / "paper"


# Wrapper for analyze_consistency to inject dependencies
def analyze_consistency(
    samples: list[SampleWithHorizon],
) -> dict[str, ConsistencyAnalysis]:
    """Analyze consistency of choices for each horizon bucket."""
    return _analyze_consistency(
        samples,
        consistency_analysis_class=ConsistencyAnalysis,
        theta_constraint_from_sample=ThetaConstraint.from_sample,
    )


# Wrapper for load_samples to inject load_preference_data
def load_samples_from_query_ids(query_ids: list[str]) -> LoadedSamples:
    """Load samples from query IDs."""
    from src.choice.data import load_samples_from_query_ids as _load

    return _load(query_ids, load_preference_data, PREFERENCE_DATA_DIR)


# Wrapper for extract_samples_with_horizons to inject load_preference_data
def extract_samples_with_horizons(data_path: Path):
    """Extract samples with horizons from preference data."""
    from src.choice.data import extract_samples_with_horizons as _extract

    return _extract(data_path, load_preference_data)


# Wrapper for filter_consistent_samples to inject dependencies
def filter_consistent_samples(
    samples: list[SampleWithHorizon],
    consistency: dict[str, ConsistencyAnalysis],
) -> list[SampleWithHorizon]:
    """Filter out samples that conflict with the best theta for their horizon."""
    return _filter_consistent_samples(samples, consistency, ThetaConstraint.from_sample)


def _create_choice_model_plots(
    model_results: dict[str, list[HorizonModelResult]],
    output_dir: Path,
    model_results_no_conflicts: Optional[dict[str, list[HorizonModelResult]]] = None,
) -> None:
    """
    Create choice model visualization plots.

    Main plot (per model spec) now focuses on:
    - Fitted θ by horizon
    - Train vs Test accuracy by horizon

    Bias diagnostics have been moved to a separate plot (see
    _create_bias_diagnostics_plots).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    for discount_type, horizon_results in model_results.items():
        if not horizon_results:
            continue

        horizon_results_nc = None
        if model_results_no_conflicts and discount_type in model_results_no_conflicts:
            horizon_results_nc = model_results_no_conflicts[discount_type]

        # ---- Collect data (all data) ----
        horizons = []
        thetas = []
        train_accs = []
        test_accs = []

        for hr in horizon_results:
            horizons.append(format_horizon_label(hr.horizon))
            thetas.append(hr.trained_params.discount.theta)
            train_accs.append(hr.train_accuracy * 100)

            tr = hr.test_results.get(hr.horizon)
            test_accs.append(tr.metrics.accuracy * 100 if tr else None)

        # ---- Collect data (no-conflicts) aligned to same horizon order ----
        thetas_nc, train_accs_nc, test_accs_nc = None, None, None
        if horizon_results_nc:
            by_h = {hr.horizon: hr for hr in horizon_results_nc}
            thetas_nc, train_accs_nc, test_accs_nc = [], [], []
            for hr_all in horizon_results:
                hr = by_h.get(hr_all.horizon)
                if hr is None:
                    thetas_nc.append(np.nan)
                    train_accs_nc.append(np.nan)
                    test_accs_nc.append(None)
                    continue
                thetas_nc.append(hr.trained_params.discount.theta)
                train_accs_nc.append(hr.train_accuracy * 100)
                tr = hr.test_results.get(hr.horizon)
                test_accs_nc.append(tr.metrics.accuracy * 100 if tr else None)

        # ---- Figure layout ----
        n_rows = 2 if horizon_results_nc else 1
        fig_height = 4.8 * n_rows + 1.8
        fig, axes = plt.subplots(n_rows, 2, figsize=(14, fig_height))

        if n_rows == 1:
            axes = [axes]  # type: ignore[list-item]

        # ---- Title and formulas ----
        parts = discount_type.split("_")
        utility_type = parts[0]
        discount_func = "_".join(parts[1:]) if len(parts) > 1 else parts[0]

        def to_title_case(s: str) -> str:
            return (
                s.replace("_", " ")
                .replace("(", " (")
                .title()
                .replace(" ", "-")
                .replace("--", " ")
                .replace("-(", " (")
                .strip("-")
            )

        natural_title = to_title_case(discount_type)

        if utility_type == "linear":
            u_formula = "U(r) = r"
        elif utility_type == "log":
            u_formula = "U(r) = log(r)"
        elif utility_type.startswith("power"):
            u_formula = "U(r) = r^α"
        else:
            u_formula = "U(r)"

        if discount_func == "exponential":
            d_formula = "D(t) = exp(-θt)"
        elif discount_func == "hyperbolic":
            d_formula = "D(t) = 1/(1+θt)"
        elif discount_func == "quasi_hyperbolic":
            d_formula = "D(t) = βδ^t"
        else:
            d_formula = "D(t)"

        # Keep titles inside the figure to avoid large top whitespace with bbox_inches="tight"
        fig.suptitle(
            f"Choice Model: {natural_title}",
            fontsize=14,
            fontweight="bold",
            y=0.99,
            fontfamily="DejaVu Sans",
        )
        fig.text(
            0.5,
            0.955,
            f"{u_formula},  {d_formula}",
            ha="center",
            fontsize=11,
            family="monospace",
            color="#333333",
        )

        # Column titles (once)
        col_titles = ["Fitted θ", "Train/Test Accuracy"]
        col_x = [0.25, 0.75]
        for x_pos, col_title in zip(col_x, col_titles):
            fig.text(
                x_pos, 0.915, col_title, ha="center", fontsize=11, fontweight="bold"
            )

        def _label_theta(theta: float) -> str:
            if theta >= 1000:
                return f"{theta:.0f}"
            if theta >= 10:
                return f"{theta:.1f}"
            return f"{theta:.2f}"

        def plot_row(row_axes, row_thetas, row_train_accs, row_test_accs):
            x = np.arange(len(horizons))
            width = 0.35

            # --- Plot 1: θ values ---
            ax1 = row_axes[0]
            bars = ax1.bar(x, row_thetas, color="steelblue", alpha=0.85)
            ax1.set_ylabel("θ (discount rate)", fontsize=10)
            ax1.set_xticks(x)
            ax1.set_xticklabels(horizons, rotation=45, ha="right")
            ax1.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)

            for bar, theta in zip(bars, row_thetas):
                if not np.isfinite(theta):
                    continue
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 0.5,
                    _label_theta(theta),
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color="white",
                    rotation=45,
                    family="monospace",
                )

            # --- Plot 2: Train vs Test accuracy ---
            ax2 = row_axes[1]
            valid_test = [a if a is not None else 0 for a in row_test_accs]

            bars1 = ax2.bar(
                x - width / 2,
                row_train_accs,
                width,
                label="Train",
                color="steelblue",
                alpha=0.85,
            )
            bars2 = ax2.bar(
                x + width / 2,
                valid_test,
                width,
                label="Test",
                color="forestgreen",
                alpha=0.85,
            )

            ax2.set_ylabel("Accuracy (%)", fontsize=11)
            ax2.set_xticks(x)
            ax2.set_xticklabels(horizons, rotation=45, ha="right")
            ax2.set_ylim(0, 110)
            ax2.axhline(
                y=50,
                color="red",
                linestyle="--",
                linewidth=1,
                alpha=0.5,
                label="Chance",
            )
            ax2.legend(loc="lower left", fontsize=8, framealpha=0.9)

            for bar, acc in zip(bars1, row_train_accs):
                if np.isfinite(acc) and acc > 0:
                    ax2.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1,
                        f"{acc:.0f}%",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
            for bar, acc in zip(bars2, valid_test):
                if acc > 0:
                    ax2.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1,
                        f"{acc:.0f}%",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        # Row 1: all data
        plot_row(axes[0], thetas, train_accs, test_accs)

        # Row 2: no-conflicts
        if horizon_results_nc and thetas_nc is not None:
            plot_row(axes[1], thetas_nc, train_accs_nc, test_accs_nc)

        # Row labels (left side), computed from axes positions for robustness
        if n_rows == 2:
            row_labels = [("ALL\nDATA", "#3498db", 11), ("NO\nCONFLICTS", "#27ae60", 9)]
            for i, (label, color, fs) in enumerate(row_labels):
                pos = axes[i][0].get_position()
                y_center = (pos.y0 + pos.y1) / 2
                fig.text(
                    0.02,
                    y_center,
                    label,
                    ha="center",
                    va="center",
                    fontsize=fs,
                    fontweight="bold",
                    color="white",
                    family="sans-serif",
                    bbox=dict(
                        boxstyle="round,pad=0.4",
                        facecolor=color,
                        edgecolor="none",
                        alpha=0.9,
                    ),
                )

        fig.text(0.5, 0.03, "Time Horizon", ha="center", fontsize=11, fontweight="bold")

        # Leave space for title/formula/column titles
        plt.tight_layout(rect=[0.06, 0.06, 1, 0.89])

        models_dir = output_dir / "choice_models"
        ensure_dir(models_dir)
        plot_path = models_dir / f"{discount_type}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"  Saved: {plot_path.relative_to(output_dir)}")


def _create_bias_diagnostics_plots(
    model_results: dict[str, list[HorizonModelResult]],
    output_dir: Path,
    model_results_no_conflicts: Optional[dict[str, list[HorizonModelResult]]] = None,
) -> None:
    """
    Create bias + class-balance diagnostic plots (per model spec).

    This complements the main choice-model plot by:
    1) Showing the Bias Score explicitly: (long_acc - short_acc)
    2) Visualizing short-vs-long accuracy tradeoffs (scatter + dumbbell)
    3) Checking calibration on "long-term rate": predicted long share vs true long share
    """
    import matplotlib.pyplot as plt
    import numpy as np

    def _extract_row(
        horizon_results: list[HorizonModelResult], horizon_order: list[str]
    ):
        """Return aligned arrays for diagnostics in a fixed horizon order."""
        by_h = {hr.horizon: hr for hr in horizon_results}

        short_accs = []
        long_accs = []
        bias_scores = []
        n_test = []
        n_short = []
        n_long = []
        pred_long_rate = []
        true_long_rate = []

        for h in horizon_order:
            hr = by_h.get(h)
            tr = hr.test_results.get(h) if hr else None
            if not tr:
                short_accs.append(np.nan)
                long_accs.append(np.nan)
                bias_scores.append(np.nan)
                n_test.append(0)
                n_short.append(0)
                n_long.append(0)
                pred_long_rate.append(np.nan)
                true_long_rate.append(np.nan)
                continue

            m = tr.metrics
            ns = int(m.n_short or 0)
            nl = int(m.n_long or 0)
            nt = int(m.num_samples or 0)

            s_acc = (m.short_term_accuracy * 100) if ns > 0 else np.nan
            l_acc = (m.long_term_accuracy * 100) if nl > 0 else np.nan
            b = (
                (l_acc - s_acc)
                if (np.isfinite(l_acc) and np.isfinite(s_acc))
                else np.nan
            )

            # Predicted vs true long share
            preds = tr.predictions or []
            if nt > 0 and preds:
                pred_long = sum(1 for p, _a in preds if p == "long_term")
                pred_rate = pred_long / nt
            else:
                pred_rate = np.nan
            true_rate = (nl / nt) if nt > 0 else np.nan

            short_accs.append(s_acc)
            long_accs.append(l_acc)
            bias_scores.append(b)
            n_test.append(nt)
            n_short.append(ns)
            n_long.append(nl)
            pred_long_rate.append(pred_rate)
            true_long_rate.append(true_rate)

        return {
            "short_accs": np.array(short_accs, dtype=float),
            "long_accs": np.array(long_accs, dtype=float),
            "bias_scores": np.array(bias_scores, dtype=float),
            "n_test": np.array(n_test, dtype=int),
            "n_short": np.array(n_short, dtype=int),
            "n_long": np.array(n_long, dtype=int),
            "pred_long_rate": np.array(pred_long_rate, dtype=float),
            "true_long_rate": np.array(true_long_rate, dtype=float),
        }

    for discount_type, horizon_results in model_results.items():
        if not horizon_results:
            continue

        horizon_results_nc = None
        if model_results_no_conflicts and discount_type in model_results_no_conflicts:
            horizon_results_nc = model_results_no_conflicts[discount_type]

        # Use the same horizon order as the main plot for this model spec
        horizon_order = [hr.horizon for hr in horizon_results]
        horizon_labels = [format_horizon_label(h) for h in horizon_order]

        row_all = _extract_row(horizon_results, horizon_order)
        row_nc = (
            _extract_row(horizon_results_nc, horizon_order)
            if horizon_results_nc
            else None
        )

        n_rows = 2 if row_nc else 1
        fig_h = 5.2 * n_rows + 1.8
        fig, axes = plt.subplots(n_rows, 4, figsize=(22, fig_h))

        if n_rows == 1:
            axes = [axes]  # type: ignore[list-item]

        # Title
        natural_title = (
            discount_type.replace("_", " ")
            .replace("(", " (")
            .title()
            .replace(" ", "-")
            .replace("--", " ")
            .replace("-(", " (")
            .strip("-")
        )

        fig.suptitle(
            f"Bias & Class-Balance Diagnostics: {natural_title}",
            fontsize=14,
            fontweight="bold",
            y=0.99,
            fontfamily="DejaVu Sans",
        )

        col_titles = [
            "Bias Score (Long Acc − Short Acc)",
            "Short vs Long Accuracy",
            "Accuracy Dumbbell (Short vs Long)",
            "Predicted vs True Long Rate",
        ]
        col_x = [0.13, 0.38, 0.63, 0.88]
        for x_pos, title in zip(col_x, col_titles):
            fig.text(x_pos, 0.93, title, ha="center", fontsize=10, fontweight="bold")

        def _plot_one_row(axrow, data, row_name: str):
            x = np.arange(len(horizon_labels))

            bias = data["bias_scores"]
            s_acc = data["short_accs"]
            l_acc = data["long_accs"]
            nt = data["n_test"]
            ns = data["n_short"]
            nl = data["n_long"]
            pred_rate = data["pred_long_rate"]
            true_rate = data["true_long_rate"]

            # ---- Panel 1: Bias diverging bars ----
            ax1 = axrow[0]
            # Use 0 for missing bars but visually mark them
            plot_bias = np.where(np.isfinite(bias), bias, 0.0)
            bars = ax1.bar(x, plot_bias, alpha=0.85)
            ax1.axhline(0, color="black", linewidth=1)
            ax1.set_ylabel("Bias (pp)", fontsize=10)
            ax1.set_xticks(x)
            ax1.set_xticklabels(horizon_labels, rotation=45, ha="right")
            ax1.set_ylim(-110, 110)

            for i, bar in enumerate(bars):
                if not np.isfinite(bias[i]):
                    bar.set_alpha(0.25)
                    # annotate missing
                    ax1.text(
                        i,
                        0,
                        "n/a",
                        ha="center",
                        va="center",
                        fontsize=8,
                        fontweight="bold",
                    )
                else:
                    val = bias[i]
                    if abs(val) >= 5:
                        ax1.text(
                            i,
                            val + (5 if val >= 0 else -5),
                            f"{val:+.0f}",
                            ha="center",
                            va=("bottom" if val >= 0 else "top"),
                            fontsize=8,
                            fontweight="bold",
                        )

                # sample-size mini-annotation
                if nt[i] > 0:
                    ax1.text(
                        i,
                        -102,
                        f"n={nt[i]}\nS={ns[i]} L={nl[i]}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        family="monospace",
                    )

            # ---- Panel 2: Scatter short vs long accuracy ----
            ax2 = axrow[1]
            # Diagonal
            ax2.plot([0, 100], [0, 100], linestyle="--", linewidth=1, alpha=0.6)

            sizes = 30 + 4 * np.clip(nt, 0, 50)  # cap for readability
            for i, (sa, la) in enumerate(zip(s_acc, l_acc)):
                if np.isfinite(sa) and np.isfinite(la):
                    ax2.scatter(sa, la, s=float(sizes[i]), alpha=0.85)
                    ax2.text(sa + 1, la + 1, horizon_labels[i], fontsize=8)
                else:
                    # If one class missing, pin to axis edge with annotation
                    if np.isfinite(sa) and not np.isfinite(la):
                        ax2.scatter(sa, 0, s=float(sizes[i]), alpha=0.35, marker="x")
                        ax2.text(sa + 1, 2, f"{horizon_labels[i]} (no L)", fontsize=8)
                    elif not np.isfinite(sa) and np.isfinite(la):
                        ax2.scatter(0, la, s=float(sizes[i]), alpha=0.35, marker="x")
                        ax2.text(2, la + 1, f"{horizon_labels[i]} (no S)", fontsize=8)

            ax2.set_xlim(0, 100)
            ax2.set_ylim(0, 100)
            ax2.set_xlabel("Short-term accuracy (%)", fontsize=10)
            ax2.set_ylabel("Long-term accuracy (%)", fontsize=10)

            # ---- Panel 3: Dumbbell plot per horizon ----
            ax3 = axrow[2]
            y = np.arange(len(horizon_labels))
            ax3.axvline(50, linestyle="--", linewidth=1, alpha=0.4)
            for i in range(len(horizon_labels)):
                sa, la = s_acc[i], l_acc[i]
                if np.isfinite(sa) and np.isfinite(la):
                    ax3.plot([sa, la], [i, i], linewidth=2, alpha=0.8)
                    ax3.scatter([sa, la], [i, i], s=45, alpha=0.9)
                elif np.isfinite(sa):
                    ax3.scatter([sa], [i], s=45, alpha=0.6)
                    ax3.text(sa + 1, i, "short only", fontsize=8, va="center")
                elif np.isfinite(la):
                    ax3.scatter([la], [i], s=45, alpha=0.6)
                    ax3.text(la + 1, i, "long only", fontsize=8, va="center")

            ax3.set_yticks(y)
            ax3.set_yticklabels(horizon_labels, fontsize=9)
            ax3.invert_yaxis()
            ax3.set_xlim(0, 100)
            ax3.set_xlabel("Accuracy (%)", fontsize=10)

            # ---- Panel 4: Calibration on long rate ----
            ax4 = axrow[3]
            ax4.plot([0, 1], [0, 1], linestyle="--", linewidth=1, alpha=0.6)
            for i, (t, p) in enumerate(zip(true_rate, pred_rate)):
                if np.isfinite(t) and np.isfinite(p):
                    ax4.scatter(t, p, s=float(sizes[i]), alpha=0.85)
                    ax4.text(t + 0.02, p + 0.02, horizon_labels[i], fontsize=8)

            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.set_xlabel("True long share", fontsize=10)
            ax4.set_ylabel("Predicted long share", fontsize=10)

            # Row name in left margin (inside figure coordinates)
            pos = axrow[0].get_position()
            y_center = (pos.y0 + pos.y1) / 2
            fig.text(
                0.02,
                y_center,
                row_name,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="white",
                family="sans-serif",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor=("#3498db" if row_name.startswith("ALL") else "#27ae60"),
                    edgecolor="none",
                    alpha=0.9,
                ),
            )

        _plot_one_row(axes[0], row_all, "ALL\nDATA")

        if row_nc:
            _plot_one_row(axes[1], row_nc, "NO\nCONFLICTS")

        plt.tight_layout(rect=[0.05, 0.06, 1, 0.90])

        diag_dir = output_dir / "choice_models_diagnostics"
        ensure_dir(diag_dir)
        plot_path = diag_dir / f"{discount_type}_bias_diagnostics.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"  Saved: {plot_path.relative_to(output_dir)}")


def _create_cluster_analysis_plots(
    analyses: dict[str, ConsistencyAnalysis],
    clusters_dir: Path,
    base_output_dir: Path,
) -> None:
    """
    Create cluster analysis visualizations for CONFLICTING samples only.

    Analyzes whether conflicts form coherent clusters among themselves,
    suggesting that conflicting choices might follow a different but
    internally consistent discount rate.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    from scripts.common.consistency import (
        analyze_conflicts_deeply,
        find_consistent_clusters,
    )

    for horizon_key in sorted(analyses.keys(), key=horizon_sort_key):
        analysis = analyses[horizon_key]
        if not analysis.constraints:
            continue

        # Get deep conflict analysis
        deep_analysis = analyze_conflicts_deeply(analysis.constraints)

        # Skip if no conflicts or only one cluster within conflicts
        if deep_analysis.n_conflicting == 0:
            continue
        if deep_analysis.conflict_n_clusters <= 1:
            continue

        # Extract just the conflicting samples
        best_theta = analysis.best_theta
        conflicting_samples = [
            c for c in analysis.constraints if not c.is_satisfied(best_theta)
        ]

        # Find clusters within conflicts only
        conflict_clusters = find_consistent_clusters(conflicting_samples)

        # Create figure with 3 subplots for richer analysis
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        horizon_label = format_horizon_label(horizon_key)
        n_clusters = len(conflict_clusters)

        # Compute cluster statistics
        cluster_stats = []
        for i, cluster in enumerate(conflict_clusters):
            if cluster:
                cluster_analysis = ConsistencyAnalysis.analyze(cluster, horizon_key)
                n_short = sum(1 for c in cluster if c.choice == "short_term")
                n_long = len(cluster) - n_short
                cluster_stats.append(
                    {
                        "size": len(cluster),
                        "theta": cluster_analysis.best_theta,
                        "n_short": n_short,
                        "n_long": n_long,
                        "pct_short": n_short / len(cluster) * 100 if cluster else 0,
                    }
                )
            else:
                cluster_stats.append(
                    {"size": 0, "theta": 0, "n_short": 0, "n_long": 0, "pct_short": 0}
                )

        # Determine main insight
        if n_clusters == 2:
            theta_diff = abs(cluster_stats[0]["theta"] - cluster_stats[1]["theta"])
            if theta_diff > 0.1:
                insight = (
                    f"Two distinct preference groups: θ differs by {theta_diff:.2f}"
                )
            else:
                insight = "Clusters have similar θ - conflicts may be noise"
        else:
            insight = f"{n_clusters} clusters found within conflicts"

        fig.suptitle(
            f"Conflict Deep-Dive: {horizon_label}",
            fontsize=14,
            fontweight="bold",
            fontfamily="DejaVu Sans",
        )

        cluster_labels = [f"Cluster {i + 1}" for i in range(n_clusters)]
        colors = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3"][:n_clusters]

        # Plot 1: Cluster composition (stacked bar showing short vs long choices)
        ax1 = axes[0]
        x = np.arange(n_clusters)
        width = 0.6

        shorts = [s["n_short"] for s in cluster_stats]
        longs = [s["n_long"] for s in cluster_stats]

        bars1 = ax1.bar(
            x, shorts, width, label="Chose short-term", color="coral", alpha=0.8
        )
        bars2 = ax1.bar(
            x,
            longs,
            width,
            bottom=shorts,
            label="Chose long-term",
            color="steelblue",
            alpha=0.8,
        )

        ax1.set_xlabel("Conflict Cluster", fontsize=11)
        ax1.set_ylabel("Number of Samples", fontsize=11)
        ax1.set_title("What Did Each Cluster Choose?", fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(cluster_labels)
        ax1.legend(loc="upper right", fontsize=8)

        # Add total labels - set y-limit to leave room above bars
        max_total = max(s + l for s, l in zip(shorts, longs))
        ax1.set_ylim(0, max_total * 1.2)  # 20% headroom for labels

        for i, (s, l) in enumerate(zip(shorts, longs)):
            total = s + l
            ax1.text(
                i,
                total + max_total * 0.02,
                f"n={total}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        # Plot 2: Theta comparison with clear interpretation
        ax2 = axes[1]
        cluster_thetas = [s["theta"] for s in cluster_stats]

        bars3 = ax2.bar(x, cluster_thetas, width, color=colors, alpha=0.8)
        ax2.axhline(
            y=best_theta,
            color="red",
            linestyle="--",
            linewidth=2.5,
            label=f"Majority θ={best_theta:.3f}",
        )

        ax2.set_xlabel("Conflict Cluster", fontsize=11)
        ax2.set_ylabel("Fitted θ (discount rate)", fontsize=11)
        ax2.set_title(
            "Cluster Impatience Level\n(Higher θ = prefers short-term)", fontsize=12
        )
        ax2.set_xticks(x)
        ax2.set_xticklabels(cluster_labels)
        ax2.legend(loc="upper right", fontsize=8)

        # Add theta labels with interpretation - inside bars to avoid collision
        max_theta = max(cluster_thetas) if cluster_thetas else 1
        for i, theta in enumerate(cluster_thetas):
            if theta > best_theta * 2:
                interp = "impatient"
                text_color = "coral"
            elif theta < best_theta * 0.5:
                interp = "patient"
                text_color = "steelblue"
            else:
                interp = ""
                text_color = "gray"

            # Place inside bar if tall enough, otherwise above
            bar_height = bars3[i].get_height()
            if bar_height > max_theta * 0.3:
                label = f"θ={theta:.2f}" + (f"\n{interp}" if interp else "")
                ax2.text(
                    i,
                    bar_height * 0.5,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                    family="monospace",
                )

        # Plot 3: Why are these conflicts? Show they disagree with majority
        ax3 = axes[2]

        # Count choices in consistent (non-conflict) samples
        consistent_samples = [
            c for c in analysis.constraints if c.is_satisfied(best_theta)
        ]
        consist_short = sum(1 for c in consistent_samples if c.choice == "short_term")
        consist_long = len(consistent_samples) - consist_short

        conflict_short = sum(s["n_short"] for s in cluster_stats)
        conflict_long = sum(s["n_long"] for s in cluster_stats)

        categories = ["Majority\n(aligned)", "Conflicts\n(clusters)"]
        x3 = np.arange(len(categories))

        maj_shorts = [consist_short, conflict_short]
        maj_longs = [consist_long, conflict_long]

        bars4 = ax3.bar(
            x3, maj_shorts, width, label="Chose short", color="coral", alpha=0.8
        )
        bars5 = ax3.bar(
            x3,
            maj_longs,
            width,
            bottom=maj_shorts,
            label="Chose long",
            color="steelblue",
            alpha=0.8,
        )

        ax3.set_xlabel("Sample Group", fontsize=11)
        ax3.set_ylabel("Number of Samples", fontsize=11)
        ax3.set_title("Choice Distribution", fontsize=12)
        ax3.set_xticks(x3)
        ax3.set_xticklabels(categories)
        ax3.legend(loc="upper right", fontsize=8)

        # Add percentage annotations
        for i, (s, l) in enumerate(zip(maj_shorts, maj_longs)):
            total = s + l
            if total > 0:
                pct_short = s / total * 100
                ax3.text(
                    i,
                    total + 1,
                    f"{pct_short:.0f}% short",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        plt.tight_layout(rect=[0, 0, 1, 0.90])

        # Save plot
        safe_horizon = horizon_key.replace(" ", "_").replace("/", "_")
        plot_path = clusters_dir / f"conflict_clusters_{safe_horizon}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"  Saved: {plot_path.relative_to(base_output_dir)}")

    # Create summary plot across all horizons
    _create_cluster_summary_plot(analyses, clusters_dir, base_output_dir)


def _create_cluster_summary_plot(
    analyses: dict[str, ConsistencyAnalysis],
    clusters_dir: Path,
    base_output_dir: Path,
) -> None:
    """Create summary plot showing conflict cluster analysis across all horizons."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    from scripts.common.consistency import analyze_conflicts_deeply

    # Collect data for all horizons
    horizon_data = []
    for horizon_key in sorted(analyses.keys(), key=horizon_sort_key):
        analysis = analyses[horizon_key]
        if not analysis.constraints:
            continue

        deep_analysis = analyze_conflicts_deeply(analysis.constraints)

        horizon_data.append(
            {
                "label": format_horizon_label(horizon_key),
                "conflict_n_clusters": deep_analysis.conflict_n_clusters,
                "n_consistent": deep_analysis.n_consistent,
                "n_conflicting": deep_analysis.n_conflicting,
                "total": deep_analysis.total_samples,
                "conflicts_internally_consistent": deep_analysis.conflicts_internally_consistent,
            }
        )

    if not horizon_data:
        return

    # Compute key insight
    total_conflicts = sum(d["n_conflicting"] for d in horizon_data)
    horizons_with_multi_cluster = sum(
        1 for d in horizon_data if d["conflict_n_clusters"] > 1
    )
    if horizons_with_multi_cluster == 0:
        main_insight = "All conflicts are internally consistent (single cluster each)"
    else:
        main_insight = (
            f"{horizons_with_multi_cluster} horizons have multiple conflict clusters"
        )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Conflict Cluster Summary",
        fontsize=14,
        fontweight="bold",
        fontfamily="DejaVu Sans",
    )

    x = np.arange(len(horizon_data))
    labels = [d["label"] for d in horizon_data]

    # Plot 1: Number of clusters WITHIN conflicts per horizon
    ax1 = axes[0]
    conflict_n_clusters = [d["conflict_n_clusters"] for d in horizon_data]
    n_conflicting_list = [d["n_conflicting"] for d in horizon_data]

    # Color: green if 0 or 1 cluster (conflicts are consistent), orange if 2, red if more
    colors = [
        "forestgreen" if n <= 1 else "orange" if n == 2 else "crimson"
        for n in conflict_n_clusters
    ]
    bars = ax1.bar(x, conflict_n_clusters, color=colors, alpha=0.8)

    ax1.set_xlabel("Time Horizon", fontsize=11)
    ax1.set_ylabel("Number of Clusters", fontsize=11)
    ax1.set_title("Preference Groups Within Conflicts", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.axhline(
        y=1,
        color="green",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Single group",
    )
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_ylim(0, max(conflict_n_clusters) + 1.5)  # More headroom for labels

    for bar, n, n_conf in zip(bars, conflict_n_clusters, n_conflicting_list):
        if n_conf > 0:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.15,
                f"n={n_conf}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # Plot 2: Conflict rate showing how much of data is conflicting
    ax2 = axes[1]
    n_consistent = [d["n_consistent"] for d in horizon_data]
    n_conflicting = [d["n_conflicting"] for d in horizon_data]
    conflict_rates = [
        c / (c + n) * 100 if (c + n) > 0 else 0
        for c, n in zip(n_conflicting, n_consistent)
    ]

    bars2 = ax2.bar(
        x,
        conflict_rates,
        color=[
            "forestgreen" if r < 15 else "orange" if r < 30 else "crimson"
            for r in conflict_rates
        ],
        alpha=0.8,
    )

    ax2.set_xlabel("Time Horizon", fontsize=11)
    ax2.set_ylabel("Conflict Rate (%)", fontsize=11)
    ax2.set_title("Conflict Rate by Horizon", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right")
    ax2.set_ylim(0, max(conflict_rates) * 1.3)  # Headroom for labels
    ax2.axhline(
        y=10,
        color="forestgreen",
        linestyle=":",
        linewidth=1.5,
        alpha=0.7,
        label="Low (10%)",
    )
    ax2.axhline(
        y=25,
        color="darkorange",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Moderate (25%)",
    )
    ax2.axhline(
        y=40,
        color="crimson",
        linestyle="-.",
        linewidth=1.5,
        alpha=0.7,
        label="High (40%)",
    )
    ax2.legend(loc="upper left", fontsize=8)

    for bar, rate, n_conf in zip(bars2, conflict_rates, n_conflicting):
        # Place labels inside bars if tall enough
        if bar.get_height() > 15:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() - 3,
                f"{rate:.0f}%\nn={n_conf}",
                ha="center",
                va="top",
                fontsize=8,
                color="white",
                fontweight="bold",
            )
        else:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{rate:.0f}%\nn={n_conf}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout(rect=[0, 0, 1, 0.90])

    plot_path = clusters_dir / "cluster_summary.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"  Saved: {plot_path.relative_to(base_output_dir)}")


def _create_no_horizon_deep_dive(
    analyses: dict[str, ConsistencyAnalysis],
    conflicts_dir: Path,
    base_output_dir: Path,
    model_results: Optional[dict[str, list[HorizonModelResult]]] = None,
) -> None:
    """
    Create deep-dive analysis specifically for No Horizon samples.

    No Horizon samples are particularly interesting because without a time
    constraint, the LLM's "natural" discount preference is revealed.
    This analysis shows:
    1. Conflict breakdown with model accuracy for each cluster
    2. Sample-level details showing why specific samples conflict
    3. Comparison of clusters to see if they represent different discount rates
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    from scripts.common.consistency import (
        analyze_conflicts_deeply,
        find_consistent_clusters,
    )

    # Get no_horizon analysis if available
    if "no_horizon" not in analyses:
        return

    analysis = analyses["no_horizon"]
    if not analysis.constraints:
        return

    deep_analysis = analyze_conflicts_deeply(analysis.constraints)
    best_theta = analysis.best_theta

    # Extract conflicting samples
    conflicting_samples = [
        c for c in analysis.constraints if not c.is_satisfied(best_theta)
    ]
    consistent_samples = [c for c in analysis.constraints if c.is_satisfied(best_theta)]

    if not conflicting_samples:
        return  # No conflicts to analyze

    # Find clusters within conflicts
    conflict_clusters = find_consistent_clusters(conflicting_samples)

    # =========================================================================
    # Figure 1: No Horizon Overview (3 panels)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("No Horizon Conflict Deep-Dive", fontsize=14, fontweight="bold")

    # Panel 1: Choice distribution - conflicts vs consistent
    ax1 = axes[0]
    consist_short = sum(1 for c in consistent_samples if c.choice == "short_term")
    consist_long = len(consistent_samples) - consist_short
    conflict_short = sum(1 for c in conflicting_samples if c.choice == "short_term")
    conflict_long = len(conflicting_samples) - conflict_short

    x = np.arange(2)
    width = 0.6
    shorts = [consist_short, conflict_short]
    longs = [consist_long, conflict_long]

    ax1.bar(x, shorts, width, label="Short-term", color="coral", alpha=0.8)
    ax1.bar(
        x, longs, width, bottom=shorts, label="Long-term", color="steelblue", alpha=0.8
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Consistent\n(fit by θ)", "Conflicts\n(don't fit)"])
    ax1.set_ylabel("Number of Samples")
    ax1.set_title("What Did Each Cluster Choose?")
    ax1.legend(loc="upper right")

    # Add headroom for labels and annotate with percentages
    max_total = max(s + l for s, l in zip(shorts, longs))
    ax1.set_ylim(0, max_total * 1.25)

    for i, (s, l) in enumerate(zip(shorts, longs)):
        total = s + l
        if total > 0:
            ax1.text(
                i,
                total + 2,
                f"{s / total * 100:.0f}% short\nn={total}",
                ha="center",
                fontsize=9,
            )

    # Panel 2: Theta requirements - why they conflict
    ax2 = axes[1]
    consistent_thresholds = [
        c.theta_threshold
        for c in consistent_samples
        if c.theta_threshold < float("inf")
    ]
    conflict_thresholds = [
        c.theta_threshold
        for c in conflicting_samples
        if c.theta_threshold < float("inf")
    ]

    if consistent_thresholds or conflict_thresholds:
        all_thresholds = consistent_thresholds + conflict_thresholds
        bins = np.linspace(min(all_thresholds), min(max(all_thresholds), 5), 30)

        if consistent_thresholds:
            ax2.hist(
                consistent_thresholds,
                bins=bins,
                alpha=0.6,
                label="Consistent",
                color="forestgreen",
            )
        if conflict_thresholds:
            ax2.hist(
                conflict_thresholds,
                bins=bins,
                alpha=0.6,
                label="Conflicts",
                color="crimson",
            )

        ax2.axvline(
            x=best_theta,
            color="black",
            linestyle="--",
            linewidth=2,
            label=f"Best θ={best_theta:.3f}",
        )
        ax2.set_xlabel("θ threshold")
        ax2.set_ylabel("Count")
        ax2.set_title("θ Requirements")
        ax2.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # Panel 3: Cluster analysis within conflicts
    ax3 = axes[2]
    n_clusters = len(conflict_clusters)
    if n_clusters > 0:
        cluster_sizes = [len(c) for c in conflict_clusters]
        cluster_thetas = []
        for cluster in conflict_clusters:
            if cluster:
                cluster_analysis = ConsistencyAnalysis.analyze(cluster, "no_horizon")
                cluster_thetas.append(cluster_analysis.best_theta)
            else:
                cluster_thetas.append(0)

        colors = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3"][:n_clusters]
        bars = ax3.bar(range(n_clusters), cluster_thetas, color=colors, alpha=0.8)
        ax3.axhline(
            y=best_theta,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Majority θ={best_theta:.3f}",
        )

        ax3.set_xlabel("Conflict Cluster")
        ax3.set_ylabel("Fitted θ")
        ax3.set_title(f"{n_clusters} Preference Groups in Conflicts")
        ax3.set_xticks(range(n_clusters))
        ax3.set_xticklabels(
            [f"Cluster {i + 1}\nn={s}" for i, s in enumerate(cluster_sizes)]
        )
        ax3.legend(loc="upper right")

        # Add headroom for theta labels
        max_theta = max(cluster_thetas) if cluster_thetas else 1
        ax3.set_ylim(0, max(max_theta * 1.2, best_theta * 1.3))

        # Label each bar with theta
        for i, (bar, theta) in enumerate(zip(bars, cluster_thetas)):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"θ={theta:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                family="monospace",
            )
    else:
        ax3.text(
            0.5,
            0.5,
            "No clusters found",
            ha="center",
            va="center",
            transform=ax3.transAxes,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plot_path = conflicts_dir / "no_horizon_overview.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {plot_path.relative_to(base_output_dir)}")

    # =========================================================================
    # Figure 2: Sample-Level Conflict Analysis
    # =========================================================================
    _create_sample_level_conflict_plot(
        conflicting_samples,
        consistent_samples,
        best_theta,
        conflicts_dir,
        base_output_dir,
    )


def _create_sample_level_conflict_plot(
    conflicting_samples: list,
    consistent_samples: list,
    best_theta: float,
    output_dir: Path,
    base_output_dir: Path,
    max_samples: int = 4,
) -> None:
    """
    Create clean table showing example conflicts.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    if not conflicting_samples:
        return

    # Select diverse examples
    long_conflicts = [c for c in conflicting_samples if c.choice == "long_term"]
    short_conflicts = [c for c in conflicting_samples if c.choice == "short_term"]
    long_sorted = sorted(
        long_conflicts, key=lambda c: abs(c.theta_threshold - best_theta), reverse=True
    )[: max_samples // 2]
    short_sorted = sorted(
        short_conflicts, key=lambda c: abs(c.theta_threshold - best_theta), reverse=True
    )[: max_samples // 2]
    selected = long_sorted + short_sorted

    if not selected:
        return

    # Create clean table figure
    fig, ax = plt.subplots(figsize=(12, 1.5 + len(selected) * 0.8))
    ax.axis("off")
    fig.suptitle(
        "Example Conflicts When No Horizon Specified", fontsize=14, fontweight="bold"
    )

    # Table headers
    headers = ["Short Option", "Long Option", "LLM Chose", "Why Conflict?"]
    col_widths = [0.20, 0.20, 0.12, 0.42]
    col_positions = [0.02]
    for w in col_widths[:-1]:
        col_positions.append(col_positions[-1] + w)

    # Draw headers
    y = 0.92
    for i, (header, x) in enumerate(zip(headers, col_positions)):
        ax.text(
            x,
            y,
            header,
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="center",
            ha="left",
        )

    # Draw separator line (use plot since axhline doesn't support transform)
    ax.plot(
        [0.02, 0.95],
        [0.88, 0.88],
        color="black",
        linewidth=1,
        transform=ax.transAxes,
        clip_on=False,
    )

    # Draw rows
    row_height = 0.8 / len(selected)
    for idx, sample in enumerate(selected):
        y = 0.85 - idx * row_height

        # Format values (no $ - rewards aren't always money)
        short_opt = f"{sample.reward_short:,.0f} @ {sample.time_short:.1f}yr"
        long_opt = f"{sample.reward_long:,.0f} @ {sample.time_long:.1f}yr"

        if sample.choice == "long_term":
            chose = "LONG"
            chose_color = "steelblue"
            why = "Most no-horizon samples chose short; this chose long"
        else:
            chose = "SHORT"
            chose_color = "coral"
            why = "Most no-horizon samples chose long; this chose short"

        # Draw cells
        ax.text(
            col_positions[0],
            y,
            short_opt,
            transform=ax.transAxes,
            fontsize=10,
            va="center",
            ha="left",
        )
        ax.text(
            col_positions[1],
            y,
            long_opt,
            transform=ax.transAxes,
            fontsize=10,
            va="center",
            ha="left",
        )
        ax.text(
            col_positions[2],
            y,
            chose,
            transform=ax.transAxes,
            fontsize=10,
            va="center",
            ha="left",
            fontweight="bold",
            color=chose_color,
        )
        ax.text(
            col_positions[3],
            y,
            why,
            transform=ax.transAxes,
            fontsize=9,
            va="center",
            ha="left",
            style="italic",
        )

    plt.tight_layout()
    plot_path = output_dir / "no_horizon_samples.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {plot_path.relative_to(base_output_dir)}")

    # =========================================================================
    # Figure 3: Scatter overview (simplified)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle("No Horizon: All Samples", fontsize=14, fontweight="bold")

    # Plot consistent samples first (background)
    for c in consistent_samples:
        ratio = c.reward_long / c.reward_short
        time_diff = c.time_long - c.time_short
        color = "steelblue" if c.choice == "long_term" else "coral"
        ax.scatter(time_diff, ratio, c=color, alpha=0.4, s=50)

    # Plot conflict samples with emphasis
    for c in conflicting_samples:
        ratio = c.reward_long / c.reward_short
        time_diff = c.time_long - c.time_short
        color = "steelblue" if c.choice == "long_term" else "coral"
        ax.scatter(
            time_diff, ratio, c=color, alpha=0.8, s=100, edgecolor="red", linewidth=2
        )

    ax.set_xlabel("Time Difference (years)", fontsize=11)
    ax.set_ylabel("Reward Ratio (long/short)", fontsize=11)
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)

    # Custom legend positioned to avoid data
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="coral",
            markersize=10,
            label="Chose short",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="steelblue",
            markersize=10,
            label="Chose long",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="red",
            markeredgewidth=2,
            markersize=10,
            label="Conflict",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper left", framealpha=0.9)

    plt.tight_layout()
    plot_path = output_dir / "no_horizon_scatter.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {plot_path.relative_to(base_output_dir)}")


def create_visualizations(result: AnalysisResult, output_dir: Path) -> None:
    """Create and save visualization plots."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Warning: matplotlib not installed. Skipping visualizations.")
        print("  Install with: uv add matplotlib")
        return

    ensure_dir(output_dir)

    # Generate choice model plots (with no-conflicts as bottom row if available)
    _create_choice_model_plots(
        result.model_results,
        output_dir,
        model_results_no_conflicts=result.model_results_no_conflicts,
    )

    # Bias + class-balance diagnostics (separate figure)
    _create_bias_diagnostics_plots(
        result.model_results,
        output_dir,
        model_results_no_conflicts=result.model_results_no_conflicts,
    )

    # Create data verification visualization if available
    verifications = []
    if result.train_verification and result.train_verification.category_stats:
        verifications.append(result.train_verification)
    if result.test_verification and result.test_verification.category_stats:
        verifications.append(result.test_verification)

    if verifications:
        plot_path = create_verification_plot(
            verifications, output_dir, result.timestamp
        )
        if plot_path:
            # Handle both Path and str returns
            if isinstance(plot_path, Path):
                print(f"  Saved: {plot_path.relative_to(output_dir)}")
            else:
                print(f"  Saved: {plot_path}")

    # Create consistency analysis plots if available
    if result.train_consistency:
        # Create conflict plots in conflicts/ subfolder
        conflicts_dir = output_dir / "conflicts"
        ensure_dir(conflicts_dir)

        plot_path = create_conflict_distribution_plot(
            result.train_consistency, conflicts_dir
        )
        if plot_path:
            if isinstance(plot_path, Path):
                print(f"  Saved: {plot_path.relative_to(output_dir)}")
            else:
                print(f"  Saved: conflicts/{Path(plot_path).name}")

        # Create cluster analysis plots in conflicts/clusters/
        clusters_dir = conflicts_dir / "clusters"
        ensure_dir(clusters_dir)
        _create_cluster_analysis_plots(
            result.train_consistency, clusters_dir, output_dir
        )

        # Create No Horizon deep-dive in conflicts/no_horizon/
        no_horizon_dir = conflicts_dir / "no_horizon"
        ensure_dir(no_horizon_dir)
        _create_no_horizon_deep_dive(
            result.train_consistency,
            no_horizon_dir,
            output_dir,
            model_results=result.model_results,
        )

        # Create horizon alignment plots in alignment/ subfolder
        alignment_dir = output_dir / "alignment"
        ensure_dir(alignment_dir)

        plot_path = create_alignment_plot(result.train_consistency, alignment_dir)
        if plot_path:
            if isinstance(plot_path, Path):
                print(f"  Saved: {plot_path.relative_to(output_dir)}")
            else:
                print(f"  Saved: alignment/{Path(plot_path).name}")


def create_model_comparison_heatmap(
    model_results: dict[str, list[HorizonModelResult]],
    output_dir: Path,
    model_results_no_conflicts: Optional[dict[str, list[HorizonModelResult]]] = None,
) -> Optional[Path]:
    """
    Create heatmap comparing model performance across horizons.

    Shows which models excel at short-term vs long-term predictions.
    If model_results_no_conflicts is provided, creates a SEPARATE heatmap file.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    # Collect data
    model_names = list(model_results.keys())
    if not model_names:
        return None

    # Get horizons from first model
    horizons = [hr.horizon for hr in model_results[model_names[0]]]
    horizon_labels = [format_horizon_label(h) for h in horizons]

    def build_accuracy_matrix(results):
        matrix = []
        for model_name in model_names:
            row = []
            for hr in results[model_name]:
                if hr.horizon in hr.test_results:
                    acc = hr.test_results[hr.horizon].metrics.accuracy * 100
                else:
                    acc = hr.train_accuracy * 100
                row.append(acc)
            matrix.append(row)
        return np.array(matrix)

    accuracy_matrix = build_accuracy_matrix(model_results)

    comparison_dir = output_dir / "model_comparison"
    ensure_dir(comparison_dir)

    # =========================================================================
    # Plot 1: Main heatmap (All Data)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))

    # Panel 1: Heatmap
    ax1 = axes[0]
    im = ax1.imshow(accuracy_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    ax1.set_xticks(np.arange(len(horizon_labels)))
    ax1.set_yticks(np.arange(len(model_names)))
    ax1.set_xticklabels(horizon_labels, rotation=45, ha="right", fontsize=9)
    ax1.set_yticklabels(model_names, fontsize=9)

    # Add text annotations - always use black for readability
    for i in range(len(model_names)):
        for j in range(len(horizon_labels)):
            val = accuracy_matrix[i, j]
            ax1.text(
                j,
                i,
                f"{val:.0f}%",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
                fontweight="bold",
            )

    ax1.set_xlabel("Time Horizon", fontsize=11)
    ax1.set_ylabel("Model", fontsize=11)
    ax1.set_title("Accuracy by Horizon", fontsize=12)

    cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
    cbar.set_label("Accuracy (%)", fontsize=10)

    # Panel 2: Best model per horizon
    ax2 = axes[1]

    best_models = []
    best_accs = []
    colors = []

    for j, horizon in enumerate(horizons):
        col_accs = accuracy_matrix[:, j]
        best_idx = np.argmax(col_accs)
        best_models.append(model_names[best_idx])
        best_accs.append(col_accs[best_idx])
        model = model_names[best_idx]
        if "quasi" in model:
            colors.append("purple")
        elif "hyperbolic" in model:
            colors.append("orange")
        else:
            colors.append("steelblue")

    x = np.arange(len(horizon_labels))
    bars = ax2.bar(
        x, best_accs, color=colors, alpha=0.8, edgecolor="white", linewidth=0.5
    )

    # Put model name inside bar, accuracy below it
    for bar, model, acc in zip(bars, best_models, best_accs):
        short_name = model.replace("linear_", "").replace("log_", "")
        short_name = short_name.replace("power(α=", "pow").replace(")", "")
        short_name = short_name.replace("exponential", "exp").replace(
            "hyperbolic", "hyp"
        )
        short_name = short_name.replace("quasi_", "q-")
        # Model name at top of bar
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() - 5,
            f"{short_name}",
            ha="center",
            va="top",
            fontsize=7,
            fontweight="bold",
            color="white",
        )
        # Accuracy in middle of bar
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f"{acc:.0f}%",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white",
        )

    ax2.set_xlabel("Time Horizon", fontsize=11)
    ax2.set_ylabel("Best Accuracy (%)", fontsize=11)
    ax2.yaxis.set_label_position("right")
    ax2.set_title("Best Model per Horizon", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(horizon_labels, rotation=45, ha="right", fontsize=9)
    ax2.set_ylim(0, 105)

    # Add legend for model type colors
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="steelblue", alpha=0.8, label="Exponential"),
        Patch(facecolor="orange", alpha=0.8, label="Hyperbolic"),
        Patch(facecolor="purple", alpha=0.8, label="Quasi-hyp"),
    ]
    ax2.legend(handles=legend_elements, loc="lower left", fontsize=7, framealpha=0.9)

    fig.suptitle("Model Comparison: All Data", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    plot_path = comparison_dir / "accuracy_heatmap.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    # =========================================================================
    # Plot 2: No-Conflicts heatmap (SEPARATE FILE)
    # =========================================================================
    if model_results_no_conflicts:
        accuracy_matrix_nc = build_accuracy_matrix(model_results_no_conflicts)

        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 8))

        # Panel 1: No-conflicts heatmap
        ax1 = axes2[0]
        im2 = ax1.imshow(
            accuracy_matrix_nc, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100
        )

        ax1.set_xticks(np.arange(len(horizon_labels)))
        ax1.set_yticks(np.arange(len(model_names)))
        ax1.set_xticklabels(horizon_labels, rotation=45, ha="right", fontsize=9)
        ax1.set_yticklabels(model_names, fontsize=9)

        for i in range(len(model_names)):
            for j in range(len(horizon_labels)):
                val = accuracy_matrix_nc[i, j]
                ax1.text(
                    j,
                    i,
                    f"{val:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                    fontweight="bold",
                )

        ax1.set_xlabel("Time Horizon", fontsize=11)
        ax1.set_ylabel("Model", fontsize=11)
        ax1.set_title("Accuracy by Horizon", fontsize=12)

        cbar2 = plt.colorbar(im2, ax=ax1, shrink=0.8)
        cbar2.set_label("Accuracy (%)", fontsize=10)

        # Panel 2: Best model per horizon (no conflicts)
        ax2 = axes2[1]

        best_models_nc = []
        best_accs_nc = []
        colors_nc = []

        for j, horizon in enumerate(horizons):
            col_accs = accuracy_matrix_nc[:, j]
            best_idx = np.argmax(col_accs)
            best_models_nc.append(model_names[best_idx])
            best_accs_nc.append(col_accs[best_idx])
            model = model_names[best_idx]
            if "quasi" in model:
                colors_nc.append("purple")
            elif "hyperbolic" in model:
                colors_nc.append("orange")
            else:
                colors_nc.append("steelblue")

        bars2 = ax2.bar(
            x,
            best_accs_nc,
            color=colors_nc,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.5,
        )

        # Put model name inside bar, accuracy below it
        for bar, model, acc in zip(bars2, best_models_nc, best_accs_nc):
            short_name = model.replace("linear_", "").replace("log_", "")
            short_name = short_name.replace("power(α=", "pow").replace(")", "")
            short_name = short_name.replace("exponential", "exp").replace(
                "hyperbolic", "hyp"
            )
            short_name = short_name.replace("quasi_", "q-")
            # Model name at top of bar
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() - 5,
                f"{short_name}",
                ha="center",
                va="top",
                fontsize=7,
                fontweight="bold",
                color="white",
            )
            # Accuracy in middle of bar
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                f"{acc:.0f}%",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="white",
            )

        ax2.set_xlabel("Time Horizon", fontsize=11)
        ax2.set_ylabel("Best Accuracy (%)", fontsize=11)
        ax2.yaxis.set_label_position("right")
        ax2.set_title("Best Model per Horizon", fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(horizon_labels, rotation=45, ha="right", fontsize=9)
        ax2.set_ylim(0, 105)

        # Add legend for model type colors
        from matplotlib.patches import Patch

        legend_elements_nc = [
            Patch(facecolor="steelblue", alpha=0.8, label="Exponential"),
            Patch(facecolor="orange", alpha=0.8, label="Hyperbolic"),
            Patch(facecolor="purple", alpha=0.8, label="Quasi-hyp"),
        ]
        ax2.legend(
            handles=legend_elements_nc, loc="lower left", fontsize=7, framealpha=0.9
        )

        fig2.suptitle(
            "Model Comparison: Conflicts Removed",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()

        nc_plot_path = comparison_dir / "accuracy_heatmap_no_conflicts.png"
        plt.savefig(nc_plot_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved: {nc_plot_path.relative_to(output_dir)}")

    return plot_path


def generate_model_comparison_json(
    model_results: dict[str, list[HorizonModelResult]],
) -> dict:
    """Generate JSON analysis of model performance by horizon type."""
    result = {
        "summary": {},
        "by_horizon": {},
        "by_model": {},
        "insights": [],
    }

    model_names = list(model_results.keys())
    if not model_names:
        return result

    horizons = [hr.horizon for hr in model_results[model_names[0]]]

    # Categorize horizons
    short_term_horizons = []
    long_term_horizons = []
    no_horizon = None

    for h in horizons:
        if h == "no_horizon":
            no_horizon = h
        else:
            sort_key = horizon_sort_key(h)
            if sort_key[0] < 1.0:
                short_term_horizons.append(h)
            else:
                long_term_horizons.append(h)

    # Compute accuracies
    for horizon in horizons:
        horizon_data = {"models": {}, "best_model": None, "best_accuracy": 0}
        for model_name in model_names:
            for hr in model_results[model_name]:
                if hr.horizon == horizon:
                    if hr.horizon in hr.test_results:
                        acc = hr.test_results[hr.horizon].metrics.accuracy
                    else:
                        acc = hr.train_accuracy
                    horizon_data["models"][model_name] = acc
                    if acc > horizon_data["best_accuracy"]:
                        horizon_data["best_accuracy"] = acc
                        horizon_data["best_model"] = model_name
        result["by_horizon"][horizon] = horizon_data

    # Compute model performance across horizon types
    for model_name in model_names:
        short_term_accs = []
        long_term_accs = []

        for hr in model_results[model_name]:
            if hr.horizon in hr.test_results:
                acc = hr.test_results[hr.horizon].metrics.accuracy
            else:
                acc = hr.train_accuracy

            if hr.horizon in short_term_horizons:
                short_term_accs.append(acc)
            elif hr.horizon in long_term_horizons:
                long_term_accs.append(acc)

        result["by_model"][model_name] = {
            "short_term_avg": sum(short_term_accs) / len(short_term_accs)
            if short_term_accs
            else 0,
            "long_term_avg": sum(long_term_accs) / len(long_term_accs)
            if long_term_accs
            else 0,
            "preference": "short_term"
            if (sum(short_term_accs) / len(short_term_accs) if short_term_accs else 0)
            > (sum(long_term_accs) / len(long_term_accs) if long_term_accs else 0)
            else "long_term",
        }

    # Generate insights
    best_short_term_model = max(
        result["by_model"].items(), key=lambda x: x[1]["short_term_avg"]
    )[0]
    best_long_term_model = max(
        result["by_model"].items(), key=lambda x: x[1]["long_term_avg"]
    )[0]

    result["insights"] = [
        f"Best model for short-term horizons (<1 year): {best_short_term_model}",
        f"Best model for long-term horizons (>1 year): {best_long_term_model}",
        "Exponential discount models typically perform better on short-term horizons",
        "Quasi-hyperbolic models often excel at long-term horizons (present bias)",
    ]

    result["summary"] = {
        "short_term_horizons": short_term_horizons,
        "long_term_horizons": long_term_horizons,
        "best_short_term_model": best_short_term_model,
        "best_long_term_model": best_long_term_model,
    }

    return result


# -----------------------------------------------------------------------------
# Main Analysis Pipeline
# -----------------------------------------------------------------------------


def analyze_choices(
    config: AnalysisConfig,
    output_dir: Path,
    visualize: bool = False,
    random_seed: int = 42,
) -> AnalysisResult:
    """Run full choice analysis pipeline."""

    # Validate config
    if not config.train_query_ids and not config.train_test_query_ids:
        raise ValueError(
            "Config must specify either 'train_data' or 'train_test' query_ids"
        )

    # Handle train_test mode
    if config.train_test_query_ids:
        print(f"Train/test query IDs: {config.train_test_query_ids}")
        print(f"Train/test split ratio: {config.train_test_split}")
        print()

        # Load all samples (already filters out ambiguous samples)
        loaded = load_samples_from_query_ids(config.train_test_query_ids)
        all_samples = loaded.valid_samples
        all_ambiguous = loaded.ambiguous_samples
        print(
            f"Loaded {len(all_samples)} valid samples ({len(all_ambiguous)} ambiguous samples filtered out)"
        )

        # Filter other problematic samples BEFORE split
        all_samples, all_problematic = filter_problematic_samples(all_samples)
        if all_problematic:
            print(
                f"Filtered out {len(all_problematic)} other problematic samples before split"
            )

        # Class-balanced split
        train_samples, test_samples = class_balanced_train_test_split(
            all_samples,
            train_ratio=config.train_test_split,
            random_seed=random_seed,
        )
        print(f"Split: {len(train_samples)} train, {len(test_samples)} test")

        # Print class distribution
        train_choices = {"short_term": 0, "long_term": 0}
        test_choices = {"short_term": 0, "long_term": 0}
        for s in train_samples:
            train_choices[s.sample.chosen] = train_choices.get(s.sample.chosen, 0) + 1
        for s in test_samples:
            test_choices[s.sample.chosen] = test_choices.get(s.sample.chosen, 0) + 1
        print(f"Train choices: {train_choices}")
        print(f"Test choices: {test_choices}")

        train_data_path = find_preference_data_by_query_id(
            config.train_test_query_ids[0]
        )
        test_data_path = train_data_path
        train_query_ids = config.train_test_query_ids

        # Track for later saving (after run_output_dir is created)
        data_sources = {"train_test": str(train_data_path)}

    else:
        # Standard mode: separate train and test
        print(f"Train query IDs: {config.train_query_ids}")
        print(f"Test query IDs: {config.test_query_ids}")
        print()

        # Load samples (already filters out ambiguous)
        train_loaded = load_samples_from_query_ids(config.train_query_ids)
        train_samples = train_loaded.valid_samples
        train_ambiguous = train_loaded.ambiguous_samples

        if config.test_query_ids:
            test_loaded = load_samples_from_query_ids(config.test_query_ids)
            test_samples = test_loaded.valid_samples
            test_ambiguous = test_loaded.ambiguous_samples
        else:
            test_samples = []
            test_ambiguous = []

        all_ambiguous = train_ambiguous + test_ambiguous
        print(
            f"Loaded {len(train_samples)} train + {len(test_samples)} test valid samples"
        )
        print(f"Filtered out {len(all_ambiguous)} ambiguous samples")

        train_data_path = find_preference_data_by_query_id(config.train_query_ids[0])
        test_data_path = (
            find_preference_data_by_query_id(config.test_query_ids[0])
            if config.test_query_ids
            else train_data_path
        )
        train_query_ids = config.train_query_ids

        # Filter other problematic samples
        print("\nFiltering other problematic samples...")
        train_samples, train_problematic = filter_problematic_samples(train_samples)
        test_samples, test_problematic = filter_problematic_samples(test_samples)

        all_problematic = train_problematic + test_problematic
        if all_problematic:
            print(
                f"  Filtered out {len(train_problematic)} train + {len(test_problematic)} test other problematic"
            )

        # Track for later saving (after run_output_dir is created)
        data_sources = {"train": str(train_data_path), "test": str(test_data_path)}

    # Try to load dataset config for verification (use first query_id)
    dataset_config = None
    time_ranges = None
    # Skip verification for now - would need to extract name/model from preference data
    print("\nNote: Dataset config verification skipped (use query_id format)")

    # Bucket training samples
    train_buckets = bucket_samples_by_horizon(train_samples)

    print("\nTraining samples by horizon:")
    total_train = 0
    for horizon in sorted(train_buckets.keys(), key=horizon_sort_key):
        print(f"  {horizon}: {len(train_buckets[horizon])} samples")
        total_train += len(train_buckets[horizon])
    print(f"  TOTAL: {total_train} samples")

    if total_train == 0:
        raise ValueError("No valid training samples found")

    print(f"\nTest samples: {len(test_samples)}")

    # Skip data verification in query_id mode (would need to extract name/model from preference data)
    train_verification = None
    test_verification = None

    # Run consistency analysis on training data
    train_consistency = analyze_consistency(train_samples)
    print_consistency_analysis(train_consistency, title="TRAINING DATA ANALYSIS")

    # Run deep conflict analysis on training data
    print()
    print("=" * 80)
    print("  DEEP CONFLICT ANALYSIS (Training Data)")
    print("=" * 80)

    for horizon in sorted(train_consistency.keys(), key=horizon_sort_key):
        analysis = train_consistency[horizon]
        if not analysis.constraints:
            continue
        deep_analysis = analyze_conflicts_deeply(analysis.constraints)
        horizon_label = horizon if horizon != "no_horizon" else "No Horizon Given"
        print(f"\n  {horizon_label}:")
        print(
            f"    Consistent: {deep_analysis.n_consistent}/{deep_analysis.total_samples} "
            f"({deep_analysis.n_consistent / deep_analysis.total_samples:.1%})"
        )
        print(f"    Conflicting: {deep_analysis.n_conflicting}")

        if deep_analysis.n_conflicting > 0:
            if deep_analysis.conflicts_internally_consistent:
                print(
                    f"    → Conflicts ARE consistent among themselves (θ={deep_analysis.conflicts_best_theta:.4f})"
                )
            else:
                print(
                    f"    → Conflicts are NOT consistent ({deep_analysis.conflicts_best_accuracy:.0%} internal acc)"
                )

        if deep_analysis.n_clusters > 1:
            print(
                f"    Clusters: {deep_analysis.n_clusters} "
                f"(sizes: {deep_analysis.cluster_sizes[:5]}...)"
                if len(deep_analysis.cluster_sizes) > 5
                else f"    Clusters: {deep_analysis.n_clusters} (sizes: {deep_analysis.cluster_sizes})"
            )

    # Run consistency analysis on test data
    test_consistency = analyze_consistency(test_samples) if test_samples else None
    if test_consistency:
        print_consistency_analysis(test_consistency, title="TEST DATA ANALYSIS")

    # Train and test ALL model combinations
    model_results: dict[str, list[HorizonModelResult]] = {}
    all_horizon_model_results: dict[
        str, dict[str, HorizonModelResult]
    ] = {}  # horizon -> {model_name -> result}

    print(f"\n{'=' * 60}")
    print(f"Training {len(ALL_MODEL_SPECS)} model combinations...")
    print("=" * 60)

    for spec in ALL_MODEL_SPECS:
        horizon_results = train_and_test_by_horizon(
            spec,
            train_buckets,
            test_samples,
            learning_rate=config.learning_rate,
            num_iterations=config.num_iterations,
            temperature=config.temperature,
            horizon_sort_key=horizon_sort_key,
        )
        model_results[spec.name] = horizon_results

        # Collect by horizon for comparison
        for hr in horizon_results:
            if hr.horizon not in all_horizon_model_results:
                all_horizon_model_results[hr.horizon] = {}
            all_horizon_model_results[hr.horizon][spec.name] = hr

    # Find best model for each horizon and print comparison
    print(f"\n{'=' * 60}")
    print("Best Model Selection by Horizon")
    print("=" * 60)

    best_models_by_horizon: dict[str, tuple[str, HorizonModelResult]] = {}

    for horizon in sorted(all_horizon_model_results.keys(), key=horizon_sort_key):
        models = all_horizon_model_results[horizon]

        # Find best by test accuracy (or train if no test)
        best_name = None
        best_result = None
        best_acc = -1.0

        for name, result in models.items():
            # Use test accuracy if available, otherwise train
            if result.horizon in result.test_results:
                acc = result.test_results[result.horizon].metrics.accuracy
            else:
                acc = result.train_accuracy

            if acc > best_acc:
                best_acc = acc
                best_name = name
                best_result = result

        best_models_by_horizon[horizon] = (best_name, best_result)

        horizon_label = horizon if horizon != "no_horizon" else "No Horizon Given"
        print(f"\n  {horizon_label}:")
        print(f"    Best model: {best_name}")
        print(
            f"    θ={best_result.trained_params.discount.theta:.4f}, "
            f"train={best_result.train_accuracy:.0%}, test={best_acc:.0%}"
        )

        # Show top 3 models for comparison
        sorted_models = sorted(
            models.items(),
            key=lambda x: (
                x[1].test_results[x[1].horizon].metrics.accuracy
                if x[1].horizon in x[1].test_results
                else x[1].train_accuracy
            ),
            reverse=True,
        )[:3]
        print(
            f"    Top 3: {', '.join([f'{n}({r.train_accuracy:.0%})' for n, r in sorted_models])}"
        )

    # Also print progress for the main discount types for backwards compat
    for spec in [
        ALL_MODEL_SPECS[0],
        ALL_MODEL_SPECS[1],
    ]:  # exponential, hyperbolic with linear
        print(f"\n{'=' * 60}")
        print(f"Training {spec.name} models...")
        print("=" * 60)
        for hr in model_results[spec.name]:
            test_result = hr.test_results.get(hr.horizon)
            if test_result:
                test_acc = f"{test_result.metrics.accuracy:.0%}"
                test_n = test_result.metrics.num_samples
            else:
                test_acc = "---"
                test_n = 0
            horizon_label = (
                hr.horizon if hr.horizon != "no_horizon" else "No Horizon Given"
            )
            print(
                f"  {horizon_label:<15} | θ={hr.trained_params.discount.theta:.4f} | "
                f"train={hr.train_accuracy:.0%} | test={test_acc} (n={test_n})"
            )

    # Train models with conflicts filtered out
    print(f"\n{'=' * 60}")
    print("Training models WITHOUT conflicts...")
    print("=" * 60)

    # Filter consistent samples using the consistency analysis
    train_samples_no_conflicts = filter_consistent_samples(
        train_samples, train_consistency
    )
    test_samples_no_conflicts = (
        filter_consistent_samples(test_samples, test_consistency)
        if test_consistency
        else test_samples
    )

    n_train_removed = len(train_samples) - len(train_samples_no_conflicts)
    n_test_removed = len(test_samples) - len(test_samples_no_conflicts)
    print(
        f"  Filtered {n_train_removed} train conflicts, {n_test_removed} test conflicts"
    )
    print(
        f"  Remaining: {len(train_samples_no_conflicts)} train, {len(test_samples_no_conflicts)} test"
    )

    # Bucket filtered training samples
    train_buckets_no_conflicts = bucket_samples_by_horizon(train_samples_no_conflicts)

    # Train all models on filtered data
    model_results_no_conflicts: dict[str, list[HorizonModelResult]] = {}
    for spec in ALL_MODEL_SPECS:
        horizon_results = train_and_test_by_horizon(
            spec,
            train_buckets_no_conflicts,
            test_samples_no_conflicts,
            learning_rate=config.learning_rate,
            num_iterations=config.num_iterations,
            temperature=config.temperature,
            horizon_sort_key=horizon_sort_key,
        )
        model_results_no_conflicts[spec.name] = horizon_results

    # Build result
    timestamp = get_timestamp()
    result = AnalysisResult(
        config=config,
        train_data_path=str(train_data_path.relative_to(PROJECT_ROOT)),
        test_data_path=str(test_data_path.relative_to(PROJECT_ROOT)),
        model_results=model_results,
        model_results_no_conflicts=model_results_no_conflicts,
        train_verification=train_verification,
        test_verification=test_verification,
        train_consistency=train_consistency,
        test_consistency=test_consistency,
        timestamp=timestamp,
    )

    # Serialize and save
    output_dict = serialize_analysis_result(result)
    # Create folder based on config schema ID (includes all params: query_ids, split ratio, etc.)
    folder_name = config.get_id()
    run_output_dir = output_dir / folder_name
    ensure_dir(run_output_dir)

    # Save config so users know what parameters created this run
    from dataclasses import asdict

    config_output = {
        "schema_id": config.get_id(),
        "config": asdict(config),
        "data_sources": data_sources,
    }
    save_json(config_output, run_output_dir / "config.json")

    # Save excluded samples as separate files inside run folder
    if all_ambiguous:
        ambiguous_output = {
            "data_sources": data_sources,
            "description": "Samples where time_horizon is between short_term_time and long_term_time (unclear expected choice)",
            "total": len(all_ambiguous),
            "samples": [
                {
                    "sample_id": s.sample_id,
                    "time_horizon_years": s.time_horizon_years,
                    "time_short_years": s.time_short_years,
                    "time_long_years": s.time_long_years,
                    "reward_short": s.reward_short,
                    "reward_long": s.reward_long,
                    "choice": s.choice,
                    "reason": s.reason,
                }
                for s in all_ambiguous
            ],
        }
        save_json(ambiguous_output, run_output_dir / "ambiguous_samples.json")
        print(f"Saved {len(all_ambiguous)} ambiguous samples to ambiguous_samples.json")

    # Always save problematic samples file (even if empty - confirms data quality)
    problematic_output = {
        "data_sources": data_sources,
        "description": "Samples where long_term_time <= short_term_time (data quality issue)",
        **serialize_problematic_samples(all_problematic),
    }
    save_json(problematic_output, run_output_dir / "problematic_samples.json")
    if all_problematic:
        print(
            f"Saved {len(all_problematic)} problematic samples to problematic_samples.json"
        )
    else:
        print("No problematic samples found (good data quality) - saved empty report")

    output_path = save_analysis_output(
        output_dict, run_output_dir, "results", timestamp
    )
    print(f"\nResults saved to: {output_path}")

    # Save detailed analysis to subfolders with readable names
    if train_consistency:
        train_dir = run_output_dir / "train"
        ensure_dir(train_dir)

        train_conflicts = generate_conflicts_analysis(
            train_consistency,
            data_source=str(train_data_path.relative_to(PROJECT_ROOT)),
        )
        save_json(train_conflicts, train_dir / "conflicts.json")

        train_alignment = generate_alignment_analysis(
            train_consistency,
            data_source=str(train_data_path.relative_to(PROJECT_ROOT)),
        )
        save_json(train_alignment, train_dir / "horizon_alignment.json")

        deep_conflicts = generate_deep_conflict_analysis(
            train_consistency,
            data_source=str(train_data_path.relative_to(PROJECT_ROOT)),
        )
        save_json(deep_conflicts, train_dir / "deep_conflict_analysis.json")

        print(f"Train analysis saved to: {train_dir}/")

    if test_consistency:
        test_dir = run_output_dir / "test"
        ensure_dir(test_dir)

        test_conflicts = generate_conflicts_analysis(
            test_consistency, data_source=str(test_data_path.relative_to(PROJECT_ROOT))
        )
        save_json(test_conflicts, test_dir / "conflicts.json")

        test_alignment = generate_alignment_analysis(
            test_consistency, data_source=str(test_data_path.relative_to(PROJECT_ROOT))
        )
        save_json(test_alignment, test_dir / "horizon_alignment.json")

        print(f"Test analysis saved to: {test_dir}/")

    # Create visualizations if requested
    if visualize:
        print("\nCreating visualizations...")
        viz_dir = run_output_dir / "viz"
        ensure_dir(viz_dir)
        result.viz_dir = viz_dir  # Store for camera-ready export

        create_visualizations(result, viz_dir)

        # Create model comparison heatmap and analysis (include conflicts-removed)
        plot_path = create_model_comparison_heatmap(
            model_results, viz_dir, model_results_no_conflicts
        )
        if plot_path:
            print(f"  Saved: {plot_path.relative_to(viz_dir)}")

        # Save model comparison JSON
        comparison_data = generate_model_comparison_json(model_results)
        comparison_dir = viz_dir / "model_comparison"
        ensure_dir(comparison_dir)
        save_json(comparison_data, comparison_dir / "analysis.json")
        print("  Saved: model_comparison/analysis.json")

    return result


# -----------------------------------------------------------------------------
# Camera-Ready Export - uses src/plotting/export.py
# The export_camera_ready function is imported from src.plotting at the top
# It exports to behavior/ folder and copies to paper/behavior/
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def get_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze LLM choices by training and testing choice models"
    )
    parser.add_argument(
        "--config",
        type=str,
        nargs="*",
        default=["default_analysis"],
        help="Analysis config name(s) from configs/choice_modeling/ (default: default_analysis)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: out/choice_modeling/)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate for optimization (default: 0.01)",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=1000,
        help="Number of optimization iterations (default: 100)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for softmax choice probability (0=deterministic, default: 0.0)",
    )
    parser.add_argument(
        "--camera-ready",
        action="store_true",
        help="Export plots to final/ folder with descriptive filenames for publication",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Show profiling summary",
    )
    return parser.parse_args()


def main() -> int:
    args = get_args()

    # Output directory
    output_dir = args.output
    if output_dir is None:
        output_dir = PROJECT_ROOT / "out" / "choice_modeling"

    for config_name in args.config:
        config_path = (
            SCRIPTS_DIR / "configs" / "choice_modeling" / f"{config_name}.json"
        )
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        config = load_analysis_config(config_path)

        # Set CLI args on config
        config.learning_rate = args.learning_rate
        config.num_iterations = args.num_iterations
        config.temperature = args.temperature

        # Run analysis (always generate plots)
        result = analyze_choices(config, output_dir, visualize=True)
        print_summary(result)

        # Export camera-ready plots if requested
        if args.camera_ready and result.viz_dir:
            export_camera_ready(
                result.viz_dir, output_name="behavior", paper_dir=PAPER_DIR
            )

    if args.profile:
        profiler = get_profiler()
        print()
        print(profiler.summary())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
