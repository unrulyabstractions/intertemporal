"""
Data verification for intertemporal choice analysis.

Verifies that model choices align with expectations based on time horizon:
- If horizon <= short_term range: expect short_term choice
- If horizon >= long_term range: expect long_term choice
- If horizon is between: ambiguous (no expectation)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.types import TimeValue

from .config_utils import categorize_time_horizon, get_expected_choice


@dataclass
class CategoryStats:
    """Statistics for a time horizon category."""

    category: str
    count: int
    expected_choice: Optional[str]  # None if ambiguous
    chose_short: int
    chose_long: int
    aligned: int  # chose as expected
    alignment_rate: float


@dataclass
class DataVerification:
    """Results of data verification against expected choices."""

    time_ranges: dict  # From dataset config
    category_stats: list[CategoryStats]
    total_samples: int
    total_aligned: int
    total_with_expectation: int  # Samples where we have an expectation
    overall_alignment: float
    dataset_label: str = ""  # Label for display (e.g., "Train" or "Test")


def verify_data_alignment(
    samples: list,  # list[SampleWithHorizon] - avoid circular import
    time_ranges: dict,
    dataset_label: str = "",
) -> DataVerification:
    """
    Verify that model choices align with expectations based on time horizon.

    Expected behavior:
    - If horizon <= short_term range: expect short_term choice
    - If horizon >= long_term range: expect long_term choice
    - If horizon is between: ambiguous (no expectation)

    Args:
        samples: Samples with horizon info (must have .horizon_key and .sample.chosen)
        time_ranges: Dict from get_time_ranges_from_dataset()
        dataset_label: Label for display (e.g., "Train" or "Test")

    Returns:
        DataVerification with statistics
    """
    from collections import defaultdict

    # Collect stats by category
    category_data: dict[str, dict] = defaultdict(
        lambda: {"count": 0, "chose_short": 0, "chose_long": 0}
    )

    for s in samples:
        # Skip no_horizon samples - they have no time-based expectation
        if s.horizon_key == "no_horizon":
            category_data["no_horizon"]["count"] += 1
            if s.sample.chosen == "short_term":
                category_data["no_horizon"]["chose_short"] += 1
            else:
                category_data["no_horizon"]["chose_long"] += 1
            continue

        # Parse horizon to years
        parts = s.horizon_key.split()
        if len(parts) != 2:
            continue
        try:
            tv = TimeValue(float(parts[0]), parts[1])
            horizon_years = tv.to_years()
        except (ValueError, KeyError):
            continue

        category = categorize_time_horizon(horizon_years, time_ranges)
        category_data[category]["count"] += 1

        if s.sample.chosen == "short_term":
            category_data[category]["chose_short"] += 1
        else:
            category_data[category]["chose_long"] += 1

    # Build category stats
    category_order = [
        "below_short",
        "within_short",
        "between",
        "within_long",
        "above_long",
        "no_horizon",
    ]
    category_stats = []
    total_aligned = 0
    total_with_expectation = 0

    for cat in category_order:
        if cat not in category_data:
            continue

        data = category_data[cat]
        expected = get_expected_choice(cat) if cat != "no_horizon" else None

        aligned = 0
        if expected == "short_term":
            aligned = data["chose_short"]
            total_with_expectation += data["count"]
        elif expected == "long_term":
            aligned = data["chose_long"]
            total_with_expectation += data["count"]
        # If expected is None (ambiguous), aligned stays 0

        total_aligned += aligned

        alignment_rate = aligned / data["count"] if data["count"] > 0 else 0.0

        category_stats.append(
            CategoryStats(
                category=cat,
                count=data["count"],
                expected_choice=expected,
                chose_short=data["chose_short"],
                chose_long=data["chose_long"],
                aligned=aligned,
                alignment_rate=alignment_rate,
            )
        )

    overall_alignment = (
        total_aligned / total_with_expectation if total_with_expectation > 0 else 0.0
    )

    return DataVerification(
        time_ranges=time_ranges,
        category_stats=category_stats,
        total_samples=len(samples),
        total_aligned=total_aligned,
        total_with_expectation=total_with_expectation,
        overall_alignment=overall_alignment,
        dataset_label=dataset_label,
    )


def print_data_verification(verification: DataVerification) -> None:
    """Print data verification results."""
    label = verification.dataset_label
    header = f"DATA VERIFICATION: {label} - Time Horizon vs Model Choices" if label else "DATA VERIFICATION: Time Horizon vs Model Choices"

    print()
    print("=" * 80)
    print(f"  {header}")
    print("=" * 80)
    print()

    tr = verification.time_ranges
    print("  Option Time Ranges (from dataset config):")
    print(
        f"    Short-term: {tr['short_term']['min']:.2f} - {tr['short_term']['max']:.2f} years"
    )
    print(
        f"    Long-term:  {tr['long_term']['min']:.2f} - {tr['long_term']['max']:.2f} years"
    )
    print()

    print(
        f"  {'Category':<15} {'N':>5} {'Expected':>12} {'Chose ST':>10} {'Chose LT':>10} {'Aligned':>10}"
    )
    print("  " + "-" * 67)

    for cs in verification.category_stats:
        expected_str = cs.expected_choice or "ambiguous"
        aligned_str = f"{cs.alignment_rate:.0%}" if cs.expected_choice else "n/a"

        print(
            f"  {cs.category:<15} {cs.count:>5} {expected_str:>12} "
            f"{cs.chose_short:>10} {cs.chose_long:>10} {aligned_str:>10}"
        )

    print()
    print(
        f"  OVERALL ALIGNMENT: {verification.overall_alignment:.1%} "
        f"({verification.total_aligned}/{verification.total_with_expectation} samples with clear expectation)"
    )
    print()

    # Interpretation
    if verification.overall_alignment >= 0.8:
        print(
            "  ✓ STRONG ALIGNMENT: Model choices strongly match time horizon expectations"
        )
    elif verification.overall_alignment >= 0.6:
        print("  ~ MODERATE ALIGNMENT: Model shows some sensitivity to time horizon")
    elif verification.overall_alignment >= 0.4:
        print("  ? WEAK ALIGNMENT: Model choices weakly correlated with time horizon")
    else:
        print(
            "  ✗ NO ALIGNMENT: Model choices do not follow time horizon expectations"
        )
    print()


def serialize_verification(verification: DataVerification) -> dict:
    """Serialize verification results to dict for JSON output."""
    return {
        "dataset_label": verification.dataset_label,
        "time_ranges": verification.time_ranges,
        "overall_alignment": verification.overall_alignment,
        "total_samples": verification.total_samples,
        "total_aligned": verification.total_aligned,
        "total_with_expectation": verification.total_with_expectation,
        "categories": [
            {
                "category": cs.category,
                "count": cs.count,
                "expected_choice": cs.expected_choice,
                "chose_short": cs.chose_short,
                "chose_long": cs.chose_long,
                "alignment_rate": cs.alignment_rate,
            }
            for cs in verification.category_stats
        ],
    }


def create_verification_plot(
    verifications: list[DataVerification],
    output_dir: Path,
    timestamp: str,
) -> Optional[Path]:
    """
    Create data verification visualization for one or more datasets.

    Args:
        verifications: List of DataVerification objects to plot
        output_dir: Directory to save plot
        timestamp: Timestamp for filename

    Returns:
        Path to saved plot, or None if matplotlib not available
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Warning: matplotlib not installed. Skipping verification plot.")
        return None

    # Filter to verifications with data
    valid_verifications = [
        v for v in verifications if v.category_stats and any(cs.count > 0 for cs in v.category_stats)
    ]
    if not valid_verifications:
        return None

    n_datasets = len(valid_verifications)

    # Create figure - one row per dataset if multiple
    if n_datasets == 1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes_list = [axes]
    else:
        fig, axes = plt.subplots(n_datasets, 2, figsize=(14, 5 * n_datasets))
        axes_list = axes if n_datasets > 1 else [axes]

    fig.suptitle(
        "Data Verification: Time Horizon vs Model Choices",
        fontsize=14,
        fontweight="bold",
    )

    for idx, verification in enumerate(valid_verifications):
        ax1, ax2 = axes_list[idx] if n_datasets > 1 else (axes_list[0][0], axes_list[0][1])

        stats = [cs for cs in verification.category_stats if cs.count > 0]
        label = verification.dataset_label or f"Dataset {idx + 1}"

        # Plot 1: Choice distribution by category
        categories = [cs.category for cs in stats]
        chose_short = [cs.chose_short for cs in stats]
        chose_long = [cs.chose_long for cs in stats]

        x = np.arange(len(categories))
        width = 0.35

        bars1 = ax1.bar(
            x - width / 2,
            chose_short,
            width,
            label="Chose Short-term",
            color="coral",
            alpha=0.8,
        )
        bars2 = ax1.bar(
            x + width / 2,
            chose_long,
            width,
            label="Chose Long-term",
            color="steelblue",
            alpha=0.8,
        )

        ax1.set_xlabel("Time Horizon Category", fontsize=11)
        ax1.set_ylabel("Number of Choices", fontsize=11)
        ax1.set_title(f"{label}: Choice Distribution by Horizon Category", fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels([c.replace("_", "\n") for c in categories], fontsize=9)
        ax1.legend(loc="upper right")

        # Add count labels
        for bar, val in zip(bars1, chose_short):
            if val > 0:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.2,
                    str(val),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        for bar, val in zip(bars2, chose_long):
            if val > 0:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.2,
                    str(val),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        # Plot 2: Alignment rate by category
        align_cats = []
        align_rates = []
        colors = []

        for cs in stats:
            if cs.expected_choice:
                align_cats.append(cs.category)
                align_rates.append(cs.alignment_rate * 100)
                if cs.expected_choice == "short_term":
                    colors.append("coral")
                else:
                    colors.append("steelblue")

        if align_cats:
            x2 = np.arange(len(align_cats))
            bars = ax2.bar(x2, align_rates, color=colors, alpha=0.8)

            ax2.set_xlabel("Time Horizon Category", fontsize=11)
            ax2.set_ylabel("Alignment Rate (%)", fontsize=11)
            ax2.set_title(f"{label}: Choice Alignment with Expectations", fontsize=12)
            ax2.set_xticks(x2)
            ax2.set_xticklabels(
                [c.replace("_", "\n") for c in align_cats], fontsize=9
            )
            ax2.set_ylim(0, 105)
            ax2.axhline(
                y=50,
                color="gray",
                linestyle="--",
                linewidth=1,
                alpha=0.5,
                label="Chance (50%)",
            )

            # Add overall alignment line
            ax2.axhline(
                y=verification.overall_alignment * 100,
                color="green",
                linestyle="-",
                linewidth=2,
                alpha=0.7,
                label=f"Overall ({verification.overall_alignment:.0%})",
            )
            ax2.legend(loc="lower right")

            # Add value labels
            for bar, rate in zip(bars, align_rates):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{rate:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        else:
            ax2.text(
                0.5,
                0.5,
                "No categories with expectations",
                transform=ax2.transAxes,
                ha="center",
                va="center",
                fontsize=12,
            )

    plt.tight_layout()

    # Save
    plot_path = output_dir / f"verification_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    return plot_path
