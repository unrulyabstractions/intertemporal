"""
Consistency analysis for intertemporal choice data.

Analyzes whether choices can be explained by a single discount rate θ.
Detects conflicts where different samples require incompatible θ values.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def horizon_sort_key(horizon_key: str) -> tuple[float, str]:
    """
    Return a sort key for horizon_key that orders by time duration.

    Parses strings like "6 months", "2 years" and returns time in years.
    "no_horizon" sorts to the end.
    """
    if horizon_key == "no_horizon":
        return (float("inf"), horizon_key)

    # Parse "N unit" format
    match = re.match(
        r"(\d+(?:\.\d+)?)\s*(months?|years?|days?|weeks?)", horizon_key, re.IGNORECASE
    )
    if match:
        value = float(match.group(1))
        unit = match.group(2).lower()

        # Convert to years
        if unit.startswith("day"):
            years = value / 365
        elif unit.startswith("week"):
            years = value / 52
        elif unit.startswith("month"):
            years = value / 12
        else:  # years
            years = value

        return (years, horizon_key)

    # Fallback: alphabetical
    return (0, horizon_key)


def format_horizon_label(horizon_key: str) -> str:
    """
    Format horizon key for display with proper grammar.

    - "no_horizon" -> "No Horizon Given"
    - "1 months" -> "1 month" (fix plural)
    - "1 years" -> "1 year" (fix plural)
    - Other values unchanged
    """
    if horizon_key == "no_horizon":
        return "No Horizon Given"

    # Fix singular/plural: "1 months" -> "1 month", "1 years" -> "1 year"
    match = re.match(r"^1\s+(months|years|days|weeks)$", horizon_key, re.IGNORECASE)
    if match:
        unit = match.group(1).lower()
        # Remove trailing 's' for singular
        return f"1 {unit.rstrip('s')}"

    return horizon_key


def sorted_horizon_keys(analyses: dict[str, "ConsistencyAnalysis"]) -> list[str]:
    """Return horizon keys sorted by time duration."""
    return sorted(analyses.keys(), key=horizon_sort_key)


@dataclass
class ThetaConstraint:
    """Constraint on θ from a single sample."""

    # Option details
    reward_short: float
    time_short: float  # in years
    reward_long: float
    time_long: float  # in years

    # What the LLM chose
    choice: str  # "short_term" or "long_term"

    # Derived constraint
    theta_threshold: float  # θ must be < or > this value
    constraint_type: str  # "less_than" or "greater_than"

    # Context
    horizon_key: str = ""
    time_horizon: Optional[float] = None  # in years, None for no_horizon

    @classmethod
    def from_sample(
        cls,
        reward_short: float,
        time_short: float,
        reward_long: float,
        time_long: float,
        choice: str,
        horizon_key: str = "",
        time_horizon: Optional[float] = None,
    ) -> "ThetaConstraint":
        """
        Compute θ constraint from a sample.

        For long_term to win: r_long * exp(-θ*t_long) > r_short * exp(-θ*t_short)
        This gives: θ < ln(r_long/r_short) / (t_long - t_short)
        """
        if time_long <= time_short:
            # Edge case: long option has shorter/equal time
            theta_threshold = float("inf")
        elif reward_long <= 0 or reward_short <= 0:
            theta_threshold = float("inf")
        else:
            theta_threshold = math.log(reward_long / reward_short) / (
                time_long - time_short
            )

        # If choice is long_term, need θ < threshold
        # If choice is short_term, need θ > threshold
        if choice == "long_term":
            constraint_type = "less_than"
        else:
            constraint_type = "greater_than"

        return cls(
            reward_short=reward_short,
            time_short=time_short,
            reward_long=reward_long,
            time_long=time_long,
            choice=choice,
            theta_threshold=theta_threshold,
            constraint_type=constraint_type,
            horizon_key=horizon_key,
            time_horizon=time_horizon,
        )

    def is_satisfied(self, theta: float) -> bool:
        """Check if this constraint is satisfied by given θ."""
        if self.constraint_type == "less_than":
            return theta < self.theta_threshold
        else:
            return theta > self.theta_threshold

    def horizon_alignment(self) -> Optional[tuple[str, bool, str]]:
        """
        Check if choice aligns with time horizon range.

        Logic based on which options deliver within the planning horizon:
        - horizon < short_term_time: Neither delivers, but short-term is closer → expect short_term
        - short_term_time <= horizon < long_term_time: Only short-term delivers → expect short_term
        - horizon >= long_term_time: Both deliver, long-term has higher reward → expect long_term

        Returns:
            None if time_horizon is not set
            (expected_choice, is_aligned, alignment_type) otherwise
            - expected_choice: "short_term" or "long_term"
            - is_aligned: True if actual choice matches expected
            - alignment_type: "before_both", "between", or "after_both"
        """
        if self.time_horizon is None:
            return None

        if self.time_horizon < self.time_short:
            # Horizon ends before short-term reward arrives
            # Neither delivers within horizon, but short-term is closer
            expected = "short_term"
            alignment_type = "before_both"
        elif self.time_horizon >= self.time_long:
            # Horizon extends past long-term reward
            # Both options deliver within horizon, long-term has higher reward
            expected = "long_term"
            alignment_type = "after_both"
        else:
            # Horizon is between: short_term_time <= horizon < long_term_time
            # Only short-term delivers within planning horizon
            expected = "short_term"
            alignment_type = "between"

        is_aligned = self.choice == expected
        return (expected, is_aligned, alignment_type)

    def conflicts_with(self, other: "ThetaConstraint") -> bool:
        """Check if this constraint conflicts with another."""
        # Conflict if one needs θ < X and other needs θ > Y where Y >= X
        if (
            self.constraint_type == "less_than"
            and other.constraint_type == "greater_than"
        ):
            return other.theta_threshold >= self.theta_threshold
        elif (
            self.constraint_type == "greater_than"
            and other.constraint_type == "less_than"
        ):
            return self.theta_threshold >= other.theta_threshold
        return False


@dataclass
class ConsistencyAnalysis:
    """Results of consistency analysis for a set of samples."""

    horizon_key: str
    constraints: list[ThetaConstraint]

    # Feasible θ range (if any)
    theta_min: float = 0.0  # Lower bound from "greater_than" constraints
    theta_max: float = float("inf")  # Upper bound from "less_than" constraints
    is_consistent: bool = True

    # Conflict details
    conflicts: list[tuple[ThetaConstraint, ThetaConstraint]] = field(
        default_factory=list
    )

    # Best achievable accuracy
    best_theta: float = 0.0
    best_accuracy: float = 0.0
    num_short_term_choices: int = 0
    num_long_term_choices: int = 0

    # Horizon alignment stats (only for samples with time_horizon)
    horizon_aligned: int = 0  # Number of choices aligned with horizon proximity
    horizon_total: int = 0  # Total samples with time_horizon (excludes no_horizon)

    def is_sample_consistent(self, constraint: ThetaConstraint) -> bool:
        """Check if a single sample is consistent with the best theta."""
        if constraint.constraint_type == "less_than":
            return self.best_theta < constraint.theta_threshold
        else:  # greater_than
            return self.best_theta > constraint.theta_threshold

    @classmethod
    def analyze(
        cls, constraints: list[ThetaConstraint], horizon_key: str = ""
    ) -> "ConsistencyAnalysis":
        """Analyze consistency of a set of constraints."""
        if not constraints:
            return cls(horizon_key=horizon_key, constraints=[])

        # Find bounds
        less_than_bounds = [
            c.theta_threshold
            for c in constraints
            if c.constraint_type == "less_than" and c.theta_threshold < float("inf")
        ]
        greater_than_bounds = [
            c.theta_threshold
            for c in constraints
            if c.constraint_type == "greater_than" and c.theta_threshold < float("inf")
        ]

        theta_max = min(less_than_bounds) if less_than_bounds else float("inf")
        theta_min = max(greater_than_bounds) if greater_than_bounds else 0.0

        is_consistent = theta_min < theta_max

        # Find specific conflicts
        conflicts = []
        for i, c1 in enumerate(constraints):
            for c2 in constraints[i + 1 :]:
                if c1.conflicts_with(c2):
                    conflicts.append((c1, c2))

        # Count choices
        num_short = sum(1 for c in constraints if c.choice == "short_term")
        num_long = sum(1 for c in constraints if c.choice == "long_term")

        # Find best θ (one that maximizes accuracy)
        # Try θ values at each threshold boundary
        test_thetas = [0.001] + sorted(
            set(
                c.theta_threshold
                for c in constraints
                if 0 < c.theta_threshold < float("inf")
            )
        )

        best_theta = 0.001
        best_accuracy = 0.0

        for theta in test_thetas:
            # Try just below and just above each threshold
            for test_val in [theta - 0.0001, theta + 0.0001]:
                if test_val <= 0:
                    continue
                correct = sum(1 for c in constraints if c.is_satisfied(test_val))
                acc = correct / len(constraints)
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_theta = test_val

        # Compute horizon alignment stats (excluding ambiguous cases)
        horizon_aligned = 0
        horizon_total = 0
        for c in constraints:
            alignment = c.horizon_alignment()
            if alignment is not None:
                expected, is_aligned, alignment_type = alignment
                # Only count non-ambiguous samples
                if expected != "ambiguous":
                    horizon_total += 1
                    if is_aligned:
                        horizon_aligned += 1

        return cls(
            horizon_key=horizon_key,
            constraints=constraints,
            theta_min=theta_min,
            theta_max=theta_max,
            is_consistent=is_consistent,
            conflicts=conflicts,
            best_theta=best_theta,
            best_accuracy=best_accuracy,
            num_short_term_choices=num_short,
            num_long_term_choices=num_long,
            horizon_aligned=horizon_aligned,
            horizon_total=horizon_total,
        )


def print_consistency_analysis(
    analyses: dict[str, ConsistencyAnalysis],
    title: str = "TRAINING DATA ANALYSIS",
) -> None:
    """Print consistency analysis for all horizon buckets."""
    print()
    print("=" * 80)
    print(f"  {title}: Theoretical Model Fit Upper Bounds")
    print("=" * 80)
    print()
    print("  Shows the MAXIMUM possible accuracy for each time horizon category,")
    print("  assuming we fit the optimal discount rate θ. Conflicts = samples that")
    print("  cannot be correctly predicted by ANY single discount rate.")
    print()

    # Header
    print(
        f"  {'Horizon Category':<20} │ {'Samples':>8} │ {'Short':>6} {'Long':>6} │ {'Max Acc':>8} {'Conflicts':>10} │ {'Optimal θ':>10}"
    )
    print(
        "  "
        + "-" * 21
        + "┼"
        + "-" * 10
        + "┼"
        + "-" * 14
        + "┼"
        + "-" * 20
        + "┼"
        + "-" * 11
    )

    total_samples = 0
    total_conflicts = 0

    for horizon_key in sorted_horizon_keys(analyses):
        analysis = analyses[horizon_key]
        n = len(analysis.constraints)
        if n == 0:
            continue

        horizon_label = (
            horizon_key if horizon_key != "no_horizon" else "No Horizon Given"
        )
        max_acc = f"{analysis.best_accuracy:.0%}"
        best_theta = f"{analysis.best_theta:.4f}"

        # Conflicts = samples that can't be correctly predicted even with optimal θ
        n_conflicts = n - int(analysis.best_accuracy * n + 0.5)
        conflict_pct = f"{(1 - analysis.best_accuracy):.0%}"

        total_samples += n
        total_conflicts += n_conflicts

        print(
            f"  {horizon_label:<20} │ {n:>8} │ {analysis.num_short_term_choices:>6} {analysis.num_long_term_choices:>6} │ {max_acc:>8} {conflict_pct:>6} ({n_conflicts:>2}) │ {best_theta:>10}"
        )

    print(
        "  "
        + "-" * 21
        + "┼"
        + "-" * 10
        + "┼"
        + "-" * 14
        + "┼"
        + "-" * 20
        + "┼"
        + "-" * 11
    )

    # Totals row
    if total_samples > 0:
        total_conflict_pct = total_conflicts / total_samples
        total_max_acc = 1 - total_conflict_pct
        print(
            f"  {'TOTAL':<20} │ {total_samples:>8} │ {'':>6} {'':>6} │ {total_max_acc:>7.0%} {total_conflict_pct:>6.0%} ({total_conflicts:>2}) │"
        )

    print()

    # Summary interpretation
    if total_conflicts == 0:
        print(
            "  ✓ No conflicts: 100% accuracy achievable (choices are internally consistent)"
        )
    else:
        print(
            f"  ⚠ {total_conflicts}/{total_samples} samples ({total_conflict_pct:.1%}) have inherent conflicts"
        )
        print("    (These cannot be correctly predicted by ANY single discount rate)")

    # Horizon alignment summary (aggregate across all categories with time_horizon)
    total_aligned = sum(a.horizon_aligned for a in analyses.values())
    total_with_horizon = sum(a.horizon_total for a in analyses.values())

    if total_with_horizon > 0:
        alignment_rate = total_aligned / total_with_horizon
        print()
        print(
            f"  Time Horizon Alignment: {total_aligned}/{total_with_horizon} ({alignment_rate:.1%}) choices match"
        )
        print(
            "    proximity expectation (if horizon closer to short_term option → expect short_term choice)"
        )
        if alignment_rate < 0.5:
            print(
                "    ⚠ Low alignment suggests LLM ignores time horizon proximity in decisions"
            )
        elif alignment_rate > 0.7:
            print("    ✓ High alignment suggests LLM considers time horizon proximity")

    print()


def create_consistency_plot(
    analyses: dict[str, ConsistencyAnalysis],
    output_dir: Path,
    title_prefix: str = "",
) -> Optional[Path]:
    """
    Create visualization showing whether LLM choices can be explained by
    a single consistent discount rate (θ).

    Key insight: If an LLM makes consistent choices based on discounting future
    rewards, there should exist a single θ that explains all its choices.
    Conflicts indicate the LLM doesn't behave like a rational discounter.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Warning: matplotlib not installed. Skipping consistency plot.")
        return None

    # Custom ordering: short-term (<1yr), then "No Horizon Given", then long-term (>1yr)
    short_term_items = []
    long_term_items = []
    no_horizon_item = None

    for horizon_key in sorted_horizon_keys(analyses):
        analysis = analyses[horizon_key]
        if not analysis.constraints:
            continue

        if horizon_key == "no_horizon":
            no_horizon_item = (horizon_key, analysis)
        else:
            sort_key = horizon_sort_key(horizon_key)
            if sort_key[0] < 1.0:
                short_term_items.append((horizon_key, analysis))
            else:
                long_term_items.append((horizon_key, analysis))

    # Build ordered list: no_horizon first (at origin), then short-term, then long-term
    ordered_items = []
    if no_horizon_item:
        ordered_items.append(no_horizon_item)
    ordered_items.extend(short_term_items)
    ordered_items.extend(long_term_items)

    if not ordered_items:
        return None

    # Prepare data
    categories = []
    n_samples = []
    best_accuracies = []
    n_conflicts_list = []
    best_thetas = []
    # 1-year line after no_horizon + short-term items
    no_horizon_offset = 1 if no_horizon_item else 0
    one_year_line_pos = no_horizon_offset + len(short_term_items) - 0.5

    for horizon_key, analysis in ordered_items:
        categories.append(format_horizon_label(horizon_key))
        n = len(analysis.constraints)
        n_samples.append(n)
        best_accuracies.append(analysis.best_accuracy * 100)
        n_correct = int(analysis.best_accuracy * n + 0.5)
        n_conflicts_list.append(n - n_correct)
        best_thetas.append(analysis.best_theta)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    title = (
        f"{title_prefix}Do LLM Choices Follow a Consistent Discount Rate?"
        if title_prefix
        else "Do LLM Choices Follow a Consistent Discount Rate?"
    )
    fig.suptitle(title, fontsize=14, fontweight="bold", fontfamily="DejaVu Sans")

    # Add explanatory subtitle
    fig.text(
        0.5,
        0.93,
        "If choices are rational, ONE discount rate (θ) should predict ALL choices. "
        "Lower bars = more contradictory/irrational choices.",
        ha="center",
        fontsize=10,
        style="italic",
        color="gray",
    )

    # Plot 1: Best achievable accuracy (primary metric)
    ax1 = axes[0]
    x = np.arange(len(categories))

    # Color based on how consistent (green=high, red=low)
    colors = []
    for acc in best_accuracies:
        if acc >= 90:
            colors.append("forestgreen")
        elif acc >= 70:
            colors.append("yellowgreen")
        elif acc >= 50:
            colors.append("orange")
        else:
            colors.append("crimson")

    bars = ax1.bar(x, best_accuracies, color=colors, alpha=0.8)

    # Add 50% (random) and 100% (perfect) reference lines
    ax1.axhline(
        y=50,
        color="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label="Random baseline (50%)",
    )
    ax1.axhline(
        y=100,
        color="green",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label="Perfect consistency (100%)",
    )

    # Add 1-year separator
    if one_year_line_pos >= 0:
        ax1.axvline(
            x=one_year_line_pos,
            color="darkviolet",
            linestyle="--",
            linewidth=2.5,
            alpha=0.8,
            label="1 year boundary",
        )

    ax1.set_xlabel("Time Horizon Category", fontsize=11)
    ax1.set_ylabel("Max Explainable by Single θ (%)", fontsize=11)
    ax1.set_title(
        "Best Achievable Accuracy with Optimal θ\n(Higher = more consistent behavior)",
        fontsize=11,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45, ha="right")
    ax1.set_ylim(0, 110)
    ax1.legend(loc="lower left", fontsize=7, framealpha=0.9)

    # Add labels inside bars (simplified for readability)
    for bar, acc, n, n_conf in zip(bars, best_accuracies, n_samples, n_conflicts_list):
        text_y = bar.get_height() / 2
        # Compact label: percentage on top, conflict count below
        label = f"{acc:.0f}%"
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            text_y + 5,
            label,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white",
        )
        if n_conf > 0:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                text_y - 8,
                f"({n_conf})",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                alpha=0.9,
            )

    # Plot 2: What does "conflict" mean? Visual explanation
    ax2 = axes[1]

    # Show the breakdown of correct vs conflicting samples
    n_correct = [int(a * n / 100 + 0.5) for a, n in zip(best_accuracies, n_samples)]

    bars1 = ax2.bar(
        x,
        n_correct,
        label="Predictable (matches optimal θ)",
        color="forestgreen",
        alpha=0.8,
    )
    bars2 = ax2.bar(
        x,
        n_conflicts_list,
        bottom=n_correct,
        label="Conflicts (contradicts optimal θ)",
        color="crimson",
        alpha=0.8,
    )

    # Add 1-year separator
    if one_year_line_pos >= 0:
        ax2.axvline(
            x=one_year_line_pos,
            color="darkviolet",
            linestyle="--",
            linewidth=2.5,
            alpha=0.8,
            label="1 year boundary",
        )

    ax2.set_xlabel("Time Horizon Category", fontsize=11)
    ax2.set_ylabel("Number of Samples", fontsize=11)
    ax2.set_title(
        "Sample Breakdown: Predictable vs Conflicting\n(Conflicts = no single θ can explain)",
        fontsize=11,
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, rotation=45, ha="right")
    ax2.legend(loc="upper right", fontsize=7, framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.92])  # Leave room for suptitle and subtitle

    # Use readable filename without timestamp
    prefix = title_prefix.lower().replace(": ", "_").replace(" ", "_").strip("_")
    filename = f"{prefix}_consistency.png" if prefix else "consistency.png"
    plot_path = output_dir / filename
    plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    return plot_path


def serialize_consistency_analysis(analyses: dict[str, ConsistencyAnalysis]) -> dict:
    """Serialize consistency analyses for JSON output."""
    result = {}
    for horizon_key, analysis in analyses.items():
        n = len(analysis.constraints)
        # num_conflicts = samples that can't be correctly predicted even with optimal theta
        num_conflicts = n - int(analysis.best_accuracy * n + 0.5)
        result[horizon_key] = {
            "num_samples": n,
            "choice_stats": {
                "num_short_term_choices": analysis.num_short_term_choices,
                "num_long_term_choices": analysis.num_long_term_choices,
            },
            "is_fully_consistent": analysis.is_consistent,
            "theta_bounds": {
                "lower": analysis.theta_min
                if analysis.theta_min < float("inf")
                else None,
                "upper": analysis.theta_max
                if analysis.theta_max < float("inf")
                else None,
            },
            "best_theta": analysis.best_theta,
            # Theoretical max accuracy: what % of choices can be explained by ANY single θ
            # This is NOT model accuracy - it's the upper bound on consistency
            "consistency_upper_bound": analysis.best_accuracy,
            "num_conflicts": num_conflicts,
            "horizon_alignment": {
                "aligned": analysis.horizon_aligned,
                "total": analysis.horizon_total,
            },
        }
    return result


def generate_conflicts_analysis(
    analyses: dict[str, ConsistencyAnalysis],
    data_source: str = "unknown",
) -> dict:
    """
    Generate detailed conflict analysis for saving to JSON.

    Returns dict with summary and per-category details of conflicting samples.
    """
    result = {
        "data_source": data_source,
        "summary": {},
        "by_category": {},
    }

    total_samples = 0
    total_conflicts = 0

    for horizon_key in sorted_horizon_keys(analyses):
        analysis = analyses[horizon_key]
        n = len(analysis.constraints)
        if n == 0:
            continue

        # Find samples that conflict with optimal theta
        best_theta = analysis.best_theta
        conflicting_samples = []

        for i, c in enumerate(analysis.constraints):
            if not c.is_satisfied(best_theta):
                # This sample is "wrong" even with optimal theta
                conflicting_samples.append(
                    {
                        "index": i,
                        "choice": c.choice,
                        "reward_short": c.reward_short,
                        "time_short_years": c.time_short,
                        "reward_long": c.reward_long,
                        "time_long_years": c.time_long,
                        "time_horizon_years": c.time_horizon,
                        "theta_threshold": c.theta_threshold
                        if c.theta_threshold < float("inf")
                        else None,
                        "constraint_type": c.constraint_type,
                        "explanation": (
                            f"Choice '{c.choice}' requires θ {c.constraint_type.replace('_', ' ')} "
                            f"{c.theta_threshold:.4f}, but optimal θ={best_theta:.4f}"
                        )
                        if c.theta_threshold < float("inf")
                        else "Invalid constraint",
                    }
                )

        n_conflicts = len(conflicting_samples)
        total_samples += n
        total_conflicts += n_conflicts

        result["by_category"][horizon_key] = {
            "num_samples": n,
            "num_conflicts": n_conflicts,
            "conflict_rate": n_conflicts / n if n > 0 else 0,
            "best_theta": best_theta,
            "best_accuracy": analysis.best_accuracy,
            "conflicting_samples": conflicting_samples,
        }

    result["summary"] = {
        "total_samples": total_samples,
        "total_conflicts": total_conflicts,
        "overall_conflict_rate": total_conflicts / total_samples
        if total_samples > 0
        else 0,
        "interpretation": (
            "Conflicts are samples that cannot be correctly predicted by ANY single "
            "discount rate. They represent inherent contradictions in the LLM's choices."
        ),
    }

    return result


def generate_alignment_analysis(
    analyses: dict[str, ConsistencyAnalysis],
    data_source: str = "unknown",
) -> dict:
    """
    Generate detailed horizon alignment analysis for saving to JSON.

    Categorizes samples as:
    - aligned: choice matches expected based on proximity
    - ambiguous: time_short < time_horizon < time_long (horizon between options)
    - unaligned: choice doesn't match expected AND horizon is not between options

    Returns dict with summary and per-category details of alignment.
    """
    result = {
        "data_source": data_source,
        "summary": {},
        "by_category": {},
    }

    total_aligned = 0
    total_ambiguous = 0
    total_unaligned = 0
    total_with_horizon = 0

    for horizon_key in sorted_horizon_keys(analyses):
        analysis = analyses[horizon_key]
        n = len(analysis.constraints)
        if n == 0:
            continue

        aligned_count = 0
        ambiguous_count = 0
        with_horizon_count = 0

        # Collect samples by category
        unaligned_samples = []
        ambiguous_samples = []

        for i, c in enumerate(analysis.constraints):
            if c.time_horizon is None:
                continue

            with_horizon_count += 1

            alignment = c.horizon_alignment()
            if alignment is None:
                continue

            expected, is_aligned, alignment_type = alignment

            sample_details = {
                "index": i,
                "time_horizon_years": c.time_horizon,
                "time_short_years": c.time_short,
                "time_long_years": c.time_long,
                "actual_choice": c.choice,
                "reward_short": c.reward_short,
                "reward_long": c.reward_long,
                "alignment_type": alignment_type,
            }

            if expected == "ambiguous":
                # Horizon is between options - no clear expected choice
                ambiguous_count += 1
                sample_details["note"] = (
                    "Horizon between options; no clear expected choice"
                )
                ambiguous_samples.append(sample_details)
            else:
                # Horizon is outside the options - clear expected choice
                sample_details["expected_choice"] = expected
                if is_aligned:
                    aligned_count += 1
                else:
                    unaligned_samples.append(sample_details)

        total_aligned += aligned_count
        total_ambiguous += ambiguous_count
        total_unaligned += len(unaligned_samples)
        total_with_horizon += with_horizon_count

        result["by_category"][horizon_key] = {
            "num_samples": n,
            "num_with_horizon": with_horizon_count,
            "num_aligned": aligned_count,
            "num_ambiguous": ambiguous_count,
            "num_unaligned": len(unaligned_samples),
            "alignment_rate": aligned_count / with_horizon_count
            if with_horizon_count > 0
            else None,
            "unaligned_samples": unaligned_samples,
            "ambiguous_samples": ambiguous_samples,
        }

    # Alignment rate only counts non-ambiguous samples
    non_ambiguous = total_with_horizon - total_ambiguous
    alignment_rate = total_aligned / non_ambiguous if non_ambiguous > 0 else None

    if alignment_rate is not None:
        alignment_pct = f"{alignment_rate:.1%}"
        high_or_low = "High" if alignment_rate > 0.7 else "Low"
        considers_or_ignores = "considers" if alignment_rate > 0.7 else "ignores"
        interpretation = (
            "Alignment measures whether choices respect time horizon boundaries. "
            "Expected: horizon < short_time → short_term, horizon > long_time → long_term. "
            "'Ambiguous' samples have short_time <= horizon <= long_time "
            "(horizon between options, any choice valid). "
            f"{high_or_low} alignment ({alignment_pct} of {non_ambiguous} non-ambiguous) "
            f"suggests the LLM {considers_or_ignores} time horizon in decisions."
        )
    else:
        interpretation = "No non-ambiguous samples with time_horizon to analyze."

    result["summary"] = {
        "total_with_horizon": total_with_horizon,
        "total_non_ambiguous": non_ambiguous,
        "total_aligned": total_aligned,
        "total_ambiguous": total_ambiguous,
        "total_unaligned": total_unaligned,
        "overall_alignment_rate": alignment_rate,
        "interpretation": interpretation,
    }

    return result


def create_conflicts_plot(
    analyses: dict[str, ConsistencyAnalysis],
    output_dir: Path,
    title_prefix: str = "",
) -> Optional[Path]:
    """Create visualization of conflicts by category."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    # Custom ordering: short-term (<1yr), then "No Horizon Given", then long-term (>1yr)
    short_term_items = []
    long_term_items = []
    no_horizon_item = None

    for horizon_key in sorted_horizon_keys(analyses):
        analysis = analyses[horizon_key]
        n = len(analysis.constraints)
        if n == 0:
            continue

        if horizon_key == "no_horizon":
            no_horizon_item = (horizon_key, analysis)
        else:
            sort_key = horizon_sort_key(horizon_key)
            if sort_key[0] < 1.0:
                short_term_items.append((horizon_key, analysis))
            else:
                long_term_items.append((horizon_key, analysis))

    # Build ordered list: no_horizon first (at origin), then short-term, then long-term
    ordered_items = []
    if no_horizon_item:
        ordered_items.append(no_horizon_item)
    ordered_items.extend(short_term_items)
    ordered_items.extend(long_term_items)

    if not ordered_items:
        return None

    # Collect data in order
    categories = []
    n_samples = []
    n_conflicts_list = []
    best_thetas = []
    # 1-year line after no_horizon + short-term items
    no_horizon_offset = 1 if no_horizon_item else 0
    one_year_line_pos = no_horizon_offset + len(short_term_items) - 0.5

    for horizon_key, analysis in ordered_items:
        categories.append(format_horizon_label(horizon_key))
        n = len(analysis.constraints)
        n_samples.append(n)
        n_correct = int(analysis.best_accuracy * n + 0.5)
        n_conflicts_list.append(n - n_correct)
        best_thetas.append(analysis.best_theta)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    title = f"{title_prefix}Conflict Analysis" if title_prefix else "Conflict Analysis"
    fig.suptitle(title, fontsize=14, fontweight="bold", fontfamily="DejaVu Sans")

    # Plot 1: Stacked bar - correct vs conflicts
    ax1 = axes[0]
    x = np.arange(len(categories))
    n_correct = [s - c for s, c in zip(n_samples, n_conflicts_list)]

    bars1 = ax1.bar(
        x, n_correct, label="Correctly predictable", color="forestgreen", alpha=0.8
    )
    bars2 = ax1.bar(
        x,
        n_conflicts_list,
        bottom=n_correct,
        label="Conflicts (unpredictable)",
        color="crimson",
        alpha=0.8,
    )

    # Add 1-year separator line
    if one_year_line_pos >= 0:
        ax1.axvline(
            x=one_year_line_pos,
            color="darkviolet",
            linestyle="--",
            linewidth=2.5,
            alpha=0.8,
            label="1 year boundary",
        )

    ax1.set_xlabel("Time Horizon Category", fontsize=11)
    ax1.set_ylabel("Number of Samples", fontsize=11)
    ax1.set_title("Predictable vs Conflicting Samples", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45, ha="right")
    ax1.legend(loc="lower left", fontsize=7, framealpha=0.9)

    # Add percentage labels
    for i, (correct, conflict, total) in enumerate(
        zip(n_correct, n_conflicts_list, n_samples)
    ):
        if conflict > 0:
            pct = conflict / total * 100
            ax1.text(
                i,
                total + 1,
                f"{pct:.0f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                color="crimson",
            )

    # Plot 2: Optimal theta by category
    ax2 = axes[1]
    colors = ["crimson" if c > 0 else "forestgreen" for c in n_conflicts_list]
    bars = ax2.bar(x, best_thetas, color=colors, alpha=0.8)

    # Add 1-year separator line
    if one_year_line_pos >= 0:
        ax2.axvline(
            x=one_year_line_pos,
            color="darkviolet",
            linestyle="--",
            linewidth=2.5,
            alpha=0.8,
            label="1 year boundary",
        )

    ax2.set_xlabel("Time Horizon Category", fontsize=11)
    ax2.set_ylabel(
        "Optimal θ (discount rate)\n↑ Higher = more impatient (prefer short-term)",
        fontsize=10,
    )
    ax2.set_title("Optimal Discount Rate by Category", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, rotation=45, ha="right")
    ax2.legend(loc="lower left", fontsize=7, framealpha=0.9)

    # Add value labels inside tall bars, above short ones
    for bar, theta in zip(bars, best_thetas):
        height = bar.get_height()
        if height > 0.5:
            # Place inside bar
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                height * 0.6,
                f"{theta:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="white",
                rotation=90,
            )
        else:
            # Place above bar
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.02,
                f"{theta:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()

    # Use readable filename without timestamp
    prefix = title_prefix.lower().replace(": ", "_").replace(" ", "_").strip("_")
    filename = f"{prefix}_conflicts.png" if prefix else "conflicts.png"
    plot_path = output_dir / filename
    plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    return plot_path


def create_conflict_distribution_plot(
    analyses: dict[str, ConsistencyAnalysis],
    output_dir: Path,
    title_prefix: str = "",
) -> Optional[Path]:
    """
    Create visualization of conflict distribution by horizon and choice type.

    Shows:
    - Left: Conflict rate by horizon category
    - Right: Conflicts broken down by choice type (short_term vs long_term)

    Key insight: At long horizons, majority chooses long_term, so optimal θ is LOW.
    LOW θ predicts long_term for all samples. Conflicts = minority who chose short_term.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    # Collect conflict data
    horizon_data = []
    for horizon_key in sorted_horizon_keys(analyses):
        analysis = analyses[horizon_key]
        n = len(analysis.constraints)
        n_conflicts = n - int(analysis.best_accuracy * n + 0.5)

        # Count conflicts by choice type
        conflicts_short = 0
        conflicts_long = 0
        for c in analysis.constraints:
            is_conflict = not analysis.is_sample_consistent(c)
            if is_conflict:
                if c.choice == "short_term":
                    conflicts_short += 1
                else:
                    conflicts_long += 1

        horizon_data.append(
            {
                "horizon": horizon_key,
                "label": format_horizon_label(horizon_key),
                "n_samples": n,
                "n_conflicts": n_conflicts,
                "conflicts_short_term": conflicts_short,
                "conflicts_long_term": conflicts_long,
                "num_short_term_choices": analysis.num_short_term_choices,
                "num_long_term_choices": analysis.num_long_term_choices,
                "best_theta": analysis.best_theta,
            }
        )

    if not horizon_data:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    title = (
        f"{title_prefix}Conflict Distribution Analysis"
        if title_prefix
        else "Conflict Distribution Analysis"
    )
    fig.suptitle(title, fontsize=14, fontweight="bold", fontfamily="DejaVu Sans")

    x = np.arange(len(horizon_data))
    labels = [d["label"] for d in horizon_data]

    # Plot 1: Conflict rate by horizon
    ax1 = axes[0]
    conflict_rates = [d["n_conflicts"] / d["n_samples"] * 100 for d in horizon_data]
    colors = [
        "crimson" if r > 20 else "orange" if r > 10 else "forestgreen"
        for r in conflict_rates
    ]
    bars = ax1.bar(x, conflict_rates, color=colors, alpha=0.8)

    ax1.set_xlabel("Time Horizon Category", fontsize=11)
    ax1.set_ylabel("Conflict Rate (%)", fontsize=11)
    ax1.set_title(
        "Conflict Rate by Horizon\n(Higher = more contradictory choices)", fontsize=11
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.axhline(
        y=20,
        color="red",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label="High conflict (20%)",
    )
    ax1.axhline(
        y=10,
        color="orange",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label="Moderate (10%)",
    )
    ax1.legend(loc="upper left", fontsize=7, framealpha=0.9)

    # Add value labels - place inside bars if tall enough, else above
    for bar, rate, d in zip(bars, conflict_rates, horizon_data):
        label_text = f"{rate:.0f}%\n({d['n_conflicts']}/{d['n_samples']})"
        if rate >= 15:
            # Place inside bar
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() - 3,
                label_text,
                ha="center",
                va="top",
                fontsize=7,
                color="white",
                fontweight="bold",
            )
        else:
            # Place above bar
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                label_text,
                ha="center",
                va="bottom",
                fontsize=7,
            )

    # Plot 2: Conflicts by choice type (stacked bar) with theta annotations
    ax2 = axes[1]
    conflicts_short = [d["conflicts_short_term"] for d in horizon_data]
    conflicts_long = [d["conflicts_long_term"] for d in horizon_data]

    bars1 = ax2.bar(
        x,
        conflicts_short,
        label="Conflicts: chose short_term",
        color="coral",
        alpha=0.8,
    )
    bars2 = ax2.bar(
        x,
        conflicts_long,
        bottom=conflicts_short,
        label="Conflicts: chose long_term",
        color="steelblue",
        alpha=0.8,
    )

    ax2.set_xlabel("Time Horizon Category", fontsize=11)
    ax2.set_ylabel("Number of Conflicts", fontsize=11)
    ax2.set_title(
        "Conflicts by Choice Type\n(What did conflicting samples choose?)", fontsize=11
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right")
    ax2.legend(loc="upper left", fontsize=7, framealpha=0.9)

    # Add total labels inside bars
    for i, d in enumerate(horizon_data):
        total = d["n_conflicts"]
        if total > 0:
            ax2.text(
                i,
                total / 2,
                f"{total}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white",
            )

    # Use rect to leave room for title/subtitle at top
    plt.tight_layout(rect=[0, 0.02, 1, 0.88])

    # Save with readable filename
    prefix = title_prefix.lower().replace(": ", "_").replace(" ", "_").strip("_")
    filename = (
        f"{prefix}_conflict_distribution.png" if prefix else "conflict_distribution.png"
    )
    plot_path = output_dir / filename
    plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    return plot_path


def create_ambiguous_analysis_plot(
    analyses: dict[str, ConsistencyAnalysis],
    output_dir: Path,
    title_prefix: str = "",
) -> Optional[Path]:
    """
    Create visualization analyzing ambiguous samples (horizon between option times).

    These are samples where short_term_time <= time_horizon <= long_term_time,
    meaning the horizon doesn't clearly favor either option.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    # Collect ambiguous samples across all horizons
    ambiguous_samples = []
    for horizon_key, analysis in analyses.items():
        for c in analysis.constraints:
            if c.time_horizon is None:
                continue

            alignment = c.horizon_alignment()
            if alignment and alignment[0] == "ambiguous":
                _, _, alignment_type = alignment
                # Compute how close to short vs long option
                range_span = c.time_long - c.time_short
                if range_span > 0:
                    # 0 = at short_term_time, 1 = at long_term_time
                    position_in_range = (c.time_horizon - c.time_short) / range_span
                else:
                    position_in_range = 0.5

                ambiguous_samples.append(
                    {
                        "time_horizon": c.time_horizon,
                        "time_short": c.time_short,
                        "time_long": c.time_long,
                        "choice": c.choice,
                        "position_in_range": position_in_range,
                        "horizon_key": horizon_key,
                        "reward_short": c.reward_short,
                        "reward_long": c.reward_long,
                    }
                )

    if not ambiguous_samples:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    title = (
        f"{title_prefix}Ambiguous Samples Analysis"
        if title_prefix
        else "Ambiguous Samples Analysis"
    )
    fig.suptitle(title, fontsize=14, fontweight="bold", fontfamily="DejaVu Sans")
    fig.text(
        0.5,
        0.92,
        f"Samples where time_horizon is BETWEEN short_term_time and long_term_time (n={len(ambiguous_samples)})",
        ha="center",
        fontsize=10,
        style="italic",
        color="gray",
    )

    # Plot 1: Choice distribution for ambiguous samples
    ax1 = axes[0]
    n_short = sum(1 for s in ambiguous_samples if s["choice"] == "short_term")
    n_long = len(ambiguous_samples) - n_short

    bars = ax1.bar(
        ["Short-term\nChoice", "Long-term\nChoice"],
        [n_short, n_long],
        color=["coral", "steelblue"],
        alpha=0.8,
    )
    ax1.set_ylabel("Number of Samples", fontsize=11)
    ax1.set_title("What do ambiguous samples choose?", fontsize=12)

    # Add percentages
    total = len(ambiguous_samples)
    for bar, n in zip(bars, [n_short, n_long]):
        pct = n / total * 100
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{n}\n({pct:.0f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # Add 50% line
    ax1.axhline(
        y=total / 2, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="50%"
    )

    # Plot 2: Position in range vs choice
    ax2 = axes[1]

    # Separate by choice
    short_choices = [s for s in ambiguous_samples if s["choice"] == "short_term"]
    long_choices = [s for s in ambiguous_samples if s["choice"] == "long_term"]

    # Create histogram of positions
    bins = np.linspace(0, 1, 11)  # 10 bins from 0 to 1

    short_positions = [s["position_in_range"] for s in short_choices]
    long_positions = [s["position_in_range"] for s in long_choices]

    ax2.hist(
        [short_positions, long_positions],
        bins=bins,
        label=["Chose short", "Chose long"],
        color=["coral", "steelblue"],
        alpha=0.7,
        stacked=True,
    )

    ax2.set_xlabel("Position in Range\n(0=at short_time, 1=at long_time)", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Where in the range is the horizon?", fontsize=12)
    ax2.legend(loc="upper center", fontsize=9)
    ax2.axvline(x=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    # Plot 3: Scatter of position vs reward ratio
    ax3 = axes[2]

    # Compute reward ratio (long/short)
    for s in ambiguous_samples:
        s["reward_ratio"] = (
            s["reward_long"] / s["reward_short"] if s["reward_short"] > 0 else 1
        )

    short_x = [s["position_in_range"] for s in short_choices]
    short_y = [s["reward_ratio"] for s in short_choices]
    long_x = [s["position_in_range"] for s in long_choices]
    long_y = [s["reward_ratio"] for s in long_choices]

    if short_x:
        ax3.scatter(
            short_x,
            short_y,
            c="coral",
            marker="o",
            alpha=0.7,
            s=60,
            label=f"Chose short (n={len(short_choices)})",
        )
    if long_x:
        ax3.scatter(
            long_x,
            long_y,
            c="steelblue",
            marker="s",
            alpha=0.7,
            s=60,
            label=f"Chose long (n={len(long_choices)})",
        )

    ax3.set_xlabel("Position in Range\n(0=at short_time, 1=at long_time)", fontsize=11)
    ax3.set_ylabel("Reward Ratio (long/short)", fontsize=11)
    ax3.set_title("Position vs Reward Ratio", fontsize=12)
    ax3.legend(loc="upper left", fontsize=9)
    ax3.axvline(x=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax3.axhline(y=1, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.90])

    # Save
    prefix = title_prefix.lower().replace(": ", "_").replace(" ", "_").strip("_")
    filename = (
        f"{prefix}_ambiguous_analysis.png" if prefix else "ambiguous_analysis.png"
    )
    plot_path = output_dir / filename
    plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    return plot_path


def create_alignment_plot(
    analyses: dict[str, ConsistencyAnalysis],
    output_dir: Path,
    title_prefix: str = "",
) -> Optional[Path]:
    """Create visualization of horizon alignment."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    # Collect alignment data across all samples
    all_samples = []
    for horizon_key, analysis in analyses.items():
        for c in analysis.constraints:
            if c.time_horizon is not None:
                alignment = c.horizon_alignment()
                if alignment:
                    expected, is_aligned, alignment_type = alignment
                    all_samples.append(
                        {
                            "time_horizon": c.time_horizon,
                            "time_short": c.time_short,
                            "time_long": c.time_long,
                            "choice": c.choice,
                            "expected": expected,
                            "is_aligned": is_aligned,
                            "alignment_type": alignment_type,
                            "horizon_key": horizon_key,
                        }
                    )

    if not all_samples:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    title = (
        f"{title_prefix}Time Horizon Alignment Analysis"
        if title_prefix
        else "Time Horizon Alignment Analysis"
    )
    fig.suptitle(
        title, fontsize=14, fontweight="bold", y=0.98, fontfamily="DejaVu Sans"
    )

    # Plot 1: Alignment rate by category
    ax1 = axes[0]

    # Custom ordering: short-term (<1yr), then "No Horizon Given", then long-term (>1yr)
    short_term_keys = []
    long_term_keys = []
    no_horizon_data = None

    for horizon_key in sorted_horizon_keys(analyses):
        analysis = analyses[horizon_key]
        if analysis.horizon_total == 0:
            continue

        if horizon_key == "no_horizon":
            no_horizon_data = (horizon_key, analysis)
        else:
            # Parse time in years
            sort_key = horizon_sort_key(horizon_key)
            if sort_key[0] < 1.0:
                short_term_keys.append((horizon_key, analysis))
            else:
                long_term_keys.append((horizon_key, analysis))

    # Build ordered list: no_horizon first (at origin), then short-term, then long-term
    ordered_items = []
    if no_horizon_data:
        ordered_items.append(no_horizon_data)
    ordered_items.extend(short_term_keys)
    ordered_items.extend(long_term_keys)

    categories = []
    aligned_counts = []
    total_counts = []
    # 1-year line after no_horizon + short-term items
    no_horizon_offset = 1 if no_horizon_data else 0
    one_year_line_pos = no_horizon_offset + len(short_term_keys) - 0.5

    for horizon_key, analysis in ordered_items:
        categories.append(format_horizon_label(horizon_key))
        aligned_counts.append(analysis.horizon_aligned)
        total_counts.append(analysis.horizon_total)

    if categories:
        x = np.arange(len(categories))
        alignment_rates = [
            a / t * 100 if t > 0 else 0 for a, t in zip(aligned_counts, total_counts)
        ]

        colors = [
            "forestgreen" if r >= 70 else "orange" if r >= 50 else "crimson"
            for r in alignment_rates
        ]
        bars = ax1.bar(x, alignment_rates, color=colors, alpha=0.8)

        ax1.set_xlabel("Time Horizon Category", fontsize=11)
        ax1.set_ylabel("Alignment Rate (%)", fontsize=11)
        ax1.set_title("Alignment Rate by Horizon", fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories, rotation=45, ha="right")
        ax1.set_ylim(0, 105)
        ax1.axhline(
            y=50, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Random"
        )
        ax1.axhline(
            y=70,
            color="green",
            linestyle="--",
            linewidth=1,
            alpha=0.5,
            label="High (70%)",
        )

        # Add 1-year separator line
        if one_year_line_pos >= 0:
            ax1.axvline(
                x=one_year_line_pos,
                color="darkviolet",
                linestyle="--",
                linewidth=2.5,
                alpha=0.8,
                label="1yr boundary",
            )

        ax1.legend(loc="upper left", fontsize=7, framealpha=0.9)

        # Add labels INSIDE the bars
        for bar, rate, n in zip(bars, alignment_rates, total_counts):
            # Put text inside bar, centered vertically
            text_y = bar.get_height() / 2
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                text_y,
                f"{rate:.0f}%\n(n={n})",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white",
            )

    # Plot 2: Stacked bar chart of misaligned samples by horizon
    ax2 = axes[1]

    # Count misaligned samples by horizon and choice type
    misaligned_by_horizon: dict[str, dict[str, int]] = {}
    for s in all_samples:
        if not s["is_aligned"]:
            horizon_key = s["horizon_key"]
            if horizon_key not in misaligned_by_horizon:
                misaligned_by_horizon[horizon_key] = {"short_term": 0, "long_term": 0}
            misaligned_by_horizon[horizon_key][s["choice"]] += 1

    # Use same ordering as plot 1
    misaligned_categories = []
    misaligned_short = []
    misaligned_long = []

    for horizon_key, analysis in ordered_items:
        if horizon_key in misaligned_by_horizon:
            misaligned_categories.append(format_horizon_label(horizon_key))
            misaligned_short.append(misaligned_by_horizon[horizon_key]["short_term"])
            misaligned_long.append(misaligned_by_horizon[horizon_key]["long_term"])

    if misaligned_categories:
        x2 = np.arange(len(misaligned_categories))

        # Stacked bar: short_term on bottom, long_term on top
        bars_short = ax2.bar(
            x2,
            misaligned_short,
            label="Chose short-term",
            color="coral",
            alpha=0.8,
        )
        bars_long = ax2.bar(
            x2,
            misaligned_long,
            bottom=misaligned_short,
            label="Chose long-term",
            color="steelblue",
            alpha=0.8,
        )

        ax2.set_xlabel("Time Horizon Category", fontsize=11)
        ax2.set_ylabel("Number of Misaligned Samples", fontsize=11)
        ax2.set_title("Misaligned Samples by Horizon", fontsize=12)
        ax2.set_xticks(x2)
        ax2.set_xticklabels(misaligned_categories, rotation=45, ha="right")
        ax2.legend(loc="upper right", fontsize=8, framealpha=0.9)

        # Add count labels inside bars
        for i, (n_short, n_long) in enumerate(zip(misaligned_short, misaligned_long)):
            total = n_short + n_long
            if total > 0:
                # Label for short-term portion (bottom)
                if n_short > 0:
                    ax2.text(
                        i,
                        n_short / 2,
                        str(n_short),
                        ha="center",
                        va="center",
                        fontsize=9,
                        fontweight="bold",
                        color="white",
                    )
                # Label for long-term portion (top)
                if n_long > 0:
                    ax2.text(
                        i,
                        n_short + n_long / 2,
                        str(n_long),
                        ha="center",
                        va="center",
                        fontsize=9,
                        fontweight="bold",
                        color="white",
                    )
    else:
        ax2.text(
            0.5,
            0.5,
            "No misaligned samples",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=12,
        )
        ax2.set_title("Misaligned Samples by Horizon", fontsize=12)

    plt.tight_layout()

    # Use readable filename without timestamp
    prefix = title_prefix.lower().replace(": ", "_").replace(" ", "_").strip("_")
    filename = f"{prefix}_alignment.png" if prefix else "alignment.png"
    plot_path = output_dir / filename
    plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    return plot_path


# =============================================================================
# Deep Conflict Analysis
# =============================================================================


@dataclass
class ConflictAnalysisResult:
    """Result of deep conflict analysis."""

    # Basic counts
    total_samples: int
    n_consistent: int  # Samples that form a consistent set (can be explained by some θ)
    n_conflicting: int  # Samples that conflict with the majority

    # Conflict inter-consistency: are conflicts consistent among themselves?
    conflicts_internally_consistent: bool
    conflicts_best_theta: float
    conflicts_best_accuracy: float

    # Accuracy comparison
    all_data_accuracy: float  # Best accuracy with all samples
    consistent_only_accuracy: float  # Accuracy if we remove conflicts (should be 100%)
    conflicts_only_accuracy: float  # Accuracy among conflicts only

    # Cluster info for ALL samples (groups that are mutually consistent)
    n_clusters: int
    cluster_sizes: list[int]
    cluster_thetas: list[float]

    # Cluster info for CONFLICTS only (clusters within the conflicting samples)
    conflict_n_clusters: int
    conflict_cluster_sizes: list[int]
    conflict_cluster_thetas: list[float]


def analyze_conflicts_deeply(
    constraints: list[ThetaConstraint],
) -> ConflictAnalysisResult:
    """
    Analyze conflicts in depth: are they consistent among themselves? Do they form clusters?

    This function examines whether the conflicting samples (those that can't be predicted
    by the optimal θ for all data) are themselves consistent - i.e., could they be
    explained by a DIFFERENT θ?
    """
    if not constraints:
        return ConflictAnalysisResult(
            total_samples=0,
            n_consistent=0,
            n_conflicting=0,
            conflicts_internally_consistent=True,
            conflicts_best_theta=0.0,
            conflicts_best_accuracy=0.0,
            all_data_accuracy=0.0,
            consistent_only_accuracy=0.0,
            conflicts_only_accuracy=0.0,
            n_clusters=0,
            cluster_sizes=[],
            cluster_thetas=[],
            conflict_n_clusters=0,
            conflict_cluster_sizes=[],
            conflict_cluster_thetas=[],
        )

    # First, find best θ for all data
    analysis_all = ConsistencyAnalysis.analyze(constraints)
    best_theta_all = analysis_all.best_theta
    all_data_accuracy = analysis_all.best_accuracy

    # Separate into consistent and conflicting samples
    consistent_samples = []
    conflicting_samples = []

    for c in constraints:
        if c.is_satisfied(best_theta_all):
            consistent_samples.append(c)
        else:
            conflicting_samples.append(c)

    n_consistent = len(consistent_samples)
    n_conflicting = len(conflicting_samples)

    # Analyze consistency of the consistent samples (should be 100%)
    if consistent_samples:
        analysis_consistent = ConsistencyAnalysis.analyze(consistent_samples)
        consistent_only_accuracy = analysis_consistent.best_accuracy
    else:
        consistent_only_accuracy = 1.0

    # Analyze consistency of the conflicting samples
    # Key question: Are conflicts consistent among themselves?
    if conflicting_samples:
        analysis_conflicts = ConsistencyAnalysis.analyze(conflicting_samples)
        conflicts_internally_consistent = analysis_conflicts.is_consistent
        conflicts_best_theta = analysis_conflicts.best_theta
        conflicts_best_accuracy = analysis_conflicts.best_accuracy
        conflicts_only_accuracy = analysis_conflicts.best_accuracy
    else:
        conflicts_internally_consistent = True
        conflicts_best_theta = 0.0
        conflicts_best_accuracy = 1.0
        conflicts_only_accuracy = 1.0

    # Find clusters for ALL samples using greedy approach
    # Each cluster is a maximal set of mutually consistent samples
    clusters = find_consistent_clusters(constraints)
    n_clusters = len(clusters)
    cluster_sizes = [len(c) for c in clusters]

    # Find best θ for each cluster
    cluster_thetas = []
    for cluster in clusters:
        if cluster:
            cluster_analysis = ConsistencyAnalysis.analyze(cluster)
            cluster_thetas.append(cluster_analysis.best_theta)

    # Find clusters WITHIN conflicts only
    conflict_n_clusters = 0
    conflict_cluster_sizes = []
    conflict_cluster_thetas = []

    if conflicting_samples:
        conflict_clusters = find_consistent_clusters(conflicting_samples)
        conflict_n_clusters = len(conflict_clusters)
        conflict_cluster_sizes = [len(c) for c in conflict_clusters]

        for cluster in conflict_clusters:
            if cluster:
                cluster_analysis = ConsistencyAnalysis.analyze(cluster)
                conflict_cluster_thetas.append(cluster_analysis.best_theta)

    return ConflictAnalysisResult(
        total_samples=len(constraints),
        n_consistent=n_consistent,
        n_conflicting=n_conflicting,
        conflicts_internally_consistent=conflicts_internally_consistent,
        conflicts_best_theta=conflicts_best_theta,
        conflicts_best_accuracy=conflicts_best_accuracy,
        all_data_accuracy=all_data_accuracy,
        consistent_only_accuracy=consistent_only_accuracy,
        conflicts_only_accuracy=conflicts_only_accuracy,
        n_clusters=n_clusters,
        cluster_sizes=cluster_sizes,
        cluster_thetas=cluster_thetas,
        conflict_n_clusters=conflict_n_clusters,
        conflict_cluster_sizes=conflict_cluster_sizes,
        conflict_cluster_thetas=conflict_cluster_thetas,
    )


def find_consistent_clusters(
    constraints: list[ThetaConstraint],
) -> list[list[ThetaConstraint]]:
    """
    Find clusters of mutually consistent samples using greedy clustering.

    Each cluster is a set of samples that can all be explained by a single θ.
    Greedy approach: Start with largest possible cluster, remove those samples,
    repeat until all samples are assigned.
    """
    if not constraints:
        return []

    remaining = list(constraints)
    clusters = []

    while remaining:
        # Find the largest cluster starting from remaining samples
        best_cluster = []
        best_theta = 0.0

        # Try each possible θ threshold as a potential cluster center
        thresholds = sorted(
            set(
                c.theta_threshold for c in remaining if c.theta_threshold < float("inf")
            )
        )

        # Also try just above and below each threshold
        test_thetas = [0.001]
        for t in thresholds:
            test_thetas.extend([t - 0.0001, t + 0.0001])

        for theta in test_thetas:
            if theta <= 0:
                continue
            cluster = [c for c in remaining if c.is_satisfied(theta)]
            if len(cluster) > len(best_cluster):
                best_cluster = cluster
                best_theta = theta

        if not best_cluster:
            # No more consistent clusters possible, each remaining sample is its own cluster
            for c in remaining:
                clusters.append([c])
            break

        clusters.append(best_cluster)

        # Remove clustered samples from remaining
        clustered_set = set(id(c) for c in best_cluster)
        remaining = [c for c in remaining if id(c) not in clustered_set]

    return clusters


def print_conflict_analysis(
    analysis: ConflictAnalysisResult,
    title: str = "CONFLICT ANALYSIS",
) -> None:
    """Print detailed conflict analysis."""
    print()
    print("=" * 80)
    print(f"  {title}: Are Conflicts Consistent Among Themselves?")
    print("=" * 80)
    print()
    print(f"  Total samples: {analysis.total_samples}")
    print(
        f"  Consistent with majority: {analysis.n_consistent} ({analysis.n_consistent / analysis.total_samples:.1%})"
    )
    print(
        f"  Conflicting: {analysis.n_conflicting} ({analysis.n_conflicting / analysis.total_samples:.1%})"
    )
    print()

    if analysis.n_conflicting > 0:
        print("  Conflict Inter-Consistency:")
        if analysis.conflicts_internally_consistent:
            print("    ✓ Conflicts ARE consistent among themselves!")
            print(
                f"      They can all be explained by θ={analysis.conflicts_best_theta:.4f}"
            )
            print(
                f"      (Accuracy among conflicts: {analysis.conflicts_best_accuracy:.0%})"
            )
            print()
            print("    Interpretation: The LLM may have TWO distinct discount rates:")
            print("      - Majority behavior: optimal θ for consistent samples")
            print(
                f"      - Minority behavior: θ={analysis.conflicts_best_theta:.4f} for conflicts"
            )
        else:
            print("    ✗ Conflicts are NOT consistent among themselves")
            print(
                f"      Best accuracy among conflicts: {analysis.conflicts_best_accuracy:.0%}"
            )
            print()
            print(
                "    Interpretation: Conflicts represent truly inconsistent/random choices"
            )
    else:
        print("  No conflicts - all samples are consistent!")

    print()
    print("  Accuracy Comparison:")
    print(f"    All data:        {analysis.all_data_accuracy:.1%}")
    print(
        f"    Consistent only: {analysis.consistent_only_accuracy:.1%} (if conflicts removed)"
    )
    if analysis.n_conflicting > 0:
        print(f"    Conflicts only:  {analysis.conflicts_only_accuracy:.1%}")
    print()

    if analysis.n_clusters > 1:
        print(f"  Cluster Analysis ({analysis.n_clusters} clusters found):")
        for i, (size, theta) in enumerate(
            zip(analysis.cluster_sizes, analysis.cluster_thetas)
        ):
            print(f"    Cluster {i + 1}: {size} samples, θ={theta:.4f}")
        print()
        if analysis.n_clusters == 2:
            print("    Two clusters suggest a bimodal discount rate distribution")
        elif analysis.n_clusters > 2:
            print("    Multiple clusters suggest complex/context-dependent discounting")
    print()


def generate_deep_conflict_analysis(
    analyses: dict[str, ConsistencyAnalysis],
    data_source: str = "unknown",
) -> dict:
    """
    Generate deep conflict analysis for JSON output.

    Analyzes whether conflicts are consistent among themselves and detects clusters.
    """
    result = {
        "data_source": data_source,
        "summary": {},
        "by_category": {},
    }

    total_samples = 0
    total_consistent = 0
    total_conflicting = 0
    all_clusters = 0

    for horizon_key in sorted_horizon_keys(analyses):
        analysis = analyses[horizon_key]
        if not analysis.constraints:
            continue

        deep_analysis = analyze_conflicts_deeply(analysis.constraints)

        total_samples += deep_analysis.total_samples
        total_consistent += deep_analysis.n_consistent
        total_conflicting += deep_analysis.n_conflicting
        all_clusters += deep_analysis.n_clusters

        category_result = {
            "total_samples": deep_analysis.total_samples,
            "n_consistent": deep_analysis.n_consistent,
            "n_conflicting": deep_analysis.n_conflicting,
            "conflicts_internally_consistent": deep_analysis.conflicts_internally_consistent,
            "conflicts_best_theta": deep_analysis.conflicts_best_theta,
            "conflicts_best_accuracy": deep_analysis.conflicts_best_accuracy,
            "all_data_accuracy": deep_analysis.all_data_accuracy,
            "consistent_only_accuracy": deep_analysis.consistent_only_accuracy,
            "conflicts_only_accuracy": deep_analysis.conflicts_only_accuracy,
            # Clusters for all samples
            "n_clusters": deep_analysis.n_clusters,
            "cluster_sizes": deep_analysis.cluster_sizes,
            "cluster_thetas": deep_analysis.cluster_thetas,
            # Clusters within conflicts only
            "conflict_clusters": {
                "n_clusters": deep_analysis.conflict_n_clusters,
                "cluster_sizes": deep_analysis.conflict_cluster_sizes,
                "cluster_thetas": deep_analysis.conflict_cluster_thetas,
            },
        }

        # Add extra details when conflicts are internally consistent
        if (
            deep_analysis.conflicts_internally_consistent
            and deep_analysis.n_conflicting > 0
        ):
            category_result["conflict_interpretation"] = (
                f"All {deep_analysis.n_conflicting} conflicting samples can be explained by "
                f"θ={deep_analysis.conflicts_best_theta:.4f}. This suggests bimodal behavior: "
                f"the LLM uses θ≈{deep_analysis.cluster_thetas[0]:.4f} for most choices but "
                f"θ≈{deep_analysis.conflicts_best_theta:.4f} for the conflicting subset."
            )

        result["by_category"][horizon_key] = category_result

    result["summary"] = {
        "total_samples": total_samples,
        "total_consistent": total_consistent,
        "total_conflicting": total_conflicting,
        "conflict_rate": total_conflicting / total_samples if total_samples > 0 else 0,
        "total_clusters": all_clusters,
        "interpretation": (
            "Deep conflict analysis examines whether conflicting samples form a "
            "consistent subset (suggesting bimodal behavior) or are truly random. "
            "'conflict_clusters' shows clusters WITHIN conflicting samples only. "
            "If conflict_clusters.n_clusters=1, all conflicts share a single θ (bimodal). "
            "If >1, conflicts themselves are inconsistent (truly noisy choices)."
        ),
    }

    return result
