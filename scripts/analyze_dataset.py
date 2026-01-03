#!/usr/bin/env python
"""
Analyze dataset structure and produce comprehensive visualizations.

Produces graphs showing:
- Distribution of rewards (short-term vs long-term)
- Distribution of delays (short-term vs long-term)
- Time horizon distribution
- Reward ratios and delay ratios
- Joint distributions and heatmaps
- Balance analysis across different dimensions

Usage:
    python scripts/analyze_dataset.py --dataset-id c35e217b8473d84a41b79c38e5b3c059
    python scripts/analyze_dataset.py --dataset-id c35e217b8473d84a41b79c38e5b3c059 --show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.io import ensure_dir, load_json

# =============================================================================
# Constants
# =============================================================================

DATASETS_DIR = PROJECT_ROOT / "out" / "datasets"
OUTPUT_DIR = PROJECT_ROOT / "out" / "dataset_analysis"

# =============================================================================
# Data Loading
# =============================================================================


def find_dataset_by_id(dataset_id: str) -> Path | None:
    """Find dataset file by ID."""
    if not DATASETS_DIR.exists():
        return None

    pattern = f"*_{dataset_id}.json"
    matches = list(DATASETS_DIR.glob(pattern))
    return matches[0] if matches else None


def time_to_months(time_val: list) -> float:
    """Convert [value, unit] to months."""
    value, unit = time_val
    if unit in ("month", "months"):
        return float(value)
    elif unit in ("year", "years"):
        return float(value) * 12
    elif unit in ("week", "weeks"):
        return float(value) / 4.33
    elif unit in ("day", "days"):
        return float(value) / 30
    else:
        return float(value)


def load_dataset(dataset_id: str) -> dict:
    """Load dataset and extract structured data."""
    path = find_dataset_by_id(dataset_id)
    if path is None:
        raise FileNotFoundError(f"No dataset found for ID: {dataset_id}")

    data = load_json(path)

    # Extract structured data
    questions = data["questions"]

    result = {
        "metadata": data["metadata"],
        "n_questions": len(questions),
        "short_rewards": [],
        "long_rewards": [],
        "short_delays": [],  # in months
        "long_delays": [],   # in months
        "time_horizons": [],  # in months, None for no horizon
        "reward_ratios": [],  # long/short
        "delay_ratios": [],   # long/short
        "questions": questions,
    }

    for q in questions:
        pair = q["preference_pair"]
        st = pair["short_term"]
        lt = pair["long_term"]

        result["short_rewards"].append(st["reward"])
        result["long_rewards"].append(lt["reward"])
        result["short_delays"].append(time_to_months(st["time"]))
        result["long_delays"].append(time_to_months(lt["time"]))

        th = q["time_horizon"]
        result["time_horizons"].append(time_to_months(th) if th else None)

        result["reward_ratios"].append(lt["reward"] / st["reward"] if st["reward"] > 0 else 0)
        result["delay_ratios"].append(
            time_to_months(lt["time"]) / time_to_months(st["time"])
            if time_to_months(st["time"]) > 0 else 0
        )

    return result


# =============================================================================
# Visualization Functions
# =============================================================================


def plot_reward_distributions(data: dict, output_dir: Path):
    """Plot reward distributions for short-term and long-term options."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Short-term rewards
    ax = axes[0]
    ax.hist(data["short_rewards"], bins=30, alpha=0.7, color="coral", edgecolor="black")
    ax.set_xlabel("Reward")
    ax.set_ylabel("Count")
    ax.set_title("Short-Term Rewards")
    ax.axvline(np.mean(data["short_rewards"]), color="red", linestyle="--",
               label=f'Mean: {np.mean(data["short_rewards"]):.0f}')
    ax.legend()

    # Long-term rewards
    ax = axes[1]
    ax.hist(data["long_rewards"], bins=30, alpha=0.7, color="steelblue", edgecolor="black")
    ax.set_xlabel("Reward")
    ax.set_ylabel("Count")
    ax.set_title("Long-Term Rewards")
    ax.axvline(np.mean(data["long_rewards"]), color="blue", linestyle="--",
               label=f'Mean: {np.mean(data["long_rewards"]):.0f}')
    ax.legend()

    # Overlaid
    ax = axes[2]
    ax.hist(data["short_rewards"], bins=30, alpha=0.5, color="coral",
            edgecolor="black", label="Short-term")
    ax.hist(data["long_rewards"], bins=30, alpha=0.5, color="steelblue",
            edgecolor="black", label="Long-term")
    ax.set_xlabel("Reward")
    ax.set_ylabel("Count")
    ax.set_title("Reward Comparison")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "reward_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_delay_distributions(data: dict, output_dir: Path):
    """Plot delay distributions for short-term and long-term options."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Short-term delays
    ax = axes[0]
    ax.hist(data["short_delays"], bins=30, alpha=0.7, color="coral", edgecolor="black")
    ax.set_xlabel("Delay (months)")
    ax.set_ylabel("Count")
    ax.set_title("Short-Term Delays")
    ax.axvline(np.mean(data["short_delays"]), color="red", linestyle="--",
               label=f'Mean: {np.mean(data["short_delays"]):.1f}')
    ax.legend()

    # Long-term delays (log scale might be useful)
    ax = axes[1]
    ax.hist(data["long_delays"], bins=30, alpha=0.7, color="steelblue", edgecolor="black")
    ax.set_xlabel("Delay (months)")
    ax.set_ylabel("Count")
    ax.set_title("Long-Term Delays")
    ax.axvline(np.mean(data["long_delays"]), color="blue", linestyle="--",
               label=f'Mean: {np.mean(data["long_delays"]):.1f}')
    ax.legend()

    # Log-scale comparison
    ax = axes[2]
    short_log = np.log10(np.array(data["short_delays"]) + 0.1)
    long_log = np.log10(np.array(data["long_delays"]) + 0.1)
    ax.hist(short_log, bins=30, alpha=0.5, color="coral", label="Short-term")
    ax.hist(long_log, bins=30, alpha=0.5, color="steelblue", label="Long-term")
    ax.set_xlabel("Log10(Delay + 0.1)")
    ax.set_ylabel("Count")
    ax.set_title("Delay Comparison (Log Scale)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "delay_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_time_horizon_distribution(data: dict, output_dir: Path):
    """Plot time horizon distribution."""
    horizons = [h for h in data["time_horizons"] if h is not None]
    n_with = len(horizons)
    n_without = len(data["time_horizons"]) - n_with

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Pie chart for with/without horizon
    ax = axes[0]
    ax.pie([n_with, n_without], labels=[f"With Horizon\n({n_with})", f"No Horizon\n({n_without})"],
           autopct="%1.1f%%", colors=["steelblue", "lightgray"])
    ax.set_title("Time Horizon Presence")

    # Distribution of horizon values
    ax = axes[1]
    if horizons:
        ax.hist(horizons, bins=30, alpha=0.7, color="steelblue", edgecolor="black")
        ax.set_xlabel("Time Horizon (months)")
        ax.set_ylabel("Count")
        ax.set_title("Time Horizon Distribution")
        ax.axvline(np.mean(horizons), color="red", linestyle="--",
                   label=f'Mean: {np.mean(horizons):.1f}')
        ax.axvline(12, color="green", linestyle=":", label="1 year")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No time horizons", ha="center", va="center", fontsize=14)
        ax.set_title("Time Horizon Distribution")

    plt.tight_layout()
    plt.savefig(output_dir / "time_horizon_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_ratio_distributions(data: dict, output_dir: Path):
    """Plot reward and delay ratio distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Reward ratios
    ax = axes[0]
    ax.hist(data["reward_ratios"], bins=30, alpha=0.7, color="green", edgecolor="black")
    ax.set_xlabel("Reward Ratio (Long/Short)")
    ax.set_ylabel("Count")
    ax.set_title("Reward Ratio Distribution")
    ax.axvline(np.mean(data["reward_ratios"]), color="darkgreen", linestyle="--",
               label=f'Mean: {np.mean(data["reward_ratios"]):.2f}')
    ax.legend()

    # Delay ratios (log scale)
    ax = axes[1]
    ratios = np.array(data["delay_ratios"])
    ratios = ratios[ratios > 0]  # Filter out zeros
    ax.hist(np.log10(ratios), bins=30, alpha=0.7, color="purple", edgecolor="black")
    ax.set_xlabel("Log10(Delay Ratio)")
    ax.set_ylabel("Count")
    ax.set_title("Delay Ratio Distribution (Log Scale)")
    ax.axvline(np.log10(np.mean(ratios)), color="darkviolet", linestyle="--",
               label=f'Mean: {np.mean(ratios):.1f}x')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "ratio_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_reward_delay_scatter(data: dict, output_dir: Path):
    """Plot reward vs delay scatter for both option types."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Short-term
    ax = axes[0]
    ax.scatter(data["short_delays"], data["short_rewards"], alpha=0.3, c="coral", s=10)
    ax.set_xlabel("Delay (months)")
    ax.set_ylabel("Reward")
    ax.set_title("Short-Term Options: Reward vs Delay")

    # Long-term
    ax = axes[1]
    ax.scatter(data["long_delays"], data["long_rewards"], alpha=0.3, c="steelblue", s=10)
    ax.set_xlabel("Delay (months)")
    ax.set_ylabel("Reward")
    ax.set_title("Long-Term Options: Reward vs Delay")

    plt.tight_layout()
    plt.savefig(output_dir / "reward_delay_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_reward_delay_heatmap(data: dict, output_dir: Path):
    """Plot 2D histograms (heatmaps) of reward vs delay."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Short-term
    ax = axes[0]
    h = ax.hist2d(data["short_delays"], data["short_rewards"], bins=20, cmap="YlOrRd")
    plt.colorbar(h[3], ax=ax, label="Count")
    ax.set_xlabel("Delay (months)")
    ax.set_ylabel("Reward")
    ax.set_title("Short-Term: Reward vs Delay")

    # Long-term
    ax = axes[1]
    h = ax.hist2d(data["long_delays"], data["long_rewards"], bins=20, cmap="YlGnBu")
    plt.colorbar(h[3], ax=ax, label="Count")
    ax.set_xlabel("Delay (months)")
    ax.set_ylabel("Reward")
    ax.set_title("Long-Term: Reward vs Delay")

    plt.tight_layout()
    plt.savefig(output_dir / "reward_delay_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_option_comparison(data: dict, output_dir: Path):
    """Plot direct comparison between short and long term options."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Reward comparison scatter
    ax = axes[0, 0]
    ax.scatter(data["short_rewards"], data["long_rewards"], alpha=0.3, s=10)
    max_r = max(max(data["short_rewards"]), max(data["long_rewards"]))
    ax.plot([0, max_r], [0, max_r], "r--", alpha=0.5, label="Equal")
    ax.set_xlabel("Short-Term Reward")
    ax.set_ylabel("Long-Term Reward")
    ax.set_title("Reward Comparison")
    ax.legend()

    # Delay comparison scatter
    ax = axes[0, 1]
    ax.scatter(data["short_delays"], data["long_delays"], alpha=0.3, s=10)
    ax.set_xlabel("Short-Term Delay (months)")
    ax.set_ylabel("Long-Term Delay (months)")
    ax.set_title("Delay Comparison")

    # Reward ratio vs delay ratio
    ax = axes[1, 0]
    ax.scatter(data["delay_ratios"], data["reward_ratios"], alpha=0.3, s=10, c="green")
    ax.set_xlabel("Delay Ratio (Long/Short)")
    ax.set_ylabel("Reward Ratio (Long/Short)")
    ax.set_title("Trade-off: Reward vs Delay Ratio")
    ax.axhline(1, color="red", linestyle="--", alpha=0.5)
    ax.axvline(1, color="red", linestyle="--", alpha=0.5)

    # "Implied discount rate" approximation
    ax = axes[1, 1]
    delay_diff = np.array(data["long_delays"]) - np.array(data["short_delays"])
    reward_ratio = np.array(data["reward_ratios"])
    # Rough implied annual discount: r_ratio = (1+r)^years
    years_diff = delay_diff / 12
    valid = (years_diff > 0) & (reward_ratio > 1)
    implied_rate = np.zeros_like(reward_ratio)
    implied_rate[valid] = (reward_ratio[valid] ** (1 / years_diff[valid])) - 1
    implied_rate = implied_rate[valid]
    ax.hist(implied_rate[implied_rate < 2], bins=30, alpha=0.7, color="teal", edgecolor="black")
    ax.set_xlabel("Implied Annual Growth Rate")
    ax.set_ylabel("Count")
    ax.set_title("Implied Annual Rate (Long/Short)")

    plt.tight_layout()
    plt.savefig(output_dir / "option_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_balance_analysis(data: dict, output_dir: Path):
    """Analyze balance across different dimensions."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Reward balance: bins and counts
    ax = axes[0, 0]
    short_bins = np.histogram(data["short_rewards"], bins=10)[0]
    long_bins = np.histogram(data["long_rewards"], bins=10)[0]
    x = np.arange(10)
    width = 0.35
    ax.bar(x - width/2, short_bins, width, label="Short-term", color="coral")
    ax.bar(x + width/2, long_bins, width, label="Long-term", color="steelblue")
    ax.set_xlabel("Reward Bin")
    ax.set_ylabel("Count")
    ax.set_title("Reward Distribution by Bins")
    ax.legend()

    # Time horizon balance
    ax = axes[0, 1]
    horizons = data["time_horizons"]
    categories = {"None": 0, "≤6mo": 0, "6mo-1yr": 0, "1-2yr": 0, "2-5yr": 0, ">5yr": 0}
    for h in horizons:
        if h is None:
            categories["None"] += 1
        elif h <= 6:
            categories["≤6mo"] += 1
        elif h <= 12:
            categories["6mo-1yr"] += 1
        elif h <= 24:
            categories["1-2yr"] += 1
        elif h <= 60:
            categories["2-5yr"] += 1
        else:
            categories[">5yr"] += 1
    ax.bar(categories.keys(), categories.values(), color="steelblue", edgecolor="black")
    ax.set_xlabel("Time Horizon Category")
    ax.set_ylabel("Count")
    ax.set_title("Time Horizon Category Balance")
    ax.tick_params(axis="x", rotation=45)

    # Delay difference distribution
    ax = axes[1, 0]
    delay_diff = np.array(data["long_delays"]) - np.array(data["short_delays"])
    ax.hist(delay_diff, bins=30, alpha=0.7, color="purple", edgecolor="black")
    ax.set_xlabel("Delay Difference (months)")
    ax.set_ylabel("Count")
    ax.set_title("Long - Short Delay Difference")
    ax.axvline(np.mean(delay_diff), color="red", linestyle="--",
               label=f'Mean: {np.mean(delay_diff):.1f}')
    ax.legend()

    # Reward difference distribution
    ax = axes[1, 1]
    reward_diff = np.array(data["long_rewards"]) - np.array(data["short_rewards"])
    ax.hist(reward_diff, bins=30, alpha=0.7, color="green", edgecolor="black")
    ax.set_xlabel("Reward Difference")
    ax.set_ylabel("Count")
    ax.set_title("Long - Short Reward Difference")
    ax.axvline(np.mean(reward_diff), color="red", linestyle="--",
               label=f'Mean: {np.mean(reward_diff):.0f}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "balance_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_summary_stats(data: dict, output_dir: Path):
    """Create a summary statistics figure."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis("off")

    stats_text = f"""
Dataset Summary Statistics
{'='*50}

Total Questions: {data['n_questions']}

SHORT-TERM OPTIONS:
  Reward: min={min(data['short_rewards']):.0f}, max={max(data['short_rewards']):.0f}, mean={np.mean(data['short_rewards']):.0f}, std={np.std(data['short_rewards']):.0f}
  Delay:  min={min(data['short_delays']):.1f}mo, max={max(data['short_delays']):.1f}mo, mean={np.mean(data['short_delays']):.1f}mo

LONG-TERM OPTIONS:
  Reward: min={min(data['long_rewards']):.0f}, max={max(data['long_rewards']):.0f}, mean={np.mean(data['long_rewards']):.0f}, std={np.std(data['long_rewards']):.0f}
  Delay:  min={min(data['long_delays']):.1f}mo, max={max(data['long_delays']):.1f}mo, mean={np.mean(data['long_delays']):.1f}mo

RATIOS:
  Reward (Long/Short): mean={np.mean(data['reward_ratios']):.2f}x, std={np.std(data['reward_ratios']):.2f}
  Delay (Long/Short):  mean={np.mean(data['delay_ratios']):.1f}x, std={np.std(data['delay_ratios']):.1f}

TIME HORIZONS:
  With horizon: {sum(1 for h in data['time_horizons'] if h is not None)}
  Without horizon: {sum(1 for h in data['time_horizons'] if h is None)}
  Horizon range: {min(h for h in data['time_horizons'] if h):.1f}mo - {max(h for h in data['time_horizons'] if h):.1f}mo
"""

    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.savefig(output_dir / "summary_stats.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_grid_coverage(data: dict, output_dir: Path):
    """Visualize coverage of the reward/delay space."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Short-term coverage
    ax = axes[0]
    h = ax.hist2d(data["short_delays"], data["short_rewards"],
                  bins=[10, 10], cmap="YlOrRd", cmin=0.5)
    plt.colorbar(h[3], ax=ax, label="Count")
    ax.set_xlabel("Delay (months)")
    ax.set_ylabel("Reward")
    ax.set_title("Short-Term Option Grid Coverage")

    # Long-term coverage
    ax = axes[1]
    h = ax.hist2d(data["long_delays"], data["long_rewards"],
                  bins=[10, 10], cmap="YlGnBu", cmin=0.5)
    plt.colorbar(h[3], ax=ax, label="Count")
    ax.set_xlabel("Delay (months)")
    ax.set_ylabel("Reward")
    ax.set_title("Long-Term Option Grid Coverage")

    plt.tight_layout()
    plt.savefig(output_dir / "grid_coverage.png", dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Analyze dataset structure")
    parser.add_argument("--dataset-id", required=True, help="Dataset ID to analyze")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    args = parser.parse_args()

    print(f"Loading dataset {args.dataset_id}...")
    data = load_dataset(args.dataset_id)

    output_dir = OUTPUT_DIR / args.dataset_id
    ensure_dir(output_dir)

    print(f"Generating visualizations in {output_dir}/...")

    # Generate all plots
    plot_reward_distributions(data, output_dir)
    print("  - reward_distributions.png")

    plot_delay_distributions(data, output_dir)
    print("  - delay_distributions.png")

    plot_time_horizon_distribution(data, output_dir)
    print("  - time_horizon_distribution.png")

    plot_ratio_distributions(data, output_dir)
    print("  - ratio_distributions.png")

    plot_reward_delay_scatter(data, output_dir)
    print("  - reward_delay_scatter.png")

    plot_reward_delay_heatmap(data, output_dir)
    print("  - reward_delay_heatmap.png")

    plot_option_comparison(data, output_dir)
    print("  - option_comparison.png")

    plot_balance_analysis(data, output_dir)
    print("  - balance_analysis.png")

    plot_summary_stats(data, output_dir)
    print("  - summary_stats.png")

    plot_grid_coverage(data, output_dir)
    print("  - grid_coverage.png")

    print(f"\nDone! Generated 10 visualizations in {output_dir}/")

    if args.show:
        # Reopen summary for display
        import subprocess
        subprocess.run(["open", str(output_dir / "summary_stats.png")])


if __name__ == "__main__":
    main()
