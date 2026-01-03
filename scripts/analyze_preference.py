#!/usr/bin/env python
"""
Analyze preference data (model choices) and produce comprehensive visualizations.

Produces graphs showing:
- Choice rates (short-term vs long-term vs unknown)
- Choice probability by delay, reward, time horizon
- Mean Subjective Value / Choice Rate by Delay
- Indifference Points (Raw, No Fit)
- Choice probability heatmaps (reward x delay)
- Conditional distributions
- Time horizon effects on choice
- Consistency analysis

Usage:
    python scripts/analyze_preference.py --query-id 7af8b316feb64ef4dd8ac94497fedf5b
    python scripts/analyze_preference.py --query-id 7af8b316feb64ef4dd8ac94497fedf5b --show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.io import ensure_dir, load_json

# =============================================================================
# Constants
# =============================================================================

PREFERENCE_DATA_DIR = PROJECT_ROOT / "out" / "preference_data"
OUTPUT_DIR = PROJECT_ROOT / "out" / "preference_analysis"

# =============================================================================
# Data Loading
# =============================================================================


def find_preference_data_by_query_id(query_id: str) -> Path | None:
    """Find preference data file by query_id suffix."""
    if not PREFERENCE_DATA_DIR.exists():
        return None

    pattern = f"*_{query_id}.json"
    matches = list(PREFERENCE_DATA_DIR.glob(pattern))
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


def load_preference_data(query_id: str) -> dict:
    """Load preference data and extract structured data for analysis."""
    path = find_preference_data_by_query_id(query_id)
    if path is None:
        raise FileNotFoundError(f"No preference data found for query_id: {query_id}")

    data = load_json(path)
    prefs = data["preferences"]

    result = {
        "metadata": data["metadata"],
        "model": data["metadata"]["model"],
        "n_samples": len(prefs),
        "choices": [],  # "short_term", "long_term", "unknown"
        "short_rewards": [],
        "long_rewards": [],
        "short_delays": [],
        "long_delays": [],
        "time_horizons": [],
        "reward_ratios": [],
        "delay_ratios": [],
        "delay_diffs": [],  # long - short
        "reward_diffs": [],  # long - short
        "preferences": prefs,
    }

    for p in prefs:
        pair = p["preference_pair"]
        st = pair["short_term"]
        lt = pair["long_term"]

        result["choices"].append(p["choice"])
        result["short_rewards"].append(st["reward"])
        result["long_rewards"].append(lt["reward"])

        short_delay = time_to_months(st["time"])
        long_delay = time_to_months(lt["time"])
        result["short_delays"].append(short_delay)
        result["long_delays"].append(long_delay)

        th = p["time_horizon"]
        result["time_horizons"].append(time_to_months(th) if th else None)

        result["reward_ratios"].append(lt["reward"] / st["reward"] if st["reward"] > 0 else 0)
        result["delay_ratios"].append(long_delay / short_delay if short_delay > 0 else 0)
        result["delay_diffs"].append(long_delay - short_delay)
        result["reward_diffs"].append(lt["reward"] - st["reward"])

    return result


# =============================================================================
# Visualization Functions
# =============================================================================


def plot_choice_distribution(data: dict, output_dir: Path):
    """Plot overall choice distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    choices = data["choices"]
    counts = {"short_term": 0, "long_term": 0, "unknown": 0}
    for c in choices:
        counts[c] = counts.get(c, 0) + 1

    # Pie chart
    ax = axes[0]
    colors = ["coral", "steelblue", "gray"]
    labels = [f"Short-term\n({counts['short_term']})",
              f"Long-term\n({counts['long_term']})",
              f"Unknown\n({counts['unknown']})"]
    sizes = [counts["short_term"], counts["long_term"], counts["unknown"]]
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors)
    ax.set_title(f"Choice Distribution\nModel: {data['model']}")

    # Bar chart
    ax = axes[1]
    ax.bar(["Short-term", "Long-term", "Unknown"],
           [counts["short_term"], counts["long_term"], counts["unknown"]],
           color=colors, edgecolor="black")
    ax.set_ylabel("Count")
    ax.set_title("Choice Counts")

    # Add choice rate annotation
    valid = counts["short_term"] + counts["long_term"]
    if valid > 0:
        lt_rate = counts["long_term"] / valid
        ax.text(0.95, 0.95, f"Long-term rate: {lt_rate:.1%}",
                transform=ax.transAxes, ha="right", va="top",
                bbox=dict(boxstyle="round", facecolor="wheat"))

    plt.tight_layout()
    plt.savefig(output_dir / "choice_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_choice_rate_by_delay(data: dict, output_dir: Path):
    """Plot choice rate (P(long-term)) as function of delay."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # By short-term delay
    ax = axes[0]
    short_delays = np.array(data["short_delays"])
    choices = np.array([1 if c == "long_term" else (0 if c == "short_term" else np.nan)
                        for c in data["choices"]])
    valid = ~np.isnan(choices)

    # Bin by short delay
    bins = np.percentile(short_delays[valid], np.linspace(0, 100, 11))
    bins = np.unique(bins)
    bin_centers = []
    bin_rates = []
    bin_stds = []
    bin_counts = []

    for i in range(len(bins) - 1):
        mask = valid & (short_delays >= bins[i]) & (short_delays < bins[i+1])
        if mask.sum() > 0:
            bin_centers.append((bins[i] + bins[i+1]) / 2)
            rate = choices[mask].mean()
            bin_rates.append(rate)
            bin_stds.append(np.sqrt(rate * (1 - rate) / mask.sum()))
            bin_counts.append(mask.sum())

    ax.errorbar(bin_centers, bin_rates, yerr=bin_stds, fmt="o-", capsize=3, color="steelblue")
    ax.set_xlabel("Short-Term Delay (months)")
    ax.set_ylabel("P(Choose Long-Term)")
    ax.set_title("Choice Rate by Short-Term Delay")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)

    # By long-term delay
    ax = axes[1]
    long_delays = np.array(data["long_delays"])

    bins = np.percentile(long_delays[valid], np.linspace(0, 100, 11))
    bins = np.unique(bins)
    bin_centers = []
    bin_rates = []
    bin_stds = []

    for i in range(len(bins) - 1):
        mask = valid & (long_delays >= bins[i]) & (long_delays < bins[i+1])
        if mask.sum() > 0:
            bin_centers.append((bins[i] + bins[i+1]) / 2)
            rate = choices[mask].mean()
            bin_rates.append(rate)
            bin_stds.append(np.sqrt(rate * (1 - rate) / mask.sum()))

    ax.errorbar(bin_centers, bin_rates, yerr=bin_stds, fmt="o-", capsize=3, color="coral")
    ax.set_xlabel("Long-Term Delay (months)")
    ax.set_ylabel("P(Choose Long-Term)")
    ax.set_title("Choice Rate by Long-Term Delay")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / "choice_rate_by_delay.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_choice_rate_by_reward(data: dict, output_dir: Path):
    """Plot choice rate as function of reward ratio."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    choices = np.array([1 if c == "long_term" else (0 if c == "short_term" else np.nan)
                        for c in data["choices"]])
    valid = ~np.isnan(choices)

    # By reward ratio
    ax = axes[0]
    reward_ratios = np.array(data["reward_ratios"])

    bins = np.percentile(reward_ratios[valid], np.linspace(0, 100, 11))
    bins = np.unique(bins)
    bin_centers = []
    bin_rates = []
    bin_stds = []

    for i in range(len(bins) - 1):
        mask = valid & (reward_ratios >= bins[i]) & (reward_ratios < bins[i+1])
        if mask.sum() > 0:
            bin_centers.append((bins[i] + bins[i+1]) / 2)
            rate = choices[mask].mean()
            bin_rates.append(rate)
            bin_stds.append(np.sqrt(rate * (1 - rate) / mask.sum()))

    ax.errorbar(bin_centers, bin_rates, yerr=bin_stds, fmt="o-", capsize=3, color="green")
    ax.set_xlabel("Reward Ratio (Long/Short)")
    ax.set_ylabel("P(Choose Long-Term)")
    ax.set_title("Choice Rate by Reward Ratio")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)

    # By delay ratio
    ax = axes[1]
    delay_ratios = np.array(data["delay_ratios"])

    bins = np.percentile(delay_ratios[valid], np.linspace(0, 100, 11))
    bins = np.unique(bins)
    bin_centers = []
    bin_rates = []
    bin_stds = []

    for i in range(len(bins) - 1):
        mask = valid & (delay_ratios >= bins[i]) & (delay_ratios < bins[i+1])
        if mask.sum() > 0:
            bin_centers.append((bins[i] + bins[i+1]) / 2)
            rate = choices[mask].mean()
            bin_rates.append(rate)
            bin_stds.append(np.sqrt(rate * (1 - rate) / mask.sum()))

    ax.errorbar(bin_centers, bin_rates, yerr=bin_stds, fmt="o-", capsize=3, color="purple")
    ax.set_xlabel("Delay Ratio (Long/Short)")
    ax.set_ylabel("P(Choose Long-Term)")
    ax.set_title("Choice Rate by Delay Ratio")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / "choice_rate_by_reward.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_choice_probability_heatmap(data: dict, output_dir: Path):
    """Plot choice probability heatmap (2D: reward ratio vs delay ratio)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    choices = np.array([1 if c == "long_term" else (0 if c == "short_term" else np.nan)
                        for c in data["choices"]])
    valid = ~np.isnan(choices)

    reward_ratios = np.array(data["reward_ratios"])[valid]
    delay_ratios = np.array(data["delay_ratios"])[valid]
    choices_valid = choices[valid]

    # Heatmap: reward ratio vs delay ratio
    ax = axes[0]
    n_bins = 8

    # Create bins
    rr_bins = np.percentile(reward_ratios, np.linspace(0, 100, n_bins + 1))
    dr_bins = np.percentile(delay_ratios, np.linspace(0, 100, n_bins + 1))

    heatmap = np.zeros((n_bins, n_bins))
    counts = np.zeros((n_bins, n_bins))

    for i in range(n_bins):
        for j in range(n_bins):
            mask = ((reward_ratios >= rr_bins[i]) & (reward_ratios < rr_bins[i+1]) &
                    (delay_ratios >= dr_bins[j]) & (delay_ratios < dr_bins[j+1]))
            if mask.sum() > 0:
                heatmap[j, i] = choices_valid[mask].mean()
                counts[j, i] = mask.sum()

    im = ax.imshow(heatmap, origin="lower", aspect="auto", cmap="RdYlBu",
                   vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="P(Long-Term)")

    # Labels
    ax.set_xticks(range(n_bins))
    ax.set_xticklabels([f"{rr_bins[i]:.1f}" for i in range(n_bins)], rotation=45)
    ax.set_yticks(range(n_bins))
    ax.set_yticklabels([f"{dr_bins[i]:.0f}" for i in range(n_bins)])
    ax.set_xlabel("Reward Ratio (Long/Short)")
    ax.set_ylabel("Delay Ratio (Long/Short)")
    ax.set_title("Choice Probability Heatmap")

    # Annotate with counts
    for i in range(n_bins):
        for j in range(n_bins):
            if counts[j, i] > 0:
                color = "white" if heatmap[j, i] < 0.3 or heatmap[j, i] > 0.7 else "black"
                ax.text(i, j, f"{heatmap[j,i]:.2f}\n({int(counts[j,i])})",
                        ha="center", va="center", fontsize=7, color=color)

    # Second heatmap: short delay vs long delay
    ax = axes[1]
    short_delays = np.array(data["short_delays"])[valid]
    long_delays = np.array(data["long_delays"])[valid]

    sd_bins = np.percentile(short_delays, np.linspace(0, 100, n_bins + 1))
    ld_bins = np.percentile(long_delays, np.linspace(0, 100, n_bins + 1))

    heatmap2 = np.zeros((n_bins, n_bins))
    counts2 = np.zeros((n_bins, n_bins))

    for i in range(n_bins):
        for j in range(n_bins):
            mask = ((short_delays >= sd_bins[i]) & (short_delays < sd_bins[i+1]) &
                    (long_delays >= ld_bins[j]) & (long_delays < ld_bins[j+1]))
            if mask.sum() > 0:
                heatmap2[j, i] = choices_valid[mask].mean()
                counts2[j, i] = mask.sum()

    im = ax.imshow(heatmap2, origin="lower", aspect="auto", cmap="RdYlBu",
                   vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="P(Long-Term)")

    ax.set_xticks(range(n_bins))
    ax.set_xticklabels([f"{sd_bins[i]:.1f}" for i in range(n_bins)], rotation=45)
    ax.set_yticks(range(n_bins))
    ax.set_yticklabels([f"{ld_bins[i]:.0f}" for i in range(n_bins)])
    ax.set_xlabel("Short-Term Delay (months)")
    ax.set_ylabel("Long-Term Delay (months)")
    ax.set_title("Choice Probability by Delays")

    plt.tight_layout()
    plt.savefig(output_dir / "choice_probability_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_indifference_points(data: dict, output_dir: Path):
    """
    Plot indifference points (raw, no fit).

    Indifference point: where P(long-term) ≈ 0.5 for a given delay.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    choices = np.array([1 if c == "long_term" else (0 if c == "short_term" else np.nan)
                        for c in data["choices"]])
    valid = ~np.isnan(choices)

    delay_diffs = np.array(data["delay_diffs"])[valid]
    reward_ratios = np.array(data["reward_ratios"])[valid]
    choices_valid = choices[valid]

    # Group by delay difference, find reward ratio at indifference
    ax = axes[0]
    delay_bins = np.percentile(delay_diffs, np.linspace(0, 100, 9))
    delay_bins = np.unique(delay_bins)

    indiff_points = []
    delay_centers = []

    for i in range(len(delay_bins) - 1):
        mask = (delay_diffs >= delay_bins[i]) & (delay_diffs < delay_bins[i+1])
        if mask.sum() >= 10:
            # Find reward ratio where choice rate crosses 0.5
            sub_rr = reward_ratios[mask]
            sub_choice = choices_valid[mask]

            # Sort by reward ratio and compute running mean
            order = np.argsort(sub_rr)
            sub_rr_sorted = sub_rr[order]
            sub_choice_sorted = sub_choice[order]

            # Find crossing point
            window = max(5, len(sub_choice_sorted) // 5)
            for j in range(len(sub_choice_sorted) - window):
                rate = sub_choice_sorted[j:j+window].mean()
                if rate >= 0.5:
                    indiff_points.append(sub_rr_sorted[j + window // 2])
                    delay_centers.append((delay_bins[i] + delay_bins[i+1]) / 2)
                    break

    if indiff_points:
        ax.scatter(delay_centers, indiff_points, s=100, c="steelblue", edgecolors="black")
        ax.plot(delay_centers, indiff_points, "b--", alpha=0.5)
    ax.set_xlabel("Delay Difference (months)")
    ax.set_ylabel("Reward Ratio at Indifference")
    ax.set_title("Indifference Points (Raw, No Fit)\nReward ratio needed for 50% long-term choice")

    # Alternative: Mean Subjective Value by delay
    ax = axes[1]

    # Compute "subjective value" as choice rate weighted by reward
    delay_bins = np.linspace(min(delay_diffs), max(delay_diffs), 12)
    msv = []  # mean subjective value
    delay_centers = []

    for i in range(len(delay_bins) - 1):
        mask = (delay_diffs >= delay_bins[i]) & (delay_diffs < delay_bins[i+1])
        if mask.sum() > 0:
            # MSV approximation: choice_rate as proxy
            msv.append(choices_valid[mask].mean())
            delay_centers.append((delay_bins[i] + delay_bins[i+1]) / 2)

    ax.plot(delay_centers, msv, "o-", color="green", markersize=8)
    ax.fill_between(delay_centers, msv, alpha=0.3, color="green")
    ax.set_xlabel("Delay Difference (Long - Short, months)")
    ax.set_ylabel("Choice Rate (Long-Term)")
    ax.set_title("Mean Choice Rate by Delay Difference")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / "indifference_points.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_time_horizon_effects(data: dict, output_dir: Path):
    """Plot how time horizon affects choices."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    choices = np.array([1 if c == "long_term" else (0 if c == "short_term" else np.nan)
                        for c in data["choices"]])
    valid = ~np.isnan(choices)
    time_horizons = data["time_horizons"]

    # Separate by horizon presence
    has_horizon = np.array([h is not None for h in time_horizons])
    horizons_values = np.array([h if h is not None else 0 for h in time_horizons])

    # Choice rate with vs without horizon
    ax = axes[0, 0]
    with_h = valid & has_horizon
    without_h = valid & ~has_horizon

    rates = []
    labels = []
    if with_h.sum() > 0:
        rates.append(choices[with_h].mean())
        labels.append(f"With Horizon\n(n={with_h.sum()})")
    if without_h.sum() > 0:
        rates.append(choices[without_h].mean())
        labels.append(f"No Horizon\n(n={without_h.sum()})")

    if rates:
        colors = ["steelblue", "lightgray"][:len(rates)]
        ax.bar(labels, rates, color=colors, edgecolor="black")
        ax.set_ylabel("P(Long-Term)")
        ax.set_title("Choice Rate by Horizon Presence")
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color="red", linestyle="--", alpha=0.5)

    # Choice rate by horizon value
    ax = axes[0, 1]
    if has_horizon.sum() > 10:
        h_valid = valid & has_horizon
        h_vals = horizons_values[h_valid]
        h_choices = choices[h_valid]

        bins = np.percentile(h_vals, np.linspace(0, 100, 9))
        bins = np.unique(bins)
        bin_centers = []
        bin_rates = []
        bin_stds = []

        for i in range(len(bins) - 1):
            mask = (h_vals >= bins[i]) & (h_vals < bins[i+1])
            if mask.sum() > 0:
                bin_centers.append((bins[i] + bins[i+1]) / 2)
                rate = h_choices[mask].mean()
                bin_rates.append(rate)
                bin_stds.append(np.sqrt(rate * (1 - rate) / mask.sum()))

        ax.errorbar(bin_centers, bin_rates, yerr=bin_stds, fmt="o-", capsize=3, color="steelblue")
        ax.set_xlabel("Time Horizon (months)")
        ax.set_ylabel("P(Long-Term)")
        ax.set_title("Choice Rate by Time Horizon Value")
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    else:
        ax.text(0.5, 0.5, "Insufficient horizon data", ha="center", va="center")
        ax.set_title("Choice Rate by Time Horizon Value")

    # Horizon vs Long delay relationship
    ax = axes[1, 0]
    if has_horizon.sum() > 10:
        h_valid = has_horizon
        h_vals = horizons_values[h_valid]
        long_d = np.array(data["long_delays"])[h_valid]
        c = choices[h_valid & valid]

        # Color by choice
        colors = ["coral" if x == 0 else "steelblue" for x in c[:len(h_vals)]]
        ax.scatter(h_vals[:len(c)], long_d[:len(c)], c=colors, alpha=0.3, s=20)
        ax.plot([0, max(h_vals)], [0, max(h_vals)], "k--", alpha=0.3, label="y=x")
        ax.set_xlabel("Time Horizon (months)")
        ax.set_ylabel("Long-Term Delay (months)")
        ax.set_title("Horizon vs Long Delay\n(blue=long-term choice, coral=short-term)")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Insufficient horizon data", ha="center", va="center")

    # Horizon categories
    ax = axes[1, 1]
    if has_horizon.sum() > 0:
        categories = {"≤6mo": [], "6mo-1yr": [], "1-2yr": [], "2-5yr": [], ">5yr": []}
        for i, h in enumerate(time_horizons):
            if h is not None and valid[i]:
                if h <= 6:
                    categories["≤6mo"].append(choices[i])
                elif h <= 12:
                    categories["6mo-1yr"].append(choices[i])
                elif h <= 24:
                    categories["1-2yr"].append(choices[i])
                elif h <= 60:
                    categories["2-5yr"].append(choices[i])
                else:
                    categories[">5yr"].append(choices[i])

        cat_names = []
        cat_rates = []
        cat_counts = []
        for name, vals in categories.items():
            if len(vals) > 0:
                cat_names.append(name)
                cat_rates.append(np.mean(vals))
                cat_counts.append(len(vals))

        bars = ax.bar(cat_names, cat_rates, color="steelblue", edgecolor="black")
        ax.set_ylabel("P(Long-Term)")
        ax.set_xlabel("Time Horizon Category")
        ax.set_title("Choice Rate by Horizon Category")
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color="red", linestyle="--", alpha=0.5)

        # Add counts
        for bar, count in zip(bars, cat_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"n={count}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "time_horizon_effects.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_conditional_distributions(data: dict, output_dir: Path):
    """Plot conditional distributions of features given choice."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    choices = data["choices"]
    short_idx = [i for i, c in enumerate(choices) if c == "short_term"]
    long_idx = [i for i, c in enumerate(choices) if c == "long_term"]

    # Reward ratio | choice
    ax = axes[0, 0]
    rr = np.array(data["reward_ratios"])
    ax.hist(rr[short_idx], bins=20, alpha=0.5, color="coral", label="Short-term", density=True)
    ax.hist(rr[long_idx], bins=20, alpha=0.5, color="steelblue", label="Long-term", density=True)
    ax.set_xlabel("Reward Ratio")
    ax.set_ylabel("Density")
    ax.set_title("Reward Ratio | Choice")
    ax.legend()

    # Delay ratio | choice
    ax = axes[0, 1]
    dr = np.array(data["delay_ratios"])
    ax.hist(np.log10(dr[short_idx] + 1), bins=20, alpha=0.5, color="coral", label="Short-term", density=True)
    ax.hist(np.log10(dr[long_idx] + 1), bins=20, alpha=0.5, color="steelblue", label="Long-term", density=True)
    ax.set_xlabel("Log10(Delay Ratio + 1)")
    ax.set_ylabel("Density")
    ax.set_title("Delay Ratio | Choice")
    ax.legend()

    # Short reward | choice
    ax = axes[0, 2]
    sr = np.array(data["short_rewards"])
    ax.hist(sr[short_idx], bins=20, alpha=0.5, color="coral", label="Short-term", density=True)
    ax.hist(sr[long_idx], bins=20, alpha=0.5, color="steelblue", label="Long-term", density=True)
    ax.set_xlabel("Short-Term Reward")
    ax.set_ylabel("Density")
    ax.set_title("Short Reward | Choice")
    ax.legend()

    # Long reward | choice
    ax = axes[1, 0]
    lr = np.array(data["long_rewards"])
    ax.hist(lr[short_idx], bins=20, alpha=0.5, color="coral", label="Short-term", density=True)
    ax.hist(lr[long_idx], bins=20, alpha=0.5, color="steelblue", label="Long-term", density=True)
    ax.set_xlabel("Long-Term Reward")
    ax.set_ylabel("Density")
    ax.set_title("Long Reward | Choice")
    ax.legend()

    # Delay diff | choice
    ax = axes[1, 1]
    dd = np.array(data["delay_diffs"])
    ax.hist(dd[short_idx], bins=20, alpha=0.5, color="coral", label="Short-term", density=True)
    ax.hist(dd[long_idx], bins=20, alpha=0.5, color="steelblue", label="Long-term", density=True)
    ax.set_xlabel("Delay Difference (months)")
    ax.set_ylabel("Density")
    ax.set_title("Delay Diff | Choice")
    ax.legend()

    # Time horizon | choice (if available)
    ax = axes[1, 2]
    th = data["time_horizons"]
    th_short = [th[i] for i in short_idx if th[i] is not None]
    th_long = [th[i] for i in long_idx if th[i] is not None]
    if th_short and th_long:
        ax.hist(th_short, bins=20, alpha=0.5, color="coral", label="Short-term", density=True)
        ax.hist(th_long, bins=20, alpha=0.5, color="steelblue", label="Long-term", density=True)
        ax.set_xlabel("Time Horizon (months)")
        ax.set_ylabel("Density")
        ax.set_title("Time Horizon | Choice")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Insufficient horizon data", ha="center", va="center")
        ax.set_title("Time Horizon | Choice")

    plt.tight_layout()
    plt.savefig(output_dir / "conditional_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_decision_boundary(data: dict, output_dir: Path):
    """Attempt to visualize decision boundary in reward-delay space."""
    fig, ax = plt.subplots(figsize=(10, 8))

    choices = np.array([1 if c == "long_term" else (0 if c == "short_term" else np.nan)
                        for c in data["choices"]])
    valid = ~np.isnan(choices)

    reward_ratios = np.array(data["reward_ratios"])[valid]
    delay_ratios = np.array(data["delay_ratios"])[valid]
    choices_valid = choices[valid]

    # Scatter plot colored by choice
    colors = ["coral" if c == 0 else "steelblue" for c in choices_valid]
    ax.scatter(delay_ratios, reward_ratios, c=colors, alpha=0.3, s=15)

    ax.set_xlabel("Delay Ratio (Long/Short)")
    ax.set_ylabel("Reward Ratio (Long/Short)")
    ax.set_title("Decision Space\n(blue=long-term, coral=short-term)")

    # Try to fit a simple logistic boundary
    try:
        from sklearn.linear_model import LogisticRegression
        X = np.column_stack([np.log(delay_ratios + 1), reward_ratios])
        clf = LogisticRegression()
        clf.fit(X, choices_valid)

        # Plot decision boundary
        xx = np.linspace(0, max(delay_ratios), 100)
        # Decision boundary: w0*log(x+1) + w1*y + b = 0
        # y = -(w0*log(x+1) + b) / w1
        w0, w1 = clf.coef_[0]
        b = clf.intercept_[0]
        yy = -(w0 * np.log(xx + 1) + b) / w1
        valid_boundary = (yy > 0) & (yy < max(reward_ratios) * 1.1)
        ax.plot(xx[valid_boundary], yy[valid_boundary], "k--", linewidth=2,
                label="Logistic boundary")
        ax.legend()
    except Exception:
        pass  # Skip if sklearn not available or fit fails

    plt.tight_layout()
    plt.savefig(output_dir / "decision_boundary.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_summary_stats(data: dict, output_dir: Path):
    """Create a summary statistics figure."""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis("off")

    choices = data["choices"]
    n_short = sum(1 for c in choices if c == "short_term")
    n_long = sum(1 for c in choices if c == "long_term")
    n_unknown = sum(1 for c in choices if c == "unknown")
    n_valid = n_short + n_long

    horizons = [h for h in data["time_horizons"] if h is not None]

    stats_text = f"""
Preference Data Summary
{'='*60}

Model: {data['model']}
Total Samples: {data['n_samples']}

CHOICE DISTRIBUTION:
  Short-term: {n_short} ({100*n_short/data['n_samples']:.1f}%)
  Long-term:  {n_long} ({100*n_long/data['n_samples']:.1f}%)
  Unknown:    {n_unknown} ({100*n_unknown/data['n_samples']:.1f}%)

  Long-term rate (valid only): {100*n_long/n_valid:.1f}% (n={n_valid})

REWARD ANALYSIS:
  Short-term: mean={np.mean(data['short_rewards']):.0f}, std={np.std(data['short_rewards']):.0f}
  Long-term:  mean={np.mean(data['long_rewards']):.0f}, std={np.std(data['long_rewards']):.0f}
  Ratio (L/S): mean={np.mean(data['reward_ratios']):.2f}x

DELAY ANALYSIS:
  Short-term: mean={np.mean(data['short_delays']):.1f}mo, std={np.std(data['short_delays']):.1f}mo
  Long-term:  mean={np.mean(data['long_delays']):.1f}mo, std={np.std(data['long_delays']):.1f}mo
  Ratio (L/S): mean={np.mean(data['delay_ratios']):.1f}x
  Difference:  mean={np.mean(data['delay_diffs']):.1f}mo

TIME HORIZONS:
  With horizon: {len(horizons)} ({100*len(horizons)/data['n_samples']:.1f}%)
  Horizon range: {min(horizons):.1f}mo - {max(horizons):.1f}mo (mean={np.mean(horizons):.1f}mo)
"""

    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.savefig(output_dir / "summary_stats.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_msv_by_delay(data: dict, output_dir: Path):
    """Plot Mean Subjective Value (approximated by choice rate) by delay."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    choices = np.array([1 if c == "long_term" else (0 if c == "short_term" else np.nan)
                        for c in data["choices"]])
    valid = ~np.isnan(choices)

    long_delays = np.array(data["long_delays"])[valid]
    choices_valid = choices[valid]

    # MSV by absolute long delay
    ax = axes[0]
    bins = np.linspace(min(long_delays), max(long_delays), 15)
    centers = []
    msv = []
    counts = []

    for i in range(len(bins) - 1):
        mask = (long_delays >= bins[i]) & (long_delays < bins[i+1])
        if mask.sum() >= 5:
            centers.append((bins[i] + bins[i+1]) / 2)
            msv.append(choices_valid[mask].mean())
            counts.append(mask.sum())

    ax.bar(centers, msv, width=(bins[1] - bins[0]) * 0.8, color="steelblue",
           edgecolor="black", alpha=0.7)
    ax.set_xlabel("Long-Term Delay (months)")
    ax.set_ylabel("Choice Rate (Long-Term)")
    ax.set_title("Mean Subjective Value by Long-Term Delay")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5)

    # MSV by delay difference
    ax = axes[1]
    delay_diffs = np.array(data["delay_diffs"])[valid]
    bins = np.linspace(min(delay_diffs), max(delay_diffs), 15)
    centers = []
    msv = []

    for i in range(len(bins) - 1):
        mask = (delay_diffs >= bins[i]) & (delay_diffs < bins[i+1])
        if mask.sum() >= 5:
            centers.append((bins[i] + bins[i+1]) / 2)
            msv.append(choices_valid[mask].mean())

    ax.bar(centers, msv, width=(bins[1] - bins[0]) * 0.8, color="coral",
           edgecolor="black", alpha=0.7)
    ax.set_xlabel("Delay Difference (Long - Short, months)")
    ax.set_ylabel("Choice Rate (Long-Term)")
    ax.set_title("Mean Subjective Value by Delay Difference")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / "msv_by_delay.png", dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Analyze preference data")
    parser.add_argument("--query-id", required=True, help="Query ID to analyze")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    args = parser.parse_args()

    print(f"Loading preference data for query {args.query_id}...")
    data = load_preference_data(args.query_id)

    output_dir = OUTPUT_DIR / args.query_id
    ensure_dir(output_dir)

    print(f"Model: {data['model']}")
    print(f"Samples: {data['n_samples']}")
    print(f"Generating visualizations in {output_dir}/...")

    # Generate all plots
    plot_choice_distribution(data, output_dir)
    print("  - choice_distribution.png")

    plot_choice_rate_by_delay(data, output_dir)
    print("  - choice_rate_by_delay.png")

    plot_choice_rate_by_reward(data, output_dir)
    print("  - choice_rate_by_reward.png")

    plot_choice_probability_heatmap(data, output_dir)
    print("  - choice_probability_heatmap.png")

    plot_indifference_points(data, output_dir)
    print("  - indifference_points.png")

    plot_time_horizon_effects(data, output_dir)
    print("  - time_horizon_effects.png")

    plot_conditional_distributions(data, output_dir)
    print("  - conditional_distributions.png")

    plot_decision_boundary(data, output_dir)
    print("  - decision_boundary.png")

    plot_msv_by_delay(data, output_dir)
    print("  - msv_by_delay.png")

    plot_summary_stats(data, output_dir)
    print("  - summary_stats.png")

    print(f"\nDone! Generated 10 visualizations in {output_dir}/")

    if args.show:
        import subprocess
        subprocess.run(["open", str(output_dir / "summary_stats.png")])


if __name__ == "__main__":
    main()
