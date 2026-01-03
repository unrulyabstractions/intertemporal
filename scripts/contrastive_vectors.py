#!/usr/bin/env python
"""
Compute Contrastive Activation Addition (CAA) steering vectors.

CAA computes steering directions by taking the difference between mean activations
of contrasting classes. For intertemporal preferences:
- direction = mean(long_term activations) - mean(short_term activations)

This produces steering vectors that can be used to push the model toward
long-term or short-term choices.

Config file options:
    {
      "query_ids": ["query_id1", "query_id2"],  // Query IDs to use
      "normalize": true,  // Whether to L2-normalize directions (default: true)
      "min_samples_per_class": 10  // Minimum samples required per class (default: 10)
    }

Usage:
    python scripts/contrastive_vectors.py  # Uses default_contrastive.json
    python scripts/contrastive_vectors.py --config my_caa_config
    python scripts/contrastive_vectors.py --query-id abc123 --no-normalize

Outputs:
    out/contrastive/{contrastive_id}/
        - vectors/caa_layer{L}_pos{P}.npy: Steering direction vectors
        - results/caa_results.json: Summary with metadata and statistics
        - viz/: Visualizations of the vectors
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

# Add project root and scripts to path
PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from src.common.io import ensure_dir, get_timestamp, load_json, save_json
from src.common.schema_utils import SchemaClass
from src.probes import (
    CombinedPreferenceData,
    TokenPositionSpec,
    load_combined_preference_data,
)
from src.plotting import (
    TokenPositionInfo,
    format_token_position_label,
)

import matplotlib.pyplot as plt


# =============================================================================
# Configuration Schema
# =============================================================================


@dataclass
class CAAConfigSchema(SchemaClass):
    """Schema for CAA config - used for deterministic folder IDs."""
    query_ids: tuple[str, ...]
    normalize: bool
    min_samples_per_class: int


@dataclass
class CAAConfig:
    """Configuration for contrastive vector computation."""
    query_ids: list[str]
    normalize: bool = True
    min_samples_per_class: int = 10

    def get_schema(self) -> CAAConfigSchema:
        return CAAConfigSchema(
            query_ids=tuple(sorted(self.query_ids)),
            normalize=self.normalize,
            min_samples_per_class=self.min_samples_per_class,
        )

    def get_id(self) -> str:
        return self.get_schema().get_id()


def load_caa_config(path: Path) -> CAAConfig:
    """Load CAA config from JSON file."""
    data = load_json(path)
    return CAAConfig(
        query_ids=data.get("query_ids", []),
        normalize=data.get("normalize", True),
        min_samples_per_class=data.get("min_samples_per_class", 10),
    )


def get_resolved_token_positions(
    query_ids: list[str],
) -> tuple[dict[int, int], dict[int, str]]:
    """
    Get resolved token positions and words from preference data.

    Args:
        query_ids: List of query IDs to load data from

    Returns:
        Tuple of (resolved_positions, tokens) dicts mapping idx -> value
    """
    from src.probes.data import find_preference_data_by_query_id, load_preference_data_file

    if not query_ids:
        return {}, {}

    # Load the first preference file to get resolved positions and tokens
    path = find_preference_data_by_query_id(query_ids[0])
    if path is None:
        return {}, {}

    data = load_preference_data_file(path)

    # Look for resolved positions in sample internals
    for pref in data.get("preferences", data.get("samples", [])):
        internals = pref.get("internals", {})
        if internals.get("token_positions") and internals.get("tokens"):
            resolved_positions = internals["token_positions"]
            tokens_list = internals["tokens"]

            # Build dicts
            resolved_dict = {i: pos for i, pos in enumerate(resolved_positions)}
            tokens_dict = {i: tok for i, tok in enumerate(tokens_list)}
            return resolved_dict, tokens_dict

    return {}, {}


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class ContrastiveVector:
    """A single contrastive activation vector."""
    layer: int
    token_position_idx: int  # Relative index (0, 1, 2, ...)
    token_position: Optional[int]  # Actual position in token sequence
    token_position_word: Optional[str]  # Token/word at that position
    direction: np.ndarray  # shape: (d_model,)
    n_short_term: int
    n_long_term: int
    mean_short_term: np.ndarray
    mean_long_term: np.ndarray
    cosine_sim_to_probe: Optional[float] = None  # If comparing to probe direction


@dataclass
class CAAOutput:
    """Output from contrastive vector computation."""
    contrastive_id: str
    query_ids: list[str]
    model: str
    timestamp: str
    n_samples: int
    n_short_term: int
    n_long_term: int
    layers: list[int]
    token_positions: list[int]
    vectors: list[ContrastiveVector]
    d_model: int
    normalized: bool


# =============================================================================
# Contrastive Vector Computation
# =============================================================================


def compute_contrastive_vectors(
    data: CombinedPreferenceData,
    normalize: bool = True,
    min_samples: int = 10,
    resolved_positions: Optional[dict[int, int]] = None,
    tokens: Optional[dict[int, str]] = None,
) -> list[ContrastiveVector]:
    """
    Compute contrastive activation vectors for all (layer, position) combinations.

    Args:
        data: Combined preference data with activations
        normalize: Whether to L2-normalize the resulting vectors
        min_samples: Minimum samples per class required
        resolved_positions: Optional mapping of token_position_idx to actual position
        tokens: Optional mapping of token_position_idx to token word

    Returns:
        List of ContrastiveVector objects
    """
    # Separate samples by choice
    short_term_samples = [s for s in data.samples if s.choice == "short_term"]
    long_term_samples = [s for s in data.samples if s.choice == "long_term"]

    print(f"  Short-term samples: {len(short_term_samples)}")
    print(f"  Long-term samples: {len(long_term_samples)}")

    if len(short_term_samples) < min_samples or len(long_term_samples) < min_samples:
        raise ValueError(
            f"Insufficient samples: need at least {min_samples} per class, "
            f"got {len(short_term_samples)} short-term and {len(long_term_samples)} long-term"
        )

    # Find all (layer, token_pos_idx) combinations
    all_keys: set[tuple[int, int]] = set()
    for sample in data.samples:
        all_keys.update(sample.activations.keys())

    vectors = []

    for layer, pos_idx in sorted(all_keys):
        # Collect activations for each class
        short_acts = []
        long_acts = []

        for sample in short_term_samples:
            if (layer, pos_idx) in sample.activations:
                short_acts.append(sample.activations[(layer, pos_idx)])

        for sample in long_term_samples:
            if (layer, pos_idx) in sample.activations:
                long_acts.append(sample.activations[(layer, pos_idx)])

        if len(short_acts) < min_samples or len(long_acts) < min_samples:
            continue

        # Compute means
        short_acts = np.stack(short_acts)
        long_acts = np.stack(long_acts)

        mean_short = short_acts.mean(axis=0)
        mean_long = long_acts.mean(axis=0)

        # Contrastive direction: long - short
        # Positive direction = toward long-term
        direction = mean_long - mean_short

        if normalize:
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                direction = direction / norm

        # Get actual token position and word if available
        token_pos = resolved_positions.get(pos_idx) if resolved_positions else None
        token_word = tokens.get(pos_idx) if tokens else None

        vectors.append(ContrastiveVector(
            layer=layer,
            token_position_idx=pos_idx,
            token_position=token_pos,
            token_position_word=token_word,
            direction=direction,
            n_short_term=len(short_acts),
            n_long_term=len(long_acts),
            mean_short_term=mean_short,
            mean_long_term=mean_long,
        ))

    return vectors


# =============================================================================
# Analysis and Visualization
# =============================================================================


def analyze_vectors(vectors: list[ContrastiveVector]) -> dict:
    """Compute statistics about the contrastive vectors."""
    stats = {
        "n_vectors": len(vectors),
        "layers": sorted(set(v.layer for v in vectors)),
        "positions": sorted(set(v.token_position_idx for v in vectors)),
        "per_vector": [],
    }

    for v in vectors:
        # Compute some statistics about each vector
        norm = np.linalg.norm(v.direction)
        mean_abs = np.abs(v.direction).mean()
        max_abs = np.abs(v.direction).max()

        # Distance between class means
        class_distance = np.linalg.norm(v.mean_long_term - v.mean_short_term)

        stats["per_vector"].append({
            "layer": v.layer,
            "position": v.token_position_idx,
            "token_position": v.token_position,
            "token_position_word": v.token_position_word,
            "n_short": v.n_short_term,
            "n_long": v.n_long_term,
            "direction_norm": float(norm),
            "direction_mean_abs": float(mean_abs),
            "direction_max_abs": float(max_abs),
            "class_distance": float(class_distance),
        })

    return stats


def plot_direction_norms(vectors: list[ContrastiveVector], output_dir: Path):
    """Plot heatmap of direction norms by layer and position."""
    layers = sorted(set(v.layer for v in vectors))
    positions = sorted(set(v.token_position_idx for v in vectors))

    # Build matrix
    norm_matrix = np.zeros((len(layers), len(positions)))
    for v in vectors:
        i = layers.index(v.layer)
        j = positions.index(v.token_position_idx)
        norm_matrix[i, j] = np.linalg.norm(v.mean_long_term - v.mean_short_term)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(norm_matrix, aspect="auto", cmap="viridis")
    plt.colorbar(im, ax=ax, label="Class Distance (L2)")

    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels([f"pos_{p}" for p in positions], rotation=45, ha="right")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"Layer {l}" for l in layers])
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Layer")
    ax.set_title("Contrastive Vector Magnitude\n(Distance between class means)")

    plt.tight_layout()
    plt.savefig(output_dir / "direction_norms.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_direction_similarity(vectors: list[ContrastiveVector], output_dir: Path):
    """Plot cosine similarity between directions at different positions."""
    if len(vectors) < 2:
        return

    # Compute pairwise cosine similarities
    n = len(vectors)
    directions = np.stack([v.direction for v in vectors])

    # Normalize for cosine similarity
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    directions_norm = directions / norms

    sim_matrix = directions_norm @ directions_norm.T

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sim_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Cosine Similarity")

    labels = [f"L{v.layer}_P{v.token_position_idx}" for v in vectors]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title("Cosine Similarity Between CAA Directions")

    plt.tight_layout()
    plt.savefig(output_dir / "direction_similarity.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_vector_components(vectors: list[ContrastiveVector], output_dir: Path, top_k: int = 50):
    """Plot top components of selected vectors."""
    # Select a few representative vectors (first, middle, last layer)
    layers = sorted(set(v.layer for v in vectors))
    if len(layers) < 3:
        selected_layers = layers
    else:
        selected_layers = [layers[0], layers[len(layers)//2], layers[-1]]

    selected = [v for v in vectors if v.layer in selected_layers and v.token_position_idx == 0]
    if not selected:
        selected = vectors[:3]

    fig, axes = plt.subplots(1, len(selected), figsize=(5 * len(selected), 4))
    if len(selected) == 1:
        axes = [axes]

    for ax, v in zip(axes, selected):
        direction = v.direction
        # Get top-k by absolute value
        abs_dir = np.abs(direction)
        top_indices = np.argsort(abs_dir)[-top_k:][::-1]

        ax.bar(range(top_k), direction[top_indices], color="steelblue", alpha=0.7)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Component Index (sorted by |value|)")
        ax.set_ylabel("Direction Value")
        ax.set_title(f"Layer {v.layer}, Pos {v.token_position_idx}\nTop {top_k} Components")

    plt.tight_layout()
    plt.savefig(output_dir / "vector_components.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_class_means(vectors: list[ContrastiveVector], output_dir: Path):
    """Plot class mean comparisons."""
    # Select one vector per layer at position 0
    layers = sorted(set(v.layer for v in vectors))
    selected = []
    for layer in layers:
        for v in vectors:
            if v.layer == layer:
                selected.append(v)
                break

    if len(selected) == 0:
        return

    n_plots = min(len(selected), 4)
    fig, axes = plt.subplots(2, n_plots, figsize=(4 * n_plots, 8))
    if n_plots == 1:
        axes = axes.reshape(2, 1)

    for i, v in enumerate(selected[:n_plots]):
        # Scatter of components
        ax = axes[0, i]
        ax.scatter(v.mean_short_term[:500], v.mean_long_term[:500], alpha=0.3, s=5)
        lim = max(abs(v.mean_short_term[:500]).max(), abs(v.mean_long_term[:500]).max())
        ax.plot([-lim, lim], [-lim, lim], "r--", alpha=0.5)
        ax.set_xlabel("Short-term Mean")
        ax.set_ylabel("Long-term Mean")
        ax.set_title(f"Layer {v.layer}, Pos {v.token_position_idx}")

        # Histogram of differences
        ax = axes[1, i]
        diff = v.mean_long_term - v.mean_short_term
        ax.hist(diff, bins=50, alpha=0.7, color="steelblue")
        ax.axvline(0, color="red", linestyle="--")
        ax.set_xlabel("Long - Short")
        ax.set_ylabel("Count")
        ax.set_title(f"Component Differences")

    plt.tight_layout()
    plt.savefig(output_dir / "class_means.png", dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# Extended Geometric Analysis
# =============================================================================


def plot_similarity_to_mean(vectors: list[ContrastiveVector], output_dir: Path):
    """Plot how similar each vector is to the average direction."""
    if len(vectors) < 2:
        return

    directions = np.stack([v.direction for v in vectors])

    # Normalize directions
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    directions_norm = directions / norms

    # Compute mean direction
    mean_direction = directions_norm.mean(axis=0)
    mean_direction = mean_direction / np.linalg.norm(mean_direction)

    # Cosine similarity to mean
    cos_to_mean = directions_norm @ mean_direction

    layers = sorted(set(v.layer for v in vectors))
    positions = sorted(set(v.token_position_idx for v in vectors))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Heatmap
    ax = axes[0]
    sim_matrix = np.zeros((len(layers), len(positions)))
    for idx, v in enumerate(vectors):
        i = layers.index(v.layer)
        j = positions.index(v.token_position_idx)
        sim_matrix[i, j] = cos_to_mean[idx]

    im = ax.imshow(sim_matrix, aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Cosine Sim to Mean")
    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels([f"P{p}" for p in positions], rotation=45)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{l}" for l in layers])
    ax.set_title("Similarity to Mean Direction")

    # Histogram
    ax = axes[1]
    ax.hist(cos_to_mean, bins=20, alpha=0.7, color="steelblue", edgecolor="black")
    ax.axvline(cos_to_mean.mean(), color="red", linestyle="--",
               label=f"Mean: {cos_to_mean.mean():.3f}")
    ax.set_xlabel("Cosine Similarity to Mean Direction")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Similarities")
    ax.legend()

    # By layer
    ax = axes[2]
    layer_means = []
    layer_stds = []
    for layer in layers:
        layer_sims = [cos_to_mean[i] for i, v in enumerate(vectors) if v.layer == layer]
        layer_means.append(np.mean(layer_sims))
        layer_stds.append(np.std(layer_sims))

    ax.errorbar(range(len(layers)), layer_means, yerr=layer_stds,
                fmt="o-", capsize=5, color="steelblue")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Sim to Mean")
    ax.set_title("Similarity to Mean by Layer")
    ax.set_ylim(-1, 1)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / "similarity_to_mean.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_subspace_analysis(vectors: list[ContrastiveVector], output_dir: Path):
    """Analyze if vectors span a low-dimensional subspace using PCA."""
    if len(vectors) < 3:
        return

    directions = np.stack([v.direction for v in vectors])

    # Center the data
    directions_centered = directions - directions.mean(axis=0)

    # SVD for PCA
    U, S, Vt = np.linalg.svd(directions_centered, full_matrices=False)

    # Explained variance
    explained_var = (S ** 2) / (S ** 2).sum()
    cumulative_var = np.cumsum(explained_var)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Scree plot
    ax = axes[0, 0]
    n_show = min(20, len(S))
    ax.bar(range(n_show), explained_var[:n_show], alpha=0.7, color="steelblue", label="Individual")
    ax.plot(range(n_show), cumulative_var[:n_show], "ro-", label="Cumulative")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("PCA Scree Plot")
    ax.legend()
    ax.axhline(0.9, color="gray", linestyle="--", alpha=0.5)

    # Find number of components for 90% variance
    n_90 = np.searchsorted(cumulative_var, 0.9) + 1
    ax.axvline(n_90 - 1, color="red", linestyle="--", alpha=0.5)
    ax.text(n_90, 0.5, f"{n_90} PCs for 90%", fontsize=9)

    # Singular values
    ax = axes[0, 1]
    ax.semilogy(range(n_show), S[:n_show], "o-", color="green")
    ax.set_xlabel("Component Index")
    ax.set_ylabel("Singular Value (log scale)")
    ax.set_title("Singular Value Spectrum")

    # Project onto first 2 PCs
    ax = axes[1, 0]
    proj = U[:, :2] * S[:2]  # Project onto first 2 PCs

    layers = [v.layer for v in vectors]
    unique_layers = sorted(set(layers))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_layers)))

    for layer, color in zip(unique_layers, colors):
        mask = [l == layer for l in layers]
        ax.scatter(proj[mask, 0], proj[mask, 1], c=[color], label=f"L{layer}", s=50, alpha=0.7)

    ax.set_xlabel(f"PC1 ({explained_var[0]:.1%})")
    ax.set_ylabel(f"PC2 ({explained_var[1]:.1%})")
    ax.set_title("Vectors Projected onto First 2 PCs")
    ax.legend(fontsize=8)

    # Project onto PCs 1, 2, 3 - 3D-like view
    ax = axes[1, 1]
    if len(S) >= 3:
        proj3 = U[:, :3] * S[:3]
        positions = [v.token_position_idx for v in vectors]
        unique_pos = sorted(set(positions))
        colors_pos = plt.cm.plasma(np.linspace(0, 1, len(unique_pos)))

        for pos, color in zip(unique_pos, colors_pos):
            mask = [p == pos for p in positions]
            ax.scatter(proj3[mask, 0], proj3[mask, 1], c=[color],
                      s=30 + proj3[mask, 2] * 10, label=f"P{pos}", alpha=0.6)

        ax.set_xlabel(f"PC1 ({explained_var[0]:.1%})")
        ax.set_ylabel(f"PC2 ({explained_var[1]:.1%})")
        ax.set_title(f"Colored by Position (size=PC3)")
        ax.legend(fontsize=6, ncol=2)

    plt.tight_layout()
    plt.savefig(output_dir / "subspace_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Return dimensionality info for stats
    return {
        "n_components_90_var": int(n_90),
        "top_5_explained_var": explained_var[:5].tolist(),
        "cumulative_5": float(cumulative_var[4]) if len(cumulative_var) > 4 else float(cumulative_var[-1]),
    }


def plot_pairwise_angles(vectors: list[ContrastiveVector], output_dir: Path):
    """Plot distribution of pairwise angles between vectors."""
    if len(vectors) < 2:
        return

    directions = np.stack([v.direction for v in vectors])

    # Normalize
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    directions_norm = directions / norms

    # Pairwise cosine similarities
    cos_sim = directions_norm @ directions_norm.T

    # Get upper triangle (excluding diagonal)
    upper_tri = cos_sim[np.triu_indices(len(vectors), k=1)]

    # Convert to angles in degrees
    angles = np.arccos(np.clip(upper_tri, -1, 1)) * 180 / np.pi

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Angle distribution
    ax = axes[0]
    ax.hist(angles, bins=30, alpha=0.7, color="steelblue", edgecolor="black")
    ax.axvline(angles.mean(), color="red", linestyle="--", label=f"Mean: {angles.mean():.1f}°")
    ax.axvline(90, color="gray", linestyle=":", alpha=0.5, label="Orthogonal")
    ax.set_xlabel("Angle (degrees)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Pairwise Angles")
    ax.legend()

    # Cosine similarity distribution
    ax = axes[1]
    ax.hist(upper_tri, bins=30, alpha=0.7, color="coral", edgecolor="black")
    ax.axvline(upper_tri.mean(), color="red", linestyle="--",
               label=f"Mean: {upper_tri.mean():.3f}")
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Pairwise Cosine Similarities")
    ax.legend()

    # Same layer vs different layer
    ax = axes[2]
    same_layer = []
    diff_layer = []
    layers = [v.layer for v in vectors]

    for i in range(len(vectors)):
        for j in range(i+1, len(vectors)):
            if layers[i] == layers[j]:
                same_layer.append(cos_sim[i, j])
            else:
                diff_layer.append(cos_sim[i, j])

    ax.hist(same_layer, bins=20, alpha=0.5, color="steelblue",
            label=f"Same layer (n={len(same_layer)})", density=True)
    ax.hist(diff_layer, bins=20, alpha=0.5, color="coral",
            label=f"Diff layer (n={len(diff_layer)})", density=True)
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.set_title("Within-Layer vs Cross-Layer Similarity")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "pairwise_angles.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_layer_evolution(vectors: list[ContrastiveVector], output_dir: Path):
    """Analyze how vectors change across layers."""
    layers = sorted(set(v.layer for v in vectors))
    positions = sorted(set(v.token_position_idx for v in vectors))

    if len(layers) < 2:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Cosine similarity between consecutive layers (same position)
    ax = axes[0, 0]
    layer_sims = {pos: [] for pos in positions}

    for pos in positions:
        pos_vectors = [v for v in vectors if v.token_position_idx == pos]
        pos_vectors.sort(key=lambda x: x.layer)

        for i in range(len(pos_vectors) - 1):
            v1, v2 = pos_vectors[i], pos_vectors[i+1]
            d1 = v1.direction / np.linalg.norm(v1.direction)
            d2 = v2.direction / np.linalg.norm(v2.direction)
            sim = np.dot(d1, d2)
            layer_sims[pos].append(sim)

    for pos in positions[:6]:  # Show first 6 positions
        if layer_sims[pos]:
            ax.plot(range(len(layer_sims[pos])), layer_sims[pos], "o-",
                   label=f"Pos {pos}", alpha=0.7)

    ax.set_xlabel("Layer Transition")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Similarity Between Consecutive Layers")
    ax.set_xticks(range(len(layers)-1))
    ax.set_xticklabels([f"L{layers[i]}→L{layers[i+1]}" for i in range(len(layers)-1)],
                       rotation=45, fontsize=8)
    ax.legend(fontsize=8)
    ax.set_ylim(-1, 1)

    # Average similarity by layer transition
    ax = axes[0, 1]
    avg_sims = []
    for i in range(len(layers) - 1):
        sims = []
        for pos in positions:
            pos_vectors = sorted([v for v in vectors if v.token_position_idx == pos],
                                key=lambda x: x.layer)
            if len(pos_vectors) > i + 1:
                v1, v2 = pos_vectors[i], pos_vectors[i+1]
                d1 = v1.direction / np.linalg.norm(v1.direction)
                d2 = v2.direction / np.linalg.norm(v2.direction)
                sims.append(np.dot(d1, d2))
        if sims:
            avg_sims.append((np.mean(sims), np.std(sims)))

    if avg_sims:
        means = [x[0] for x in avg_sims]
        stds = [x[1] for x in avg_sims]
        ax.errorbar(range(len(avg_sims)), means, yerr=stds, fmt="o-", capsize=5)
        ax.set_xlabel("Layer Transition")
        ax.set_ylabel("Avg Cosine Similarity")
        ax.set_title("Average Layer-to-Layer Similarity")
        ax.set_xticks(range(len(avg_sims)))
        ax.set_xticklabels([f"L{layers[i]}→L{layers[i+1]}" for i in range(len(avg_sims))],
                          rotation=45, fontsize=8)

    # Norm evolution by layer
    ax = axes[1, 0]
    for pos in positions[:6]:
        pos_vectors = sorted([v for v in vectors if v.token_position_idx == pos],
                            key=lambda x: x.layer)
        norms = [np.linalg.norm(v.mean_long_term - v.mean_short_term) for v in pos_vectors]
        layer_vals = [v.layer for v in pos_vectors]
        ax.plot(layer_vals, norms, "o-", label=f"Pos {pos}", alpha=0.7)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Class Distance (L2 norm)")
    ax.set_title("Class Separation by Layer")
    ax.legend(fontsize=8)

    # First vs last layer comparison
    ax = axes[1, 1]
    first_layer = layers[0]
    last_layer = layers[-1]

    first_vecs = {v.token_position_idx: v for v in vectors if v.layer == first_layer}
    last_vecs = {v.token_position_idx: v for v in vectors if v.layer == last_layer}

    common_pos = set(first_vecs.keys()) & set(last_vecs.keys())
    sims = []
    pos_list = []
    for pos in sorted(common_pos):
        d1 = first_vecs[pos].direction / np.linalg.norm(first_vecs[pos].direction)
        d2 = last_vecs[pos].direction / np.linalg.norm(last_vecs[pos].direction)
        sims.append(np.dot(d1, d2))
        pos_list.append(pos)

    ax.bar(range(len(sims)), sims, color="steelblue", alpha=0.7)
    ax.set_xticks(range(len(sims)))
    ax.set_xticklabels([f"P{p}" for p in pos_list])
    ax.set_xlabel("Position")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title(f"First (L{first_layer}) vs Last (L{last_layer}) Layer")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylim(-1, 1)

    plt.tight_layout()
    plt.savefig(output_dir / "layer_evolution.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_position_analysis(vectors: list[ContrastiveVector], output_dir: Path):
    """Analyze how vectors vary across token positions."""
    layers = sorted(set(v.layer for v in vectors))
    positions = sorted(set(v.token_position_idx for v in vectors))

    if len(positions) < 2:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Similarity between consecutive positions (same layer)
    ax = axes[0, 0]
    for layer in layers:
        layer_vecs = sorted([v for v in vectors if v.layer == layer],
                           key=lambda x: x.token_position_idx)
        sims = []
        for i in range(len(layer_vecs) - 1):
            d1 = layer_vecs[i].direction / np.linalg.norm(layer_vecs[i].direction)
            d2 = layer_vecs[i+1].direction / np.linalg.norm(layer_vecs[i+1].direction)
            sims.append(np.dot(d1, d2))
        if sims:
            ax.plot(range(len(sims)), sims, "o-", label=f"L{layer}", alpha=0.7)

    ax.set_xlabel("Position Transition")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Similarity Between Consecutive Positions")
    ax.legend(fontsize=8)
    ax.set_ylim(-1, 1)

    # Average norm by position
    ax = axes[0, 1]
    pos_norms = {pos: [] for pos in positions}
    for v in vectors:
        norm = np.linalg.norm(v.mean_long_term - v.mean_short_term)
        pos_norms[v.token_position_idx].append(norm)

    means = [np.mean(pos_norms[p]) for p in positions]
    stds = [np.std(pos_norms[p]) for p in positions]
    ax.errorbar(range(len(positions)), means, yerr=stds, fmt="o-", capsize=5, color="green")
    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels([f"P{p}" for p in positions])
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Class Distance (L2)")
    ax.set_title("Average Class Separation by Position")

    # Within-position consistency (how similar are vectors at same position across layers)
    ax = axes[1, 0]
    consistency = []
    for pos in positions:
        pos_vecs = [v for v in vectors if v.token_position_idx == pos]
        if len(pos_vecs) > 1:
            dirs = np.stack([v.direction / np.linalg.norm(v.direction) for v in pos_vecs])
            # Average pairwise similarity
            sim_matrix = dirs @ dirs.T
            upper = sim_matrix[np.triu_indices(len(pos_vecs), k=1)]
            consistency.append(upper.mean())
        else:
            consistency.append(1.0)

    ax.bar(range(len(positions)), consistency, color="purple", alpha=0.7)
    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels([f"P{p}" for p in positions])
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Avg Pairwise Similarity")
    ax.set_title("Within-Position Consistency (across layers)")
    ax.set_ylim(-1, 1)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)

    # Early vs late position comparison
    ax = axes[1, 1]
    early_pos = positions[:len(positions)//2]
    late_pos = positions[len(positions)//2:]

    early_dirs = [v.direction / np.linalg.norm(v.direction)
                  for v in vectors if v.token_position_idx in early_pos]
    late_dirs = [v.direction / np.linalg.norm(v.direction)
                 for v in vectors if v.token_position_idx in late_pos]

    if early_dirs and late_dirs:
        early_mean = np.stack(early_dirs).mean(axis=0)
        early_mean = early_mean / np.linalg.norm(early_mean)
        late_mean = np.stack(late_dirs).mean(axis=0)
        late_mean = late_mean / np.linalg.norm(late_mean)

        sim = np.dot(early_mean, late_mean)
        ax.bar(["Early vs Late"], [sim], color="teal", alpha=0.7)
        ax.set_ylabel("Cosine Similarity")
        ax.set_title(f"Early Positions ({early_pos}) vs Late ({late_pos})")
        ax.set_ylim(-1, 1)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.text(0, sim + 0.05, f"{sim:.3f}", ha="center")

    plt.tight_layout()
    plt.savefig(output_dir / "position_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_component_overlap(vectors: list[ContrastiveVector], output_dir: Path, top_k: int = 100):
    """Analyze which components are consistently important across vectors."""
    if len(vectors) < 2:
        return

    d_model = len(vectors[0].direction)

    # For each vector, get top-k components by absolute value
    top_components = []
    for v in vectors:
        abs_dir = np.abs(v.direction)
        top_k_idx = np.argsort(abs_dir)[-top_k:]
        top_components.append(set(top_k_idx))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Component frequency (how often each dim appears in top-k)
    ax = axes[0, 0]
    component_counts = np.zeros(d_model)
    for tc in top_components:
        for idx in tc:
            component_counts[idx] += 1

    # Show top 50 most frequent
    top_freq = np.argsort(component_counts)[-50:][::-1]
    ax.bar(range(50), component_counts[top_freq], color="steelblue", alpha=0.7)
    ax.set_xlabel("Component (sorted by frequency)")
    ax.set_ylabel(f"Frequency in top-{top_k}")
    ax.set_title(f"Most Frequent Components Across All Vectors")
    ax.axhline(len(vectors) * 0.5, color="red", linestyle="--", alpha=0.5,
               label="50% threshold")
    ax.legend()

    # Jaccard similarity between vectors' top components
    ax = axes[0, 1]
    n = len(vectors)
    jaccard = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            intersection = len(top_components[i] & top_components[j])
            union = len(top_components[i] | top_components[j])
            jaccard[i, j] = intersection / union if union > 0 else 0

    im = ax.imshow(jaccard, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Jaccard Similarity")
    ax.set_title(f"Top-{top_k} Component Overlap (Jaccard)")
    ax.set_xlabel("Vector Index")
    ax.set_ylabel("Vector Index")

    # Component importance by layer
    ax = axes[1, 0]
    layers = sorted(set(v.layer for v in vectors))

    # Average absolute value of each component by layer
    layer_importance = {layer: np.zeros(d_model) for layer in layers}
    layer_counts = {layer: 0 for layer in layers}

    for v in vectors:
        layer_importance[v.layer] += np.abs(v.direction)
        layer_counts[v.layer] += 1

    for layer in layers:
        if layer_counts[layer] > 0:
            layer_importance[layer] /= layer_counts[layer]

    # Show distribution of max importance by layer
    max_by_layer = {layer: layer_importance[layer].max() for layer in layers}
    ax.bar([f"L{l}" for l in layers], [max_by_layer[l] for l in layers],
           color="coral", alpha=0.7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Max Component Importance")
    ax.set_title("Peak Component Importance by Layer")

    # Sparsity analysis
    ax = axes[1, 1]
    sparsities = []
    for v in vectors:
        # Fraction of components with |value| > 0.01 * max
        threshold = 0.01 * np.abs(v.direction).max()
        n_active = (np.abs(v.direction) > threshold).sum()
        sparsities.append(n_active / d_model)

    ax.hist(sparsities, bins=20, alpha=0.7, color="green", edgecolor="black")
    ax.axvline(np.mean(sparsities), color="red", linestyle="--",
               label=f"Mean: {np.mean(sparsities):.1%}")
    ax.set_xlabel("Fraction of Active Components")
    ax.set_ylabel("Count")
    ax.set_title("Vector Sparsity (>1% of max)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "component_overlap.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_magnitude_analysis(vectors: list[ContrastiveVector], output_dir: Path):
    """Analyze magnitudes and norms of vectors."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Pre-normalization norms (class distances)
    ax = axes[0, 0]
    norms = [np.linalg.norm(v.mean_long_term - v.mean_short_term) for v in vectors]
    ax.hist(norms, bins=20, alpha=0.7, color="steelblue", edgecolor="black")
    ax.axvline(np.mean(norms), color="red", linestyle="--",
               label=f"Mean: {np.mean(norms):.2f}")
    ax.set_xlabel("Class Distance (L2)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Class Distances")
    ax.legend()

    # Norm by layer
    ax = axes[0, 1]
    layers = sorted(set(v.layer for v in vectors))
    layer_norms = {layer: [] for layer in layers}
    for v in vectors:
        layer_norms[v.layer].append(np.linalg.norm(v.mean_long_term - v.mean_short_term))

    bp = ax.boxplot([layer_norms[l] for l in layers], labels=[f"L{l}" for l in layers])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Class Distance")
    ax.set_title("Class Distance Distribution by Layer")

    # Mean activation magnitudes
    ax = axes[1, 0]
    short_mags = [np.linalg.norm(v.mean_short_term) for v in vectors]
    long_mags = [np.linalg.norm(v.mean_long_term) for v in vectors]

    ax.scatter(short_mags, long_mags, alpha=0.5, c=[v.layer for v in vectors], cmap="viridis")
    max_mag = max(max(short_mags), max(long_mags))
    ax.plot([0, max_mag], [0, max_mag], "r--", alpha=0.5, label="Equal")
    ax.set_xlabel("||Mean Short-term||")
    ax.set_ylabel("||Mean Long-term||")
    ax.set_title("Class Mean Magnitudes (color=layer)")
    ax.legend()

    # Relative magnitude change
    ax = axes[1, 1]
    rel_change = [(l - s) / s if s > 0 else 0 for s, l in zip(short_mags, long_mags)]
    ax.hist(rel_change, bins=20, alpha=0.7, color="purple", edgecolor="black")
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(np.mean(rel_change), color="red", linestyle="--",
               label=f"Mean: {np.mean(rel_change):.2%}")
    ax.set_xlabel("Relative Change (Long-Short)/Short")
    ax.set_ylabel("Count")
    ax.set_title("Relative Magnitude Change")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "magnitude_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_clustering_analysis(vectors: list[ContrastiveVector], output_dir: Path):
    """Analyze if vectors cluster by layer or position."""
    if len(vectors) < 4:
        return

    directions = np.stack([v.direction for v in vectors])
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions_norm = directions / np.maximum(norms, 1e-8)

    # PCA for visualization
    directions_centered = directions_norm - directions_norm.mean(axis=0)
    U, S, Vt = np.linalg.svd(directions_centered, full_matrices=False)
    proj = U[:, :2] * S[:2]

    layers = np.array([v.layer for v in vectors])
    positions = np.array([v.token_position_idx for v in vectors])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Color by layer
    ax = axes[0, 0]
    scatter = ax.scatter(proj[:, 0], proj[:, 1], c=layers, cmap="viridis", s=60, alpha=0.7)
    plt.colorbar(scatter, ax=ax, label="Layer")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Vectors in PC Space (colored by Layer)")

    # Color by position
    ax = axes[0, 1]
    scatter = ax.scatter(proj[:, 0], proj[:, 1], c=positions, cmap="plasma", s=60, alpha=0.7)
    plt.colorbar(scatter, ax=ax, label="Position")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Vectors in PC Space (colored by Position)")

    # Within-cluster vs between-cluster distances (layer)
    ax = axes[1, 0]
    unique_layers = sorted(set(layers))

    within_layer = []
    between_layer = []

    cos_sim = directions_norm @ directions_norm.T

    for i in range(len(vectors)):
        for j in range(i+1, len(vectors)):
            if layers[i] == layers[j]:
                within_layer.append(cos_sim[i, j])
            else:
                between_layer.append(cos_sim[i, j])

    ax.boxplot([within_layer, between_layer], labels=["Within Layer", "Between Layers"])
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Layer Clustering: Within vs Between")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)

    # Within-cluster vs between-cluster distances (position)
    ax = axes[1, 1]
    within_pos = []
    between_pos = []

    for i in range(len(vectors)):
        for j in range(i+1, len(vectors)):
            if positions[i] == positions[j]:
                within_pos.append(cos_sim[i, j])
            else:
                between_pos.append(cos_sim[i, j])

    ax.boxplot([within_pos, between_pos], labels=["Within Position", "Between Positions"])
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Position Clustering: Within vs Between")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / "clustering_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_summary_statistics(vectors: list[ContrastiveVector], output_dir: Path,
                           subspace_stats: dict = None):
    """Create a text summary of all geometric statistics."""
    directions = np.stack([v.direction for v in vectors])
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions_norm = directions / np.maximum(norms, 1e-8)

    # Mean direction
    mean_dir = directions_norm.mean(axis=0)
    mean_dir = mean_dir / np.linalg.norm(mean_dir)

    # Similarities to mean
    cos_to_mean = directions_norm @ mean_dir

    # Pairwise similarities
    cos_sim = directions_norm @ directions_norm.T
    upper_tri = cos_sim[np.triu_indices(len(vectors), k=1)]

    # Class distances
    class_dists = [np.linalg.norm(v.mean_long_term - v.mean_short_term) for v in vectors]

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis("off")

    text = f"""
CONTRASTIVE VECTORS: GEOMETRIC ANALYSIS SUMMARY
{'='*60}

BASIC INFO:
  Number of vectors: {len(vectors)}
  Dimensions (d_model): {len(vectors[0].direction)}
  Layers: {sorted(set(v.layer for v in vectors))}
  Positions: {sorted(set(v.token_position_idx for v in vectors))}

DIRECTION ALIGNMENT:
  Mean cosine sim to average direction: {cos_to_mean.mean():.4f} ± {cos_to_mean.std():.4f}
  Min similarity to mean: {cos_to_mean.min():.4f}
  Max similarity to mean: {cos_to_mean.max():.4f}

  → Interpretation: {"Vectors are highly aligned" if cos_to_mean.mean() > 0.7 else "Moderate alignment" if cos_to_mean.mean() > 0.3 else "Vectors point in different directions"}

PAIRWISE RELATIONSHIPS:
  Mean pairwise cosine similarity: {upper_tri.mean():.4f} ± {upper_tri.std():.4f}
  Min pairwise similarity: {upper_tri.min():.4f}
  Max pairwise similarity: {upper_tri.max():.4f}
  Mean pairwise angle: {np.arccos(np.clip(upper_tri.mean(), -1, 1)) * 180 / np.pi:.1f}°

  → Interpretation: {"Vectors are clustered" if upper_tri.mean() > 0.5 else "Some diversity" if upper_tri.mean() > 0 else "Vectors oppose each other"}

CLASS SEPARATION (pre-normalization):
  Mean class distance: {np.mean(class_dists):.2f} ± {np.std(class_dists):.2f}
  Min class distance: {np.min(class_dists):.2f}
  Max class distance: {np.max(class_dists):.2f}

SUBSPACE ANALYSIS:
  Components for 90% variance: {subspace_stats.get('n_components_90_var', 'N/A') if subspace_stats else 'N/A'}
  Top 5 PCs explain: {subspace_stats.get('cumulative_5', 0)*100:.1f}% variance

  → Interpretation: {"Low-dimensional structure" if subspace_stats and subspace_stats.get('n_components_90_var', 100) < 10 else "Distributed across many dimensions"}

CONSISTENCY:
  Same-layer pairs more similar than cross-layer: Check clustering_analysis.png
  Early vs late positions: Check position_analysis.png
"""

    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.savefig(output_dir / "summary_statistics.png", dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# Saving
# =============================================================================


def save_vectors(
    vectors: list[ContrastiveVector],
    output: CAAOutput,
    output_dir: Path,
):
    """Save contrastive vectors and metadata."""
    vectors_dir = output_dir / "vectors"
    results_dir = output_dir / "results"
    ensure_dir(vectors_dir)
    ensure_dir(results_dir)

    # Save each vector
    for v in vectors:
        filename = f"caa_layer{v.layer}_pos{v.token_position_idx}.npy"
        np.save(vectors_dir / filename, v.direction)

        # Also save means for analysis
        np.save(vectors_dir / f"mean_short_layer{v.layer}_pos{v.token_position_idx}.npy",
                v.mean_short_term)
        np.save(vectors_dir / f"mean_long_layer{v.layer}_pos{v.token_position_idx}.npy",
                v.mean_long_term)

    # Save results JSON
    stats = analyze_vectors(vectors)

    results = {
        "contrastive_id": output.contrastive_id,
        "query_ids": output.query_ids,
        "model": output.model,
        "timestamp": output.timestamp,
        "n_samples": output.n_samples,
        "n_short_term": output.n_short_term,
        "n_long_term": output.n_long_term,
        "d_model": output.d_model,
        "normalized": output.normalized,
        "layers": output.layers,
        "token_positions": output.token_positions,
        "statistics": stats,
    }

    save_json(results, results_dir / "caa_results.json")

    # Save index for easy loading
    index = {
        "vectors": [
            {
                "layer": v.layer,
                "position": v.token_position_idx,
                "token_position": v.token_position,
                "token_position_word": v.token_position_word,
                "file": f"caa_layer{v.layer}_pos{v.token_position_idx}.npy",
                "n_short": v.n_short_term,
                "n_long": v.n_long_term,
            }
            for v in vectors
        ]
    }
    save_json(index, vectors_dir / "index.json")


# =============================================================================
# Main Pipeline
# =============================================================================


def run_caa_pipeline(
    config: CAAConfig,
    output_base: Path,
) -> CAAOutput:
    """Run the full CAA pipeline."""
    contrastive_id = config.get_id()
    output_dir = output_base / contrastive_id
    viz_dir = output_dir / "viz"
    ensure_dir(viz_dir)

    print(f"Contrastive ID: {contrastive_id}")
    print(f"Output directory: {output_dir}")

    # Load data
    print("\nLoading preference data...")
    data = load_combined_preference_data(config.query_ids)
    print(f"  Model: {data.model}")
    print(f"  Samples: {len(data.samples)}")
    print(f"  Layers: {data.layers}")
    print(f"  Token positions: {len(data.token_position_specs)}")
    print(f"  Hidden dim: {data.d_model}")

    # Get resolved token positions for metadata
    resolved_positions, tokens = get_resolved_token_positions(config.query_ids)
    if resolved_positions:
        print(f"  Resolved {len(resolved_positions)} token positions")

    # Compute contrastive vectors
    print("\nComputing contrastive vectors...")
    vectors = compute_contrastive_vectors(
        data,
        normalize=config.normalize,
        min_samples=config.min_samples_per_class,
        resolved_positions=resolved_positions,
        tokens=tokens,
    )
    print(f"  Computed {len(vectors)} vectors")

    # Build output
    n_short = sum(1 for s in data.samples if s.choice == "short_term")
    n_long = sum(1 for s in data.samples if s.choice == "long_term")

    output = CAAOutput(
        contrastive_id=contrastive_id,
        query_ids=config.query_ids,
        model=data.model,
        timestamp=get_timestamp(),
        n_samples=len(data.samples),
        n_short_term=n_short,
        n_long_term=n_long,
        layers=data.layers,
        token_positions=list(range(len(data.token_position_specs))),
        vectors=vectors,
        d_model=data.d_model,
        normalized=config.normalize,
    )

    # Save vectors
    print("\nSaving vectors...")
    save_vectors(vectors, output, output_dir)

    # Generate visualizations
    print("\nGenerating visualizations...")

    # Basic visualizations
    plot_direction_norms(vectors, viz_dir)
    print("  - direction_norms.png")

    plot_direction_similarity(vectors, viz_dir)
    print("  - direction_similarity.png")

    plot_vector_components(vectors, viz_dir)
    print("  - vector_components.png")

    plot_class_means(vectors, viz_dir)
    print("  - class_means.png")

    # Extended geometric analysis
    print("\nExtended geometric analysis...")

    plot_similarity_to_mean(vectors, viz_dir)
    print("  - similarity_to_mean.png")

    subspace_stats = plot_subspace_analysis(vectors, viz_dir)
    print("  - subspace_analysis.png")

    plot_pairwise_angles(vectors, viz_dir)
    print("  - pairwise_angles.png")

    plot_layer_evolution(vectors, viz_dir)
    print("  - layer_evolution.png")

    plot_position_analysis(vectors, viz_dir)
    print("  - position_analysis.png")

    plot_component_overlap(vectors, viz_dir)
    print("  - component_overlap.png")

    plot_magnitude_analysis(vectors, viz_dir)
    print("  - magnitude_analysis.png")

    plot_clustering_analysis(vectors, viz_dir)
    print("  - clustering_analysis.png")

    plot_summary_statistics(vectors, viz_dir, subspace_stats)
    print("  - summary_statistics.png")

    return output


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Compute Contrastive Activation Addition (CAA) steering vectors"
    )
    parser.add_argument(
        "--query-id",
        help="Single query ID to use (alternative to config file)"
    )
    parser.add_argument(
        "--config",
        help="Config file name (without .json) in scripts/configs/contrastive/"
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Don't L2-normalize the direction vectors"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="Minimum samples per class (default: 10)"
    )
    args = parser.parse_args()

    # Determine config
    if args.config:
        config_path = SCRIPTS_DIR / "configs" / "contrastive" / f"{args.config}.json"
        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)
        config = load_caa_config(config_path)
    elif args.query_id:
        config = CAAConfig(
            query_ids=[args.query_id],
            normalize=not args.no_normalize,
            min_samples_per_class=args.min_samples,
        )
    else:
        # Default to default_contrastive config
        config_path = SCRIPTS_DIR / "configs" / "contrastive" / "default_contrastive.json"
        if not config_path.exists():
            print(f"Error: Default config file not found: {config_path}")
            print("Specify --query-id or --config, or create default_contrastive.json")
            sys.exit(1)
        config = load_caa_config(config_path)

    # Override normalize if specified
    if args.no_normalize:
        config.normalize = False

    output_base = PROJECT_ROOT / "out" / "contrastive"

    print("=" * 60)
    print("Contrastive Activation Addition (CAA)")
    print("=" * 60)
    print(f"Query IDs: {config.query_ids}")
    print(f"Normalize: {config.normalize}")
    print(f"Min samples per class: {config.min_samples_per_class}")

    output = run_caa_pipeline(config, output_base)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Contrastive ID: {output.contrastive_id}")
    print(f"Model: {output.model}")
    print(f"Vectors: {len(output.vectors)}")
    print(f"Output: {output_base / output.contrastive_id}")


if __name__ == "__main__":
    main()
