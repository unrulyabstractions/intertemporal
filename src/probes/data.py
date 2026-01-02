"""
Data loading and preparation for probe training.

Handles:
- Loading preference data by query_id
- Loading and matching activation files
- Building datasets for each (layer, token_position) combination
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.io import load_json


# =============================================================================
# Constants
# =============================================================================

PREFERENCE_DATA_DIR = PROJECT_ROOT / "out" / "preference_data"
INTERNALS_DIR = PROJECT_ROOT / "out" / "internals"


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class TokenPositionSpec:
    """
    Abstract token position specification from config.

    Attributes:
        index: Index p in metadata.internals.token_positions list
        spec: The raw spec dict (e.g., {"text": "I select:", "location": "continuation"})
    """
    index: int
    spec: dict


@dataclass
class ActivationSample:
    """
    Activation data for one sample.

    Attributes:
        sample_id: ID of the sample
        activations: Dict mapping (layer, token_pos_idx) to activation vector
        choice: "short_term", "long_term", or "unknown"
        time_horizon: [value, unit] list or None if no time horizon
    """
    sample_id: int
    activations: dict[tuple[int, int], np.ndarray]  # (layer, token_pos_idx) -> vector
    choice: str
    time_horizon: Optional[list]


@dataclass
class ProbeDataset:
    """
    Dataset for training a single probe at (layer, token_position_idx).

    Attributes:
        layer: Layer index
        token_position_idx: Abstract token position index p
        X: Activation matrix (n_samples, d_model)
        sample_ids: Array of sample IDs
    """
    layer: int
    token_position_idx: int
    X: np.ndarray
    sample_ids: np.ndarray


@dataclass
class ChoiceLabels:
    """
    Binary classification labels: 0=short_term, 1=long_term.

    Attributes:
        y: Label array (n_samples,)
        valid_mask: Boolean mask, True for non-unknown choices
    """
    y: np.ndarray
    valid_mask: np.ndarray


@dataclass
class TimeHorizonCategoryLabels:
    """
    Binary classification labels for time horizon: 0=short (<=1yr), 1=long (>1yr).

    Attributes:
        y: Label array (n_samples,)
        valid_mask: Boolean mask, True for non-null time_horizon
    """
    y: np.ndarray
    valid_mask: np.ndarray


@dataclass
class TimeHorizonValueLabels:
    """
    Regression labels for time horizon in months.

    Attributes:
        y: Time horizon values in months (n_samples,)
        valid_mask: Boolean mask, True for non-null time_horizon
    """
    y: np.ndarray
    valid_mask: np.ndarray


@dataclass
class CombinedPreferenceData:
    """
    Combined preference data from multiple query_ids.

    Attributes:
        samples: List of ActivationSample objects
        layers: List of layer indices to use
        token_position_specs: List of TokenPositionSpec objects
        model: Model name (from first file)
        d_model: Model hidden dimension
    """
    samples: list[ActivationSample]
    layers: list[int]
    token_position_specs: list[TokenPositionSpec]
    model: str
    d_model: int


# =============================================================================
# Utility Functions
# =============================================================================


def time_to_months(time_horizon: list) -> float:
    """Convert [value, unit] to months."""
    value, unit = time_horizon
    if unit in ("month", "months"):
        return float(value)
    elif unit in ("year", "years"):
        return float(value) * 12
    elif unit in ("day", "days"):
        return float(value) / 30
    else:
        raise ValueError(f"Unknown time unit: {unit}")


def categorize_time_horizon(time_horizon: list) -> int:
    """
    Categorize time horizon as short (0) or long (1).

    Short: <= 1 year (12 months)
    Long: > 1 year
    """
    months = time_to_months(time_horizon)
    return 0 if months <= 12 else 1


# =============================================================================
# Data Loading
# =============================================================================


def find_preference_data_by_query_id(query_id: str) -> Optional[Path]:
    """
    Find preference data file by query_id suffix.

    Args:
        query_id: Query ID to search for (file ends with _{query_id}.json)

    Returns:
        Path to file, or None if not found
    """
    if not PREFERENCE_DATA_DIR.exists():
        return None

    pattern = f"*_{query_id}.json"
    matches = list(PREFERENCE_DATA_DIR.glob(pattern))

    if matches:
        return matches[0]
    return None


def load_preference_data_file(path: Path) -> dict:
    """Load preference data JSON file."""
    return load_json(path)


def extract_layers_from_config(query_config: dict) -> list[int]:
    """
    Extract layer indices from query config's internals section.

    Args:
        query_config: Query configuration dict

    Returns:
        List of layer indices (resolved for negative indices)
    """
    internals = query_config.get("internals", {})
    activations = internals.get("activations", {})

    # Look for resid_post layers (most common)
    resid_post = activations.get("resid_post", {})
    layers = resid_post.get("layers", [])

    return layers


def extract_token_position_specs(query_config: dict) -> list[TokenPositionSpec]:
    """
    Extract token position specs from query config.

    Args:
        query_config: Query configuration dict

    Returns:
        List of TokenPositionSpec objects
    """
    internals = query_config.get("internals", {})
    token_positions = internals.get("token_positions", [])

    specs = []
    for i, spec in enumerate(token_positions):
        specs.append(TokenPositionSpec(index=i, spec=spec))

    return specs


def load_activation_file(internals_ref: dict, project_root: Path = PROJECT_ROOT) -> dict:
    """
    Load activation file from InternalsReference.

    Args:
        internals_ref: InternalsReference dict with file_path key
        project_root: Project root for resolving relative paths

    Returns:
        Dict mapping activation names to tensors
    """
    file_path = project_root / internals_ref["file_path"]
    if not file_path.exists():
        raise FileNotFoundError(f"Activation file not found: {file_path}")

    return torch.load(file_path, map_location="cpu")


def parse_activation_key(key: str) -> tuple[int, int]:
    """
    Parse activation key like "blocks.8.hook_resid_post_pos45" into (layer, position).

    Args:
        key: Activation key string

    Returns:
        Tuple of (layer_index, position_index)
    """
    # Format: blocks.{layer}.hook_resid_post_pos{position}
    parts = key.split(".")
    layer = int(parts[1])

    # Extract position from last part (format: hook_resid_post_pos{N})
    last_part = parts[-1]
    pos_idx = last_part.rindex("_pos")  # Use rindex to find the last "_pos"
    position = int(last_part[pos_idx + 4:])

    return layer, position


def load_combined_preference_data(query_ids: list[str]) -> CombinedPreferenceData:
    """
    Load and combine preference data from multiple query_ids.

    Args:
        query_ids: List of query IDs to load

    Returns:
        CombinedPreferenceData with all samples combined

    Raises:
        FileNotFoundError: If any query_id file is not found
        ValueError: If token position specs don't match across files
    """
    all_samples: list[ActivationSample] = []
    layers: Optional[list[int]] = None
    token_specs: Optional[list[TokenPositionSpec]] = None
    model_name: Optional[str] = None
    d_model: Optional[int] = None

    for query_id in query_ids:
        path = find_preference_data_by_query_id(query_id)
        if path is None:
            raise FileNotFoundError(f"No preference data found for query_id: {query_id}")

        data = load_preference_data_file(path)
        metadata = data["metadata"]
        query_config = metadata.get("query_config", {})

        # Extract configuration from first file
        if layers is None:
            layers = extract_layers_from_config(query_config)
        if token_specs is None:
            token_specs = extract_token_position_specs(query_config)
        if model_name is None:
            model_name = metadata.get("model", "unknown")

        # Validate token positions match
        current_specs = extract_token_position_specs(query_config)
        if len(current_specs) != len(token_specs):
            raise ValueError(
                f"Token position count mismatch for query_id {query_id}: "
                f"expected {len(token_specs)}, got {len(current_specs)}"
            )

        # Load samples
        for pref in data["preferences"]:
            sample_id = pref["sample_id"]
            choice = pref["choice"]
            time_horizon = pref.get("time_horizon")

            # Load activations if available
            activations = {}
            if pref.get("internals"):
                try:
                    activation_data = load_activation_file(pref["internals"])

                    # Parse activation keys and build dict
                    for key, tensor in activation_data.items():
                        if "hook_resid_post" in key:
                            layer_idx, pos = parse_activation_key(key)

                            # Find which token_position_idx this corresponds to
                            # The position in the key is the resolved position
                            # We need to map back to token_position_idx based on order
                            internals_ref = pref["internals"]
                            token_positions = internals_ref.get("token_positions", [])

                            # Find the index in token_positions that matches this resolved position
                            # Skip -1 placeholders (unresolved positions)
                            for tp_idx, tp in enumerate(token_positions):
                                if tp >= 0 and tp == pos:
                                    # Convert to numpy array (handles both tensors and lists)
                                    if hasattr(tensor, 'numpy'):
                                        arr = tensor.numpy()
                                    else:
                                        arr = np.array(tensor)
                                    activations[(layer_idx, tp_idx)] = arr
                                    if d_model is None:
                                        d_model = arr.shape[-1]
                                    break
                except FileNotFoundError:
                    # Skip samples with missing activation files
                    continue

            if activations:  # Only add samples that have activations
                all_samples.append(ActivationSample(
                    sample_id=sample_id,
                    activations=activations,
                    choice=choice,
                    time_horizon=time_horizon,
                ))

    if not all_samples:
        raise ValueError("No samples with activations found")

    return CombinedPreferenceData(
        samples=all_samples,
        layers=layers or [],
        token_position_specs=token_specs or [],
        model=model_name or "unknown",
        d_model=d_model or 0,
    )


def class_balanced_train_test_split(
    data: CombinedPreferenceData,
    train_ratio: float = 0.7,
    random_seed: int = 42,
) -> tuple[CombinedPreferenceData, CombinedPreferenceData]:
    """
    Split data into train/test with class balance across choice and time_horizon.

    Stratifies by:
    - choice: short_term, long_term, unknown
    - time_horizon: null vs non-null

    Args:
        data: Combined preference data to split
        train_ratio: Fraction for training (default 0.7)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_data, test_data)
    """
    import random

    random.seed(random_seed)

    # Group samples by (choice, has_time_horizon)
    groups: dict[tuple[str, bool], list[ActivationSample]] = {}
    for sample in data.samples:
        has_th = sample.time_horizon is not None
        key = (sample.choice, has_th)
        if key not in groups:
            groups[key] = []
        groups[key].append(sample)

    train_samples = []
    test_samples = []

    # Split each group proportionally
    for key, samples in groups.items():
        random.shuffle(samples)
        n_train = int(len(samples) * train_ratio)
        # Ensure at least 1 in each split if we have enough samples
        if len(samples) >= 2:
            n_train = max(1, min(n_train, len(samples) - 1))

        train_samples.extend(samples[:n_train])
        test_samples.extend(samples[n_train:])

    # Shuffle final sets
    random.shuffle(train_samples)
    random.shuffle(test_samples)

    train_data = CombinedPreferenceData(
        samples=train_samples,
        layers=data.layers,
        token_position_specs=data.token_position_specs,
        model=data.model,
        d_model=data.d_model,
    )

    test_data = CombinedPreferenceData(
        samples=test_samples,
        layers=data.layers,
        token_position_specs=data.token_position_specs,
        model=data.model,
        d_model=data.d_model,
    )

    return train_data, test_data


# =============================================================================
# Dataset Building
# =============================================================================


def build_probe_datasets(
    data: CombinedPreferenceData,
) -> dict[tuple[int, int], ProbeDataset]:
    """
    Build ProbeDataset for each (layer, token_position_idx) combination.

    Args:
        data: Combined preference data

    Returns:
        Dict mapping (layer, token_pos_idx) to ProbeDataset
    """
    datasets: dict[tuple[int, int], ProbeDataset] = {}

    # Collect all unique (layer, token_pos_idx) combinations
    all_keys: set[tuple[int, int]] = set()
    for sample in data.samples:
        all_keys.update(sample.activations.keys())

    for key in sorted(all_keys):
        layer, tp_idx = key

        # Collect samples that have this activation
        X_list = []
        sample_ids = []

        for sample in data.samples:
            if key in sample.activations:
                X_list.append(sample.activations[key])
                sample_ids.append(sample.sample_id)

        if X_list:
            datasets[key] = ProbeDataset(
                layer=layer,
                token_position_idx=tp_idx,
                X=np.stack(X_list),
                sample_ids=np.array(sample_ids),
            )

    return datasets


def build_choice_labels(data: CombinedPreferenceData) -> ChoiceLabels:
    """
    Build choice labels for binary classification.

    Args:
        data: Combined preference data

    Returns:
        ChoiceLabels with y and valid_mask
    """
    y = []
    valid_mask = []

    for sample in data.samples:
        if sample.choice == "short_term":
            y.append(0)
            valid_mask.append(True)
        elif sample.choice == "long_term":
            y.append(1)
            valid_mask.append(True)
        else:  # unknown
            y.append(0)  # Placeholder
            valid_mask.append(False)

    return ChoiceLabels(
        y=np.array(y),
        valid_mask=np.array(valid_mask),
    )


def build_time_horizon_category_labels(data: CombinedPreferenceData) -> TimeHorizonCategoryLabels:
    """
    Build time horizon category labels: 0=short (<=1yr), 1=long (>1yr).

    Args:
        data: Combined preference data

    Returns:
        TimeHorizonCategoryLabels with y and valid_mask
    """
    y = []
    valid_mask = []

    for sample in data.samples:
        if sample.time_horizon is not None:
            category = categorize_time_horizon(sample.time_horizon)
            y.append(category)
            valid_mask.append(True)
        else:
            y.append(0)  # Placeholder
            valid_mask.append(False)

    return TimeHorizonCategoryLabels(
        y=np.array(y),
        valid_mask=np.array(valid_mask),
    )


def build_time_horizon_value_labels(data: CombinedPreferenceData) -> TimeHorizonValueLabels:
    """
    Build time horizon value labels in months.

    Args:
        data: Combined preference data

    Returns:
        TimeHorizonValueLabels with y and valid_mask
    """
    y = []
    valid_mask = []

    for sample in data.samples:
        if sample.time_horizon is not None:
            months = time_to_months(sample.time_horizon)
            y.append(months)
            valid_mask.append(True)
        else:
            y.append(0.0)  # Placeholder
            valid_mask.append(False)

    return TimeHorizonValueLabels(
        y=np.array(y),
        valid_mask=np.array(valid_mask),
    )
