"""
Utilities for finding and loading configs by schema IDs.

The ID chain:
- choice_modeling config has query_id
- preference_data/{name}_{model}_{query_id}.json has dataset_id in metadata
- datasets/{name}_{dataset_id}.json has full dataset config with time ranges
"""

from pathlib import Path
from typing import Optional

from src.common.io import load_json

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


def find_preference_data_by_query_id(
    query_id: str,
    name: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[Path]:
    """
    Find preference data file by query_id.

    Args:
        query_id: Query config ID
        name: Optional dataset name to narrow search
        model: Optional model name to narrow search

    Returns:
        Path to preference data file, or None if not found
    """
    data_dir = PROJECT_ROOT / "out" / "preference_data"
    if not data_dir.exists():
        return None

    # Build pattern
    if name and model:
        pattern = f"{name}_{model}_{query_id}.json"
    elif name:
        pattern = f"{name}_*_{query_id}.json"
    else:
        pattern = f"*_{query_id}.json"

    matches = list(data_dir.glob(pattern))
    if matches:
        return matches[0]

    # Fallback: search all files for query_id in filename
    for f in data_dir.glob("*.json"):
        if query_id in f.name:
            return f

    return None


def find_dataset_by_id(dataset_id: str, name: Optional[str] = None) -> Optional[Path]:
    """
    Find dataset file by dataset_id.

    Args:
        dataset_id: Dataset config ID
        name: Optional dataset name to narrow search

    Returns:
        Path to dataset file, or None if not found
    """
    data_dir = PROJECT_ROOT / "out" / "datasets"
    if not data_dir.exists():
        return None

    # Build pattern
    if name:
        pattern = f"{name}_{dataset_id}.json"
    else:
        pattern = f"*_{dataset_id}.json"

    matches = list(data_dir.glob(pattern))
    if matches:
        return matches[0]

    return None


def get_dataset_id_from_preference_data(preference_data_path: Path) -> Optional[str]:
    """
    Extract dataset_id from preference data metadata.

    Args:
        preference_data_path: Path to preference data JSON

    Returns:
        Dataset ID string, or None if not found
    """
    try:
        data = load_json(preference_data_path)
        metadata = data.get("metadata", {})
        return metadata.get("dataset_id")
    except Exception:
        return None


def load_dataset_config(dataset_path: Path) -> Optional[dict]:
    """
    Load dataset config from dataset output file.

    Args:
        dataset_path: Path to dataset JSON

    Returns:
        Dataset config dict, or None if not found
    """
    try:
        data = load_json(dataset_path)
        metadata = data.get("metadata", {})
        return metadata.get("config")
    except Exception:
        return None


def get_time_ranges_from_dataset(dataset_config: dict) -> dict:
    """
    Extract time ranges from dataset config.

    Args:
        dataset_config: Dataset config dict

    Returns:
        Dict with short_term and long_term time ranges in years:
        {
            "short_term": {"min": float, "max": float},
            "long_term": {"min": float, "max": float}
        }
    """
    from src.types import TimeValue

    options = dataset_config.get("options", {})

    result = {}

    for key in ["short_term", "long_term"]:
        opt = options.get(key, {})
        time_range = opt.get("time_range", [])

        # time_range is a list: [[min_value, min_unit], [max_value, max_unit]]
        if isinstance(time_range, list) and len(time_range) >= 2:
            min_time = time_range[0]  # First element is min
            max_time = time_range[1]  # Second element is max
        else:
            # Fallback defaults
            min_time = [0, "years"]
            max_time = [100, "years"]

        min_tv = TimeValue(min_time[0], min_time[1])
        max_tv = TimeValue(max_time[0], max_time[1])

        result[key] = {
            "min": min_tv.to_years(),
            "max": max_tv.to_years(),
        }

    return result


def categorize_time_horizon(
    horizon_years: float,
    time_ranges: dict,
) -> str:
    """
    Categorize time horizon relative to option time ranges.

    Categories:
    - "below_short": horizon < short_term.min
    - "within_short": short_term.min <= horizon <= short_term.max
    - "between": short_term.max < horizon < long_term.min
    - "within_long": long_term.min <= horizon <= long_term.max
    - "above_long": horizon > long_term.max

    Args:
        horizon_years: Time horizon in years
        time_ranges: Dict from get_time_ranges_from_dataset()

    Returns:
        Category string
    """
    short = time_ranges["short_term"]
    long = time_ranges["long_term"]

    if horizon_years < short["min"]:
        return "below_short"
    elif horizon_years <= short["max"]:
        return "within_short"
    elif horizon_years < long["min"]:
        return "between"
    elif horizon_years <= long["max"]:
        return "within_long"
    else:
        return "above_long"


def get_expected_choice(category: str) -> Optional[str]:
    """
    Get expected choice based on time horizon category.

    Args:
        category: Category from categorize_time_horizon()

    Returns:
        "short_term", "long_term", or None (ambiguous)
    """
    if category in ("below_short", "within_short"):
        return "short_term"
    elif category in ("within_long", "above_long"):
        return "long_term"
    else:  # "between"
        return None  # Ambiguous


def resolve_dataset_config_from_query_id(
    query_id: str,
    name: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[dict]:
    """
    Resolve full dataset config from query_id by following the ID chain.

    query_id -> preference_data -> dataset_id -> dataset config

    Args:
        query_id: Query config ID
        name: Optional dataset name
        model: Optional model name

    Returns:
        Dataset config dict, or None if not found
    """
    # Find preference data
    pref_path = find_preference_data_by_query_id(query_id, name, model)
    if not pref_path:
        return None

    # Get dataset_id
    dataset_id = get_dataset_id_from_preference_data(pref_path)
    if not dataset_id:
        return None

    # Find dataset
    dataset_path = find_dataset_by_id(dataset_id, name)
    if not dataset_path:
        return None

    # Load config
    return load_dataset_config(dataset_path)
