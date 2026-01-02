"""
Base I/O utilities for saving and loading data.

Provides core JSON/JSONL utilities. Output-specific functions are in scripts/io.py.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


# =============================================================================
# Base I/O Functions
# =============================================================================


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _make_text_readable(obj):
    """Recursively convert long text fields to arrays of lines for readability."""
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if k in ("text", "raw_text", "trace") and isinstance(v, str) and "\n" in v:
                # Convert multiline text to array of lines
                result[k] = v.split("\n")
            else:
                result[k] = _make_text_readable(v)
        return result
    elif isinstance(obj, list):
        return [_make_text_readable(item) for item in obj]
    else:
        return obj


def save_json(data, path: Path, readable_text: bool = True) -> None:
    """Save dictionary as pretty JSON."""
    if readable_text:
        data = _make_text_readable(data)
    with open(path, "w") as f:
        json.dump(data, f, indent=4, default=str, ensure_ascii=False)


def _restore_text_fields(obj):
    """Recursively restore text fields from arrays back to strings."""
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if k in ("text", "raw_text", "trace") and isinstance(v, list):
                # Join array of lines back to string
                result[k] = "\n".join(v)
            else:
                result[k] = _restore_text_fields(v)
        return result
    elif isinstance(obj, list):
        return [_restore_text_fields(item) for item in obj]
    else:
        return obj


def load_json(path: Path) -> dict:
    """Load JSON file."""
    with open(path) as f:
        data = json.load(f)
    return _restore_text_fields(data)


def save_jsonl(items: list[dict], path: Path) -> None:
    """Save list of dicts as JSONL."""
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item, default=str) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file."""
    items = []
    with open(path) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


# =============================================================================
# Legacy Dataset Loading (uses internal types only)
# =============================================================================


def load_dataset(dataset_dir: Path):
    """
    Load dataset from directory (legacy format).

    Sample JSON for samples.jsonl (see schema: DatasetSample):
    {"id": 0, "prompt": {...}, "response": null, "domain": "housing"}

    Sample JSON for metadata.json (see schema: DatasetMetadata):
    {
        "config_name": "cityhousing",
        "domain": "housing",
        "num_samples": 8,
        "time_horizons": [[5, "months"], [15, "years"]],
        "seed": 42
    }

    Args:
        dataset_dir: Directory containing samples.jsonl and metadata.json

    Returns:
        Tuple of (samples, metadata)
    """
    from ..types import (
        DatasetSample,
        DatasetMetadata,
        Prompt,
        PreferenceQuestion,
        PreferencePair,
        IntertemporalOption,
        TimeValue,
        RewardValue,
        Response,
    )

    # Load samples
    samples_path = dataset_dir / "samples.jsonl"
    samples_data = load_jsonl(samples_path)

    samples = []
    for s in samples_data:
        # Reconstruct nested objects
        q = s["prompt"]["question"]
        pair = q["pair"]

        short_term = IntertemporalOption(
            label=pair["short_term"]["label"],
            time=TimeValue(**pair["short_term"]["time"]),
            reward=RewardValue(**pair["short_term"]["reward"]),
        )
        long_term = IntertemporalOption(
            label=pair["long_term"]["label"],
            time=TimeValue(**pair["long_term"]["time"]),
            reward=RewardValue(**pair["long_term"]["reward"]),
        )

        question = PreferenceQuestion(
            pair=PreferencePair(short_term=short_term, long_term=long_term),
            time_horizon=TimeValue(**q["time_horizon"]),
        )

        prompt = Prompt(
            question=question,
            context=s["prompt"]["context"],
            text=s["prompt"]["text"],
            response_format=s["prompt"]["response_format"],
        )

        response = None
        if s.get("response"):
            response = Response(**s["response"])

        samples.append(
            DatasetSample(
                id=s["id"],
                prompt=prompt,
                response=response,
                domain=s.get("domain", ""),
            )
        )

    # Load metadata
    metadata_path = dataset_dir / "metadata.json"
    meta_data = load_json(metadata_path)

    time_horizons = [TimeValue(**t) for t in meta_data.get("time_horizons", [])]
    metadata = DatasetMetadata(
        config_name=meta_data["config_name"],
        domain=meta_data["domain"],
        num_samples=meta_data["num_samples"],
        time_horizons=time_horizons,
        seed=meta_data.get("seed", 42),
        description=meta_data.get("description", ""),
    )

    return samples, metadata
