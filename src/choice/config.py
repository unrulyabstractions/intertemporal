"""Configuration for choice analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.common.io import load_json
from src.common.schema_utils import SchemaClass


@dataclass
class AnalysisConfig(SchemaClass):
    """Configuration for choice analysis.

    Extends SchemaClass so that get_id() returns a deterministic hash
    based on all config parameters (query_ids, train_test_split, etc.).
    """

    train_query_ids: list[str]
    test_query_ids: list[str]
    train_test_query_ids: list[str]  # For train_test mode
    train_test_split: float  # Split ratio (default 0.7)
    learning_rate: float = 0.01
    num_iterations: int = 100
    temperature: float = 0.0  # Deterministic by default


def load_analysis_config(path: Path) -> AnalysisConfig:
    """Load analysis config from JSON file."""
    data = load_json(path)

    # Support query_ids format (like default_probes.json)
    query_ids = data.get("query_ids", {})
    train_query_ids = query_ids.get("train_data", [])
    test_query_ids = query_ids.get("test_data", [])
    train_test_query_ids = query_ids.get("train_test", [])
    train_test_split = data.get("train_test_split", 0.7)

    return AnalysisConfig(
        train_query_ids=train_query_ids,
        test_query_ids=test_query_ids,
        train_test_query_ids=train_test_query_ids,
        train_test_split=train_test_split,
        # learning_rate, num_iterations, temperature are set via CLI args
    )
