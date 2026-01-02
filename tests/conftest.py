"""
Pytest configuration and shared fixtures for all tests.
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


# =============================================================================
# Path Fixtures
# =============================================================================


@pytest.fixture
def project_root() -> Path:
    """Get project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def test_configs_dir(project_root: Path) -> Path:
    """Get test configs directory."""
    return project_root / "scripts" / "configs"


@pytest.fixture
def test_outputs_dir(project_root: Path) -> Path:
    """Get test outputs directory."""
    return project_root / "out"


# =============================================================================
# Dataset Config Fixtures
# =============================================================================


@pytest.fixture
def minimal_dataset_config() -> dict:
    """Minimal valid dataset config."""
    return {
        "name": "test_minimal",
        "context": {
            "reward_unit": "dollars",
            "role": "a decision maker",
            "situation": "Choose between options.",
            "action_in_question": "select",
            "reasoning_ask": "why you chose this",
            "domain": "test",
            "labels": ["a)", "b)"],
            "method": "grid",
            "seed": 42
        },
        "options": {
            "short_term": {
                "reward_range": [100, 100],
                "time_range": [[1, "months"], [1, "months"]],
                "reward_steps": [0, "linear"],
                "time_steps": [0, "linear"]
            },
            "long_term": {
                "reward_range": [500, 500],
                "time_range": [[1, "years"], [1, "years"]],
                "reward_steps": [0, "linear"],
                "time_steps": [0, "linear"]
            }
        },
        "time_horizons": [None],
        "add_formatting_variations": False
    }


@pytest.fixture
def formatting_config() -> dict:
    """Default formatting config."""
    return {
        "question_template": "Choose: [LEFT_TERM_LABEL] [LEFT_TERM_REWARD] in [LEFT_TERM_TIME] or [RIGHT_TERM_LABEL] [RIGHT_TERM_REWARD] in [RIGHT_TERM_TIME]",
        "response_format": "\nRespond: [CHOICE_PREFIX] <choice>",
        "choice_prefix": "I choose:",
        "time_horizon_spec": "",
        "max_reasoning_length": "1 sentence"
    }


# =============================================================================
# Sample Data Fixtures
# =============================================================================


@pytest.fixture
def sample_preference_pair() -> dict:
    """Sample preference pair for testing."""
    return {
        "short_term": {
            "label": "a)",
            "time": [3, "months"],
            "reward": 100.0
        },
        "long_term": {
            "label": "b)",
            "time": [1, "years"],
            "reward": 500.0
        }
    }


@pytest.fixture
def sample_time_values() -> list:
    """Sample time values for testing conversions."""
    return [
        ([1, "years"], 1.0),
        ([12, "months"], 1.0),
        ([365, "days"], 1.0),
        ([52.14, "weeks"], 1.0),
        ([0.1, "decades"], 1.0),
    ]


# =============================================================================
# Test Data Directory Helpers
# =============================================================================


def get_test_dataset_path(name: str = "minimal") -> Path:
    """Get path to a test dataset fixture."""
    return PROJECT_ROOT / "out" / "datasets" / "test" / f"{name}.json"


def get_test_preference_data_path(name: str = "minimal") -> Path:
    """Get path to test preference data fixture."""
    return PROJECT_ROOT / "out" / "preference_data" / "test" / f"{name}.json"


def get_test_config_path(config_type: str, name: str) -> Path:
    """Get path to a test config file."""
    return PROJECT_ROOT / "scripts" / "configs" / config_type / "test" / f"{name}.json"
