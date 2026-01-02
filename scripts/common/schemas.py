"""
Output schemas for script results.

This file contains schemas that correspond to script output JSON files:
- generate_dataset.py outputs
- query_model.py outputs
- train_choice_model.py outputs
- test_choice_model.py outputs
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.schema_utils import SchemaClass
from src.common.schemas import SCHEMA_VERSION
from src.types import (
    Prompt,
    Response,
    TrainingResult,
    ValueFunctionParams,
)


# Re-export SCHEMA_VERSION for convenience
__all__ = ["SCHEMA_VERSION"]


# =============================================================================
# Dataset Output Schemas (from generate_dataset.py)
# =============================================================================


@dataclass
class OptionOutput(SchemaClass):
    """
    Option in output format.

    Sample JSON:
    { "label": "a", "time": [3, "months"], "reward": 2000 }
    """

    label: str
    time: list  # [value, unit]
    reward: float


@dataclass
class PreferencePairOutput(SchemaClass):
    """
    Preference pair in output format.

    Sample JSON:
    {
        "short_term": { <OptionOutput> },
        "long_term": { <OptionOutput> }
    }
    """

    short_term: OptionOutput
    long_term: OptionOutput


@dataclass
class QuestionOutput(SchemaClass):
    """
    Question in output format.

    Sample JSON:
    {
        "sample_id": 0,
        "question_text": "Situation: Plan for housing...",
        "time_horizon": [5, "months"],  // or null if no time horizon
        "preference_pair": { <PreferencePairOutput> }
    }
    """

    sample_id: int
    question_text: str
    time_horizon: list | None  # [value, unit] or None if no time horizon
    preference_pair: PreferencePairOutput


@dataclass
class DatasetOutputMetadata(SchemaClass):
    """
    Metadata for generated dataset.

    Sample JSON:
    {
        "version": "1.0",
        "dataset_id": "79642578326872f28277b0f349873061",
        "dataset_run_id": "a1b2c3d4e5f6...",
        "config": { <DatasetConfig as dict> },
        "num_questions": 8,
        "timestamp": "20251229_195540"
    }
    """

    version: str
    dataset_id: str
    config: dict  # Full DatasetConfig as dict
    num_questions: int
    timestamp: str = ""
    dataset_run_id: str = ""  # Shared across all datasets from same --config invocation (optional)


@dataclass
class DatasetOutput(SchemaClass):
    """
    Full dataset output format (output of generate_dataset.py).

    Sample JSON:
    {
        "metadata": { <DatasetOutputMetadata> },
        "questions": [ <QuestionOutput>, ... ]
    }
    """

    metadata: DatasetOutputMetadata
    questions: list[QuestionOutput]


@dataclass
class GenerateDatasetOutput(SchemaClass):
    """Output summary from dataset generation script (not saved to file)."""

    dataset_id: str
    num_samples: int
    num_horizons: int
    output_path: str
    timestamp: str = ""


# =============================================================================
# Preference Data Output Schemas (from query_llm.py)
# =============================================================================


@dataclass
class PreferenceDataMetadata(SchemaClass):
    """
    Metadata for preference data output.

    Sample JSON:
    {
        "version": "1.0",
        "dataset_id": "5584270069cdc0e03965fdb0e4f5e10b",
        "formatting_id": "a1b2c3d4e5f6...",
        "query_run_id": "c3d4e5f6a7b8...",
        "query_id": "f7e8d9c0b1a2...",
        "model": "gpt2",
        "query_config": { <QueryConfig as dict> },
        "sample_prompt": "SITUATION: Plan for housing...",
        "sample_continuation": "I select: a) ..."
    }
    """

    version: str
    dataset_id: str
    formatting_id: str
    query_id: str  # Unique per model+dataset+params (SingleQuerySpec.get_id())
    model: str
    query_config: dict  # Full QueryConfig as dict for reproducibility
    query_run_id: str = ""  # Shared across all queries from same config (optional)
    sample_prompt: str = ""  # Example prompt for reference
    sample_continuation: str = ""  # Example continuation for reference


@dataclass
class PreferenceDebugInfo(SchemaClass):
    """
    Debug information for a preference query.

    Always saved for reproducibility and analysis.

    Sample JSON:
    {
        "raw_prompt": "Situation: Plan for housing...",
        "raw_continuation": "I choose: a. The reasoning...",
        "parsed_label": "a",
        "num_prompt_tokens": 150,
        "num_continuation_tokens": 45
    }
    """

    raw_prompt: str
    raw_continuation: str
    parsed_label: Optional[str] = None
    num_prompt_tokens: int = 0
    num_continuation_tokens: int = 0


@dataclass
class InternalsReference(SchemaClass):
    """
    Reference to saved internals file with metadata.

    Sample JSON:
    {
        "file_path": "out/internals/cityhousing_Qwen2.5-7B-Instruct_a1b2c3d4_sample_0.pt",
        "activations": ["blocks.8.hook_resid_post_pos45", "blocks.14.hook_resid_post_pos45"],
        "token_positions": [0, 45],
        "tokens": ["<bos>", "choose"]
    }

    Note: file_path is relative to repo root.
    """

    file_path: str  # Relative path to saved .pt file (from repo root)
    activations: list[str]  # Names of captured activations
    token_positions: list[int]  # Token positions captured
    tokens: list[str]  # Token strings at positions


@dataclass
class PreferenceItem(SchemaClass):
    """
    Single preference item from model query.

    Sample JSON:
    {
        "sample_id": 0,
        "time_horizon": [5, "months"],  // or null if no time horizon
        "preference_pair": { <PreferencePairOutput> },
        "choice": "short_term",
        "choice_probability": 0.85,
        "alternative_probability": 0.12,
        "internals": null,
        "debug": null
    }
    """

    sample_id: int
    time_horizon: list | None  # [value, unit] or None if no time horizon
    preference_pair: PreferencePairOutput
    choice: str  # "short_term", "long_term", or "unknown"
    choice_probability: Optional[float] = None  # Probability of chosen label token
    alternative_probability: Optional[float] = None  # Probability of alternative label token
    internals: Optional[InternalsReference] = None
    debug: Optional[PreferenceDebugInfo] = None


@dataclass
class PreferenceDataOutput(SchemaClass):
    """
    Full preference data output (output of query_llm.py).

    Sample JSON:
    {
        "metadata": { <PreferenceDataMetadata> },
        "preferences": [ <PreferenceItem>, ... ]
    }
    """

    metadata: PreferenceDataMetadata
    preferences: list[PreferenceItem]


# =============================================================================
# Training Output Schemas (from train_choice_model.py)
# =============================================================================


@dataclass
class ChoiceModelConfig(SchemaClass):
    """
    Configuration for training a choice model.

    Sample JSON:
    {
        "model_type": "value_function",
        "value_params": { <ValueFunctionParams> },
        "learning_rate": 0.01,
        "num_iterations": 100,
        "temperature": 0.0
    }
    """

    model_type: str = "value_function"  # "value_function", "logistic", etc.
    value_params: ValueFunctionParams = field(default_factory=ValueFunctionParams)
    learning_rate: float = 0.01
    num_iterations: int = 100
    temperature: float = 0.0  # Deterministic by default


# TrainingResult is imported from src.types


@dataclass
class TrainOutput(SchemaClass):
    """
    Output from training a choice model (output of train_choice_model.py).

    Sample JSON:
    {
        "config": { <ChoiceModelConfig> },
        "result": { <TrainingResult> },
        "dataset_name": "cityhousing",
        "timestamp": "20251229_195540"
    }
    """

    config: ChoiceModelConfig
    result: TrainingResult
    dataset_name: str
    timestamp: str = ""


# =============================================================================
# Test Output Schemas (from test_choice_model.py)
# =============================================================================


@dataclass
class EvaluationMetrics(SchemaClass):
    """
    Metrics from evaluating a choice model.

    Sample JSON:
    {
        "accuracy": 0.85,
        "num_samples": 100,
        "num_correct": 85,
        "short_term_accuracy": 0.9,
        "long_term_accuracy": 0.8,
        "n_short": 60,
        "n_long": 40,
        "per_horizon_accuracy": {"5 months": 0.88, "15 years": 0.82}
    }
    """

    accuracy: float
    num_samples: int
    num_correct: int
    short_term_accuracy: float = 0.0
    long_term_accuracy: float = 0.0
    n_short: int = 0
    n_long: int = 0
    per_horizon_accuracy: dict = field(default_factory=dict)


@dataclass
class TestOutput(SchemaClass):
    """
    Output from testing a choice model (output of test_choice_model.py).

    Sample JSON:
    {
        "model_path": "out/models/model_housing_20251229.json",
        "dataset_name": "cityhousing",
        "metrics": { <EvaluationMetrics> },
        "predictions": [[0, "a", "a"], [1, "b", "a"], ...],
        "timestamp": "20251229_195540"
    }
    """

    model_path: str
    dataset_name: str
    metrics: EvaluationMetrics
    predictions: list[tuple[int, str, str]]
    timestamp: str = ""


# =============================================================================
# Choice Analysis Output Schemas (from analyze_choices.py)
# =============================================================================


@dataclass
class DataReference(SchemaClass):
    """Reference to preference data file."""
    name: str
    model: str
    query_id: str


@dataclass
class AnalysisConfig(SchemaClass):
    """Configuration for choice analysis."""
    train_data: DataReference
    test_data: DataReference
    models: list  # List of model specs
    learning_rate: float = 0.01
    num_iterations: int = 100
    temperature: float = 0.0  # Deterministic by default


@dataclass
class ModelResultOutput(SchemaClass):
    """
    Result from training and testing a single model (output format).

    Note: This is the serialized form. The internal ModelResult class
    uses actual dataclass objects for spec, trained_params, etc.
    """
    model: dict  # {utility_type, discount_type}
    trained_params: dict  # Serialized ValueFunctionParams
    train_loss: float
    train_accuracy: float
    test_metrics: dict  # Serialized EvaluationMetrics


@dataclass
class AnalysisOutput(SchemaClass):
    """
    Full output from choice analysis (output of analyze_choices.py).

    Sample JSON:
    {
        "train_data": "out/preference_data/...",
        "test_data": "out/preference_data/...",
        "config": { learning_rate, num_iterations, temperature },
        "results": [ <ModelResultOutput>, ... ],
        "timestamp": "20251230_..."
    }
    """
    train_data_path: str
    test_data_path: str
    config: dict  # Serialized config subset
    results: list[ModelResultOutput]
    timestamp: str = ""


# =============================================================================
# Probe Training Output Schemas (from train_contrastive_probe.py)
# =============================================================================


@dataclass
class ProbeResultOutput(SchemaClass):
    """
    Result from training a single probe (output format).

    Sample JSON:
    {
        "layer": 8,
        "cv_accuracy_mean": 0.85,
        "cv_accuracy_std": 0.1,
        "test_accuracy": 0.9,
        "n_train": 100,
        "n_test": 20,
        "n_features": 768,
        "confusion_matrix": [[10, 2], [1, 7]]
    }
    """
    layer: int
    cv_accuracy_mean: float
    cv_accuracy_std: float
    test_accuracy: float
    n_train: int
    n_test: int
    n_features: int
    confusion_matrix: list  # 2D list


@dataclass
class ProbeTrainingOutput(SchemaClass):
    """
    Full output from probe training (output of train_contrastive_probe.py).

    Sample JSON:
    {
        "config_path": "scripts/configs/probes/default_probes.json",
        "data_path": "out/preference_data/...",
        "best_layer": 20,
        "best_accuracy": 0.95,
        "results": [ <ProbeResultOutput>, ... ],
        "timestamp": "20251230_..."
    }
    """
    config_path: str
    data_path: str
    best_layer: int
    best_accuracy: float
    results: list[ProbeResultOutput]
    timestamp: str = ""
