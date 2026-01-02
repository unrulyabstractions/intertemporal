"""
Output-specific I/O functions for scripts.

Provides save/load functions for script outputs. Uses schemas from scripts/schemas.py.
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.io import ensure_dir, get_timestamp, load_json, save_json
from src.common.profiling import get_profiler
from src.types import (
    DatasetMetadata,
    DatasetSample,
    DiscountFunctionParams,
    DiscountType,
    IntertemporalOption,
    PreferencePair,
    PreferenceQuestion,
    Prompt,
    Response,
    RewardValue,
    TimeValue,
    UtilityType,
    ValueFunctionParams,
)
from .schemas import (
    SCHEMA_VERSION,
    ChoiceModelConfig,
    DatasetOutput,
    DatasetOutputMetadata,
    EvaluationMetrics,
    GenerateDatasetOutput,
    InternalsReference,
    OptionOutput,
    PreferenceDataMetadata,
    PreferenceDataOutput,
    PreferenceDebugInfo,
    PreferenceItem,
    PreferencePairOutput,
    QuestionOutput,
    TestOutput,
    TrainingResult,
    TrainOutput,
)


# =============================================================================
# Dataset Output I/O
# =============================================================================


def save_dataset(
    samples: list[DatasetSample],
    dataset_id: str,
    dataset_run_id: str,
    config_dict: dict,
    output_path: Path,
) -> GenerateDatasetOutput:
    """
    Save generated dataset as single JSON file.

    Args:
        samples: List of dataset samples
        dataset_id: ID from DatasetConfig.get_id()
        dataset_run_id: Shared ID for all datasets from same --config invocation
        config_dict: Full DatasetConfig as dict
        output_path: Output file path

    Returns:
        GenerateDatasetOutput with save info
    """
    profiler = get_profiler()

    # Ensure parent directory exists
    ensure_dir(output_path.parent)

    # Build questions list using schema classes
    questions = []
    with profiler.measure("build_questions", {"num_samples": len(samples)}):
        for sample in samples:
            pair = sample.prompt.question.pair

            th = sample.prompt.question.time_horizon
            question = QuestionOutput(
                sample_id=sample.id,
                question_text=sample.prompt.text,
                time_horizon=th.to_list() if th is not None else None,
                preference_pair=PreferencePairOutput(
                    short_term=OptionOutput(
                        label=pair.short_term.label,
                        time=pair.short_term.time.to_list(),
                        reward=pair.short_term.reward.value,
                    ),
                    long_term=OptionOutput(
                        label=pair.long_term.label,
                        time=pair.long_term.time.to_list(),
                        reward=pair.long_term.reward.value,
                    ),
                ),
            )
            questions.append(question)

    # Build output using schema classes
    timestamp = get_timestamp()
    dataset_output = DatasetOutput(
        metadata=DatasetOutputMetadata(
            version=SCHEMA_VERSION,
            dataset_id=dataset_id,
            config=config_dict,
            num_questions=len(questions),
            timestamp=timestamp,
            dataset_run_id=dataset_run_id,
        ),
        questions=questions,
    )

    with profiler.measure("save_json", {"path": str(output_path)}):
        save_json(asdict(dataset_output), output_path)

    return GenerateDatasetOutput(
        dataset_id=dataset_id,
        num_samples=len(samples),
        num_horizons=len(set(str(s.prompt.question.time_horizon) for s in samples)),
        output_path=str(output_path),
        timestamp=timestamp,
    )


def load_dataset_output(path: Path) -> DatasetOutput:
    """
    Load generated dataset from JSON file.

    Sample JSON (see schemas: DatasetOutput, DatasetOutputMetadata, QuestionOutput):
    {
        "metadata": {
            "version": "1.0",
            "config_id": "79642578326872f28277b0f349873061",
            "config": { ... },
            "num_questions": 8,
            "timestamp": "20251229_195540"
        },
        "questions": [
            {
                "question_text": "Situation: Plan for housing...",
                "time_horizon": [5, "months"],
                "preference_pair": {
                    "short_term": { "label": "a", "time": [3, "months"], "reward": 2000 },
                    "long_term": { "label": "b", "time": [10, "years"], "reward": 20000 }
                }
            },
            ...
        ]
    }
    """
    data = load_json(path)

    # Validate version
    file_version = data["metadata"].get("version", "unknown")
    if file_version != SCHEMA_VERSION:
        raise ValueError(
            f"Version mismatch: file has version '{file_version}', "
            f"expected '{SCHEMA_VERSION}'"
        )

    # Support both old config_id and new dataset_id
    dataset_id = data["metadata"].get("dataset_id") or data["metadata"].get("config_id")
    metadata = DatasetOutputMetadata(
        version=file_version,
        dataset_id=dataset_id,
        config=data["metadata"]["config"],
        num_questions=data["metadata"]["num_questions"],
        timestamp=data["metadata"].get("timestamp", ""),
    )

    questions = []
    for i, q in enumerate(data["questions"]):
        pair = q["preference_pair"]
        question = QuestionOutput(
            sample_id=q.get("sample_id", i),  # Fallback to index for old data
            question_text=q["question_text"],
            time_horizon=q["time_horizon"],
            preference_pair=PreferencePairOutput(
                short_term=OptionOutput(
                    label=pair["short_term"]["label"],
                    time=pair["short_term"]["time"],
                    reward=pair["short_term"]["reward"],
                ),
                long_term=OptionOutput(
                    label=pair["long_term"]["label"],
                    time=pair["long_term"]["time"],
                    reward=pair["long_term"]["reward"],
                ),
            ),
        )
        questions.append(question)

    return DatasetOutput(metadata=metadata, questions=questions)


# =============================================================================
# Preference Data I/O (from query_llm.py)
# =============================================================================


def save_preference_data(
    output: PreferenceDataOutput,
    output_dir: Path,
    dataset_name: str,
    model_name: str,
    query_id: str,
) -> Path:
    """
    Save preference data output.

    Args:
        output: PreferenceDataOutput to save
        output_dir: Output directory
        dataset_name: Name of the dataset
        model_name: Model name (will be sanitized for filename)
        query_id: Query config ID

    Returns:
        Path to saved file

    Filename format: {dataset_name}_{model}_{query_id}.json
    """
    ensure_dir(output_dir)

    # Use short model name (after /) for filename
    short_model_name = model_name.split("/")[-1]
    filename = f"{dataset_name}_{short_model_name}_{query_id}.json"
    output_path = output_dir / filename

    save_json(asdict(output), output_path)
    return output_path


def load_preference_data(path: Path) -> PreferenceDataOutput:
    """
    Load preference data from file.

    Sample JSON:
    {
        "metadata": {
            "version": "1.0",
            "dataset_id": "5584270069cdc0e03965fdb0e4f5e10b",
            "formatting_id": "a1b2c3d4...",
            "query_id": "f7e8d9c0...",
            "model": "gpt2",
            "query_config": { <QueryConfig as dict> }
        },
        "preferences": [
            {
                "sample_id": 0,
                "time_horizon": [5, "months"],
                "preference_pair": { ... },
                "choice": "short_term",
                "internals": null
            },
            ...
        ]
    }
    """
    data = load_json(path)

    # Support both old config_id and new dataset_id
    meta = data["metadata"]
    dataset_id = meta.get("dataset_id") or meta.get("config_id", "")
    formatting_id = meta.get("formatting_id", "")
    query_id = meta.get("query_id", "")
    query_config = meta.get("query_config", {})

    metadata = PreferenceDataMetadata(
        version=meta["version"],
        dataset_id=dataset_id,
        formatting_id=formatting_id,
        query_id=query_id,
        model=meta["model"],
        query_config=query_config,
    )

    preferences = []
    for p in data["preferences"]:
        pair = p["preference_pair"]

        # Parse debug info if present
        debug_info = None
        if p.get("debug"):
            debug_info = PreferenceDebugInfo(**p["debug"])

        preference = PreferenceItem(
            sample_id=p["sample_id"],
            time_horizon=p["time_horizon"],
            preference_pair=PreferencePairOutput(
                short_term=OptionOutput(
                    label=pair["short_term"]["label"],
                    time=pair["short_term"]["time"],
                    reward=pair["short_term"]["reward"],
                ),
                long_term=OptionOutput(
                    label=pair["long_term"]["label"],
                    time=pair["long_term"]["time"],
                    reward=pair["long_term"]["reward"],
                ),
            ),
            choice=p["choice"],
            internals=InternalsReference(**p["internals"]) if p.get("internals") else None,
            debug=debug_info,
        )
        preferences.append(preference)

    return PreferenceDataOutput(metadata=metadata, preferences=preferences)


# =============================================================================
# Training Output I/O
# =============================================================================


def save_train_output(output: TrainOutput, output_dir: Path) -> Path:
    """
    Save training output.

    Args:
        output: TrainOutput to save
        output_dir: Output directory

    Returns:
        Path to saved file
    """
    ensure_dir(output_dir)

    timestamp = output.timestamp or get_timestamp()
    filename = f"model_{output.dataset_name}_{timestamp}.json"
    output_path = output_dir / filename

    save_json(asdict(output), output_path)
    return output_path


def load_train_output(path: Path) -> TrainOutput:
    """
    Load training output from file.

    Sample JSON (see schemas: TrainOutput, ChoiceModelConfig, TrainingResult):
    {
        "config": {
            "model_type": "value_function",
            "value_params": {
                "utility_type": "linear",
                "alpha": 1.0,
                "discount": {
                    "discount_type": "exponential",
                    "theta": 0.1,
                    "beta": 1.0,
                    "delta": 0.99
                }
            },
            "learning_rate": 0.01,
            "num_iterations": 100,
            "temperature": 1.0
        },
        "result": {
            "params": { ... },
            "loss": 0.05,
            "num_samples": 100,
            "accuracy": 0.95
        },
        "dataset_name": "cityhousing",
        "timestamp": "20251229_195540"
    }
    """
    data = load_json(path)

    # Reconstruct config
    cfg = data["config"]
    vp = cfg["value_params"]
    dp = vp["discount"]

    discount_params = DiscountFunctionParams(
        discount_type=DiscountType(dp["discount_type"]),
        theta=dp["theta"],
        beta=dp["beta"],
        delta=dp["delta"],
    )

    value_params = ValueFunctionParams(
        utility_type=UtilityType(vp["utility_type"]),
        alpha=vp["alpha"],
        discount=discount_params,
    )

    config = ChoiceModelConfig(
        model_type=cfg["model_type"],
        value_params=value_params,
        learning_rate=cfg["learning_rate"],
        num_iterations=cfg["num_iterations"],
        temperature=cfg["temperature"],
    )

    # Reconstruct result
    res = data["result"]
    rp = res["params"]
    rdp = rp["discount"]

    result_discount = DiscountFunctionParams(
        discount_type=DiscountType(rdp["discount_type"]),
        theta=rdp["theta"],
        beta=rdp["beta"],
        delta=rdp["delta"],
    )

    result_params = ValueFunctionParams(
        utility_type=UtilityType(rp["utility_type"]),
        alpha=rp["alpha"],
        discount=result_discount,
    )

    result = TrainingResult(
        params=result_params,
        loss=res["loss"],
        num_samples=res["num_samples"],
        accuracy=res["accuracy"],
    )

    return TrainOutput(
        config=config,
        result=result,
        dataset_name=data["dataset_name"],
        timestamp=data.get("timestamp", ""),
    )


# =============================================================================
# Test Output I/O
# =============================================================================


def save_test_output(output: TestOutput, output_dir: Path) -> Path:
    """
    Save test output.

    Args:
        output: TestOutput to save
        output_dir: Output directory

    Returns:
        Path to saved file
    """
    ensure_dir(output_dir)

    timestamp = output.timestamp or get_timestamp()
    filename = f"test_{output.dataset_name}_{timestamp}.json"
    output_path = output_dir / filename

    save_json(asdict(output), output_path)
    return output_path


def load_test_output(path: Path) -> TestOutput:
    """
    Load test output from file.

    Sample JSON (see schemas: TestOutput, EvaluationMetrics):
    {
        "model_path": "out/models/model_housing_20251229.json",
        "dataset_name": "cityhousing",
        "metrics": {
            "accuracy": 0.85,
            "num_samples": 100,
            "num_correct": 85,
            "short_term_accuracy": 0.9,
            "long_term_accuracy": 0.8,
            "per_horizon_accuracy": {"5 months": 0.88, "15 years": 0.82}
        },
        "predictions": [[0, "a", "a"], [1, "b", "a"], ...],
        "timestamp": "20251229_195540"
    }
    """
    data = load_json(path)

    metrics = EvaluationMetrics(**data["metrics"])

    return TestOutput(
        model_path=data["model_path"],
        dataset_name=data["dataset_name"],
        metrics=metrics,
        predictions=data["predictions"],
        timestamp=data.get("timestamp", ""),
    )


# =============================================================================
# Choice Analysis Output I/O
# =============================================================================


def save_analysis_output(
    output_dict: dict,
    output_dir: Path,
    dataset_name: str,
    timestamp: str,
) -> Path:
    """
    Save choice analysis output.

    Args:
        output_dict: Serialized analysis output dict
        output_dir: Output directory
        dataset_name: Dataset name for filename
        timestamp: Timestamp for filename

    Returns:
        Path to saved file
    """
    ensure_dir(output_dir)

    # Use simple filename without timestamp
    filename = f"analysis_{dataset_name}.json"
    output_path = output_dir / filename

    save_json(output_dict, output_path)
    return output_path


# =============================================================================
# Probe Training Output I/O
# =============================================================================


def save_probe_output(
    output_dict: dict,
    output_dir: Path,
    timestamp: str,
) -> Path:
    """
    Save probe training output.

    Args:
        output_dict: Serialized probe output dict
        output_dir: Output directory
        timestamp: Timestamp for filename

    Returns:
        Path to saved file
    """
    ensure_dir(output_dir)

    filename = f"probe_results_{timestamp}.json"
    output_path = output_dir / filename

    save_json(output_dict, output_path)
    return output_path
