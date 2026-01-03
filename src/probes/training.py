"""
Probe training and evaluation pipeline.

Provides:
- Cross-validation for probe training
- Training probes across all (layer, token_position) combinations
- Evaluation with special handling for null time horizons
- Saving/loading trained probes for future use (e.g., steering experiments)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix

from .data import (
    CombinedPreferenceData,
    ChoiceLabels,
    TimeHorizonCategoryLabels,
    TimeHorizonValueLabels,
    ProbeDataset,
    build_probe_datasets,
    build_choice_labels,
    build_time_horizon_category_labels,
    build_time_horizon_value_labels,
)
from .models import (
    BaseProbe,
    ChoiceProbe,
    TimeHorizonCategoryProbe,
    TimeHorizonValueProbe,
    ProbeType,
    ProbeResult,
    ClassificationMetrics,
    RegressionMetrics,
    create_probe,
)


# =============================================================================
# Training Configuration
# =============================================================================


@dataclass
class ProbeTrainingConfig:
    """Configuration for probe training."""
    probe_types: list[ProbeType]
    n_cv_folds: int = 5
    random_state: int = 42
    regularization_C: float = 1.0  # For classification
    regularization_alpha: float = 1.0  # For regression


# =============================================================================
# Single Probe Training
# =============================================================================


def train_classification_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray],
    y_test: Optional[np.ndarray],
    probe_type: ProbeType,
    layer: int,
    token_position_idx: int,
    config: ProbeTrainingConfig,
) -> tuple[BaseProbe, ProbeResult]:
    """
    Train a classification probe with cross-validation.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features (optional)
        y_test: Test labels (optional)
        probe_type: Type of probe
        layer: Layer index
        token_position_idx: Token position index
        config: Training configuration

    Returns:
        Tuple of (trained probe, result)
    """
    # Create probe
    if probe_type == ProbeType.CHOICE:
        probe = ChoiceProbe(
            random_state=config.random_state,
            C=config.regularization_C,
        )
    else:  # TIME_HORIZON_CATEGORY
        probe = TimeHorizonCategoryProbe(
            random_state=config.random_state,
            C=config.regularization_C,
        )

    # Cross-validation
    cv_scores = []
    if len(np.unique(y_train)) > 1 and len(y_train) >= config.n_cv_folds:
        try:
            cv = StratifiedKFold(n_splits=config.n_cv_folds, shuffle=True, random_state=config.random_state)
            cv_scores = cross_val_score(probe.model, X_train, y_train, cv=cv, scoring="accuracy")
        except ValueError:
            # Not enough samples for stratified k-fold
            pass

    cv_mean = np.mean(cv_scores) if len(cv_scores) > 0 else 0.0
    cv_std = np.std(cv_scores) if len(cv_scores) > 0 else 0.0

    # Fit on full training data
    probe.fit(X_train, y_train)

    # Evaluate on training data
    train_metrics = probe.evaluate(X_train, y_train)

    # Evaluate on test data if provided
    test_metrics = None
    conf_matrix = None
    if X_test is not None and y_test is not None and len(y_test) > 0:
        test_metrics = probe.evaluate(X_test, y_test)
        y_pred = probe.predict(X_test)
        conf_matrix = confusion_matrix(y_test, y_pred).tolist()

    return probe, ProbeResult(
        layer=layer,
        token_position_idx=token_position_idx,
        probe_type=probe_type,
        cv_accuracy_mean=cv_mean,
        cv_accuracy_std=cv_std,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        confusion_matrix=conf_matrix,
        n_train=len(y_train),
        n_test=len(y_test) if y_test is not None else 0,
        n_features=X_train.shape[1],
    )


def train_regression_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray],
    y_test: Optional[np.ndarray],
    layer: int,
    token_position_idx: int,
    config: ProbeTrainingConfig,
) -> tuple[BaseProbe, ProbeResult]:
    """
    Train a regression probe.

    Args:
        X_train: Training features
        y_train: Training values
        X_test: Test features (optional)
        y_test: Test values (optional)
        layer: Layer index
        token_position_idx: Token position index
        config: Training configuration

    Returns:
        Tuple of (trained probe, result)
    """
    probe = TimeHorizonValueProbe(
        random_state=config.random_state,
        alpha=config.regularization_alpha,
    )

    # Fit on training data
    probe.fit(X_train, y_train)

    # Evaluate on training data
    train_metrics = probe.evaluate(X_train, y_train)

    # Evaluate on test data if provided
    test_metrics = None
    if X_test is not None and y_test is not None and len(y_test) > 0:
        test_metrics = probe.evaluate(X_test, y_test)

    return probe, ProbeResult(
        layer=layer,
        token_position_idx=token_position_idx,
        probe_type=ProbeType.TIME_HORIZON_VALUE,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        n_train=len(y_train),
        n_test=len(y_test) if y_test is not None else 0,
        n_features=X_train.shape[1],
    )


# =============================================================================
# Full Training Pipeline
# =============================================================================


@dataclass
class TrainedProbe:
    """A trained probe with metadata for saving/loading."""
    probe: BaseProbe
    probe_type: ProbeType
    layer: int
    token_position_idx: int
    result: ProbeResult


@dataclass
class ProbeTrainingOutput:
    """Output from training all probes."""
    results: list[ProbeResult]
    trained_probes: list[TrainedProbe]  # All trained probe models
    best_by_type: dict[ProbeType, ProbeResult]
    train_data_query_ids: list[str]
    test_data_query_ids: list[str]
    model_name: str
    config: ProbeTrainingConfig


def train_all_probes(
    train_data: CombinedPreferenceData,
    test_data: Optional[CombinedPreferenceData],
    config: ProbeTrainingConfig,
) -> ProbeTrainingOutput:
    """
    Train probes for all (layer, token_position, probe_type) combinations.

    Args:
        train_data: Combined training data
        test_data: Combined test data (optional)
        config: Training configuration

    Returns:
        ProbeTrainingOutput with all results and trained probes
    """
    results: list[ProbeResult] = []
    trained_probes: list[TrainedProbe] = []

    # Build datasets
    train_datasets = build_probe_datasets(train_data)
    test_datasets = build_probe_datasets(test_data) if test_data else {}

    # Build labels
    train_choice_labels = build_choice_labels(train_data)
    train_th_cat_labels = build_time_horizon_category_labels(train_data)
    train_th_val_labels = build_time_horizon_value_labels(train_data)

    test_choice_labels = build_choice_labels(test_data) if test_data else None
    test_th_cat_labels = build_time_horizon_category_labels(test_data) if test_data else None
    test_th_val_labels = build_time_horizon_value_labels(test_data) if test_data else None

    # Train for each (layer, token_pos_idx) combination
    for key, train_dataset in train_datasets.items():
        layer, tp_idx = key

        # Get test dataset if available
        test_dataset = test_datasets.get(key)

        for probe_type in config.probe_types:
            if probe_type == ProbeType.CHOICE:
                # Filter to valid samples
                train_mask = train_choice_labels.valid_mask
                X_train = train_dataset.X[train_mask]
                y_train = train_choice_labels.y[train_mask]

                X_test = None
                y_test = None
                if test_dataset and test_choice_labels:
                    test_mask = test_choice_labels.valid_mask
                    X_test = test_dataset.X[test_mask]
                    y_test = test_choice_labels.y[test_mask]

                if len(y_train) >= 2 and len(np.unique(y_train)) > 1:
                    probe, result = train_classification_probe(
                        X_train, y_train, X_test, y_test,
                        probe_type, layer, tp_idx, config,
                    )
                    results.append(result)
                    trained_probes.append(TrainedProbe(
                        probe=probe,
                        probe_type=probe_type,
                        layer=layer,
                        token_position_idx=tp_idx,
                        result=result,
                    ))

            elif probe_type == ProbeType.TIME_HORIZON_CATEGORY:
                # Filter to valid samples (non-null time horizon)
                train_mask = train_th_cat_labels.valid_mask
                X_train = train_dataset.X[train_mask]
                y_train = train_th_cat_labels.y[train_mask]

                X_test = None
                y_test = None
                if test_dataset and test_th_cat_labels:
                    test_mask = test_th_cat_labels.valid_mask
                    X_test = test_dataset.X[test_mask]
                    y_test = test_th_cat_labels.y[test_mask]

                if len(y_train) >= 2 and len(np.unique(y_train)) > 1:
                    probe, result = train_classification_probe(
                        X_train, y_train, X_test, y_test,
                        probe_type, layer, tp_idx, config,
                    )
                    results.append(result)
                    trained_probes.append(TrainedProbe(
                        probe=probe,
                        probe_type=probe_type,
                        layer=layer,
                        token_position_idx=tp_idx,
                        result=result,
                    ))

            elif probe_type == ProbeType.TIME_HORIZON_VALUE:
                # Filter to valid samples (non-null time horizon)
                train_mask = train_th_val_labels.valid_mask
                X_train = train_dataset.X[train_mask]
                y_train = train_th_val_labels.y[train_mask]

                X_test = None
                y_test = None
                if test_dataset and test_th_val_labels:
                    test_mask = test_th_val_labels.valid_mask
                    X_test = test_dataset.X[test_mask]
                    y_test = test_th_val_labels.y[test_mask]

                if len(y_train) >= 2:
                    probe, result = train_regression_probe(
                        X_train, y_train, X_test, y_test,
                        layer, tp_idx, config,
                    )
                    results.append(result)
                    trained_probes.append(TrainedProbe(
                        probe=probe,
                        probe_type=ProbeType.TIME_HORIZON_VALUE,
                        layer=layer,
                        token_position_idx=tp_idx,
                        result=result,
                    ))

    # Find best probe for each type
    best_by_type: dict[ProbeType, ProbeResult] = {}
    for probe_type in config.probe_types:
        type_results = [r for r in results if r.probe_type == probe_type]
        if type_results:
            if probe_type in (ProbeType.CHOICE, ProbeType.TIME_HORIZON_CATEGORY):
                # Best by test accuracy (or CV if no test)
                best = max(type_results, key=lambda r: (
                    r.test_metrics.accuracy if r.test_metrics else r.cv_accuracy_mean
                ))
            else:  # Regression
                # Best by lowest test MAE (or train if no test)
                best = min(type_results, key=lambda r: (
                    r.test_metrics.mae if r.test_metrics else r.train_metrics.mae
                ))
            best_by_type[probe_type] = best

    return ProbeTrainingOutput(
        results=results,
        trained_probes=trained_probes,
        best_by_type=best_by_type,
        train_data_query_ids=[],  # Filled by caller
        test_data_query_ids=[],
        model_name=train_data.model,
        config=config,
    )


# =============================================================================
# Special Evaluation for Null Time Horizons
# =============================================================================


def evaluate_category_probe_on_null_horizons(
    probe: TimeHorizonCategoryProbe,
    data: CombinedPreferenceData,
    datasets: dict[tuple[int, int], ProbeDataset],
    layer: int,
    token_position_idx: int,
) -> dict:
    """
    Evaluate time horizon category probe on samples with null time_horizon.

    For null time_horizon samples, check if predicted category matches the choice term.
    - If choice is "short_term" and pred is 0 (short), that's a match
    - If choice is "long_term" and pred is 1 (long), that's a match

    Args:
        probe: Trained time horizon category probe
        data: Combined preference data
        datasets: Probe datasets
        layer: Layer index
        token_position_idx: Token position index

    Returns:
        Dict with evaluation results
    """
    dataset = datasets.get((layer, token_position_idx))
    if dataset is None:
        return {"n_samples": 0}

    # Find samples with null time horizon
    null_indices = []
    null_choices = []
    for i, sample in enumerate(data.samples):
        if sample.time_horizon is None:
            null_indices.append(i)
            null_choices.append(sample.choice)

    if not null_indices:
        return {"n_samples": 0}

    # Get activations for null samples
    X_null = dataset.X[null_indices]
    predictions = probe.predict(X_null)

    # Compare predictions to choices
    matches = 0
    for pred, choice in zip(predictions, null_choices):
        if choice == "short_term" and pred == 0:
            matches += 1
        elif choice == "long_term" and pred == 1:
            matches += 1

    return {
        "n_samples": len(null_indices),
        "n_matches": matches,
        "accuracy": matches / len(null_indices) if null_indices else 0.0,
    }


def evaluate_probes_on_no_horizon(
    output: ProbeTrainingOutput,
    train_data: CombinedPreferenceData,
    test_data: Optional[CombinedPreferenceData] = None,
) -> "NoHorizonResults":
    """
    Evaluate choice probes specifically on no-horizon samples.

    Args:
        output: ProbeTrainingOutput containing trained probes
        train_data: Training data with samples
        test_data: Test data with samples (optional)

    Returns:
        NoHorizonResults with accuracy for each (layer, token_position) combination
    """
    # Filter to choice probes only
    choice_probes = [
        tp for tp in output.trained_probes
        if tp.probe_type == ProbeType.CHOICE
    ]

    if not choice_probes:
        return NoHorizonResults(results={}, n_train=0, n_test=0)

    # Build datasets for evaluation
    train_datasets = build_probe_datasets(train_data)
    test_datasets = build_probe_datasets(test_data) if test_data else {}

    # Find no-horizon sample indices
    train_no_horizon_indices = [
        i for i, s in enumerate(train_data.samples)
        if s.time_horizon is None and s.choice != "unknown"
    ]
    test_no_horizon_indices = []
    if test_data:
        test_no_horizon_indices = [
            i for i, s in enumerate(test_data.samples)
            if s.time_horizon is None and s.choice != "unknown"
        ]

    # Build choice labels for no-horizon samples
    train_y = np.array([
        0 if train_data.samples[i].choice == "short_term" else 1
        for i in train_no_horizon_indices
    ])
    test_y = np.array([
        0 if test_data.samples[i].choice == "short_term" else 1
        for i in test_no_horizon_indices
    ]) if test_no_horizon_indices else None

    results = {}
    for tp in choice_probes:
        key = (tp.layer, tp.token_position_idx)
        train_dataset = train_datasets.get(key)
        test_dataset = test_datasets.get(key)

        if train_dataset is None:
            continue

        # Evaluate on train no-horizon samples
        X_train = train_dataset.X[train_no_horizon_indices]
        train_pred = tp.probe.predict(X_train)
        train_acc = np.mean(train_pred == train_y) if len(train_y) > 0 else 0.0

        # Evaluate on test no-horizon samples
        test_acc = None
        if test_dataset is not None and len(test_no_horizon_indices) > 0:
            X_test = test_dataset.X[test_no_horizon_indices]
            test_pred = tp.probe.predict(X_test)
            test_acc = np.mean(test_pred == test_y) if test_y is not None and len(test_y) > 0 else None

        results[key] = NoHorizonProbeResult(
            layer=tp.layer,
            token_position_idx=tp.token_position_idx,
            train_accuracy=train_acc,
            test_accuracy=test_acc,
            n_train=len(train_no_horizon_indices),
            n_test=len(test_no_horizon_indices),
        )

    return NoHorizonResults(
        results=results,
        n_train=len(train_no_horizon_indices),
        n_test=len(test_no_horizon_indices),
    )


@dataclass
class NoHorizonProbeResult:
    """Result from evaluating a probe on no-horizon samples."""
    layer: int
    token_position_idx: int
    train_accuracy: float
    test_accuracy: Optional[float]
    n_train: int
    n_test: int


@dataclass
class NoHorizonResults:
    """Results from evaluating all probes on no-horizon samples."""
    results: dict[tuple[int, int], NoHorizonProbeResult]
    n_train: int
    n_test: int


# =============================================================================
# Probe Saving/Loading for Future Experiments
# =============================================================================


def save_probes(
    output: ProbeTrainingOutput,
    probes_dir: Path,
    save_all: bool = False,
    tp_info: Optional[object] = None,  # TokenPositionInfo, optional import to avoid circular
) -> dict[str, Path]:
    """
    Save trained probes to disk for future experiments (e.g., steering).

    Saves probes as joblib files with metadata JSON alongside.

    Args:
        output: ProbeTrainingOutput from training
        probes_dir: Directory to save probes
        save_all: If True, save all probes. If False, only save best per type.
        tp_info: Optional TokenPositionInfo for computing relative positions

    Returns:
        Dict mapping probe identifier to saved path
    """
    from datetime import datetime

    probes_dir = Path(probes_dir)
    probes_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = {}
    probe_summaries = []

    # Build position lookup if tp_info provided
    # Maps token_position_idx to its sequence position in the order
    seq_positions = {}
    boundary_positions = {}
    if tp_info is not None:
        for seq_pos, tp_idx in enumerate(tp_info.order):
            seq_positions[tp_idx] = seq_pos
        # Get boundary sequence positions
        if tp_info.choices_presented_idx >= 0 and tp_info.choices_presented_idx in seq_positions:
            boundary_positions["choices_presented"] = seq_positions[tp_info.choices_presented_idx]
        if tp_info.time_horizon_spec_idx >= 0 and tp_info.time_horizon_spec_idx in seq_positions:
            boundary_positions["time_horizon"] = seq_positions[tp_info.time_horizon_spec_idx]
        if tp_info.choice_made_idx >= 0 and tp_info.choice_made_idx in seq_positions:
            boundary_positions["choice_made"] = seq_positions[tp_info.choice_made_idx]
        if tp_info.prompt_end_idx >= 0:
            boundary_positions["prompt_end"] = tp_info.prompt_end_idx

    # Determine which probes to save
    if save_all:
        probes_to_save = output.trained_probes
    else:
        # Only save best probe for each type
        probes_to_save = []
        for probe_type, best_result in output.best_by_type.items():
            for tp in output.trained_probes:
                if (tp.probe_type == probe_type and
                    tp.layer == best_result.layer and
                    tp.token_position_idx == best_result.token_position_idx):
                    probes_to_save.append(tp)
                    break

    for tp in probes_to_save:
        # Create identifier for this probe
        probe_id = f"{tp.probe_type.value}_layer{tp.layer}_pos{tp.token_position_idx}"

        # Save sklearn model
        model_path = probes_dir / f"{probe_id}.joblib"
        joblib.dump(tp.probe.model, model_path)

        # Save probe direction (coefficients) as numpy for easy steering
        coef_path = probes_dir / f"{probe_id}_direction.npy"
        np.save(coef_path, tp.probe.coef_)

        # Save intercept if classification
        intercept_file = None
        if hasattr(tp.probe, 'intercept_'):
            intercept_path = probes_dir / f"{probe_id}_intercept.npy"
            np.save(intercept_path, tp.probe.intercept_)
            intercept_file = intercept_path.name

        # Build structured metadata
        is_classification = tp.probe_type in (ProbeType.CHOICE, ProbeType.TIME_HORIZON_CATEGORY)

        # Get actual token position and word if available
        token_position = None
        token_position_word = None
        if tp_info is not None:
            if hasattr(tp_info, 'resolved_positions') and tp.token_position_idx in tp_info.resolved_positions:
                token_position = tp_info.resolved_positions[tp.token_position_idx]
            if hasattr(tp_info, 'tokens') and tp.token_position_idx in tp_info.tokens:
                token_position_word = tp_info.tokens[tp.token_position_idx]

        metadata = {
            # Model info
            "model": output.model_name,

            # Probe location in model
            "probe_type": tp.probe_type.value,
            "layer": tp.layer,
            "token_position_idx": tp.token_position_idx,
            "token_position": token_position,  # Actual position in token sequence
            "token_position_word": token_position_word,  # Token/word at that position

            # Training info
            "training": {
                "n_samples": tp.result.n_train,
                "n_features": tp.result.n_features,
                "train_query_ids": output.train_data_query_ids,
                "test_query_ids": output.test_data_query_ids,
            },

            # Performance metrics
            "metrics": {},

            # Files
            "files": {
                "model": f"{probe_id}.joblib",
                "direction": coef_path.name,
            },
        }

        if intercept_file:
            metadata["files"]["intercept"] = intercept_file

        # Add relative position info if available
        if seq_positions and tp.token_position_idx in seq_positions:
            seq_pos = seq_positions[tp.token_position_idx]
            position_info = {
                "sequence_position": seq_pos,
            }
            # Add before/after flags for each boundary
            if "choices_presented" in boundary_positions:
                position_info["after_choices_presented"] = seq_pos >= boundary_positions["choices_presented"]
            if "time_horizon" in boundary_positions:
                position_info["after_time_horizon"] = seq_pos >= boundary_positions["time_horizon"]
            if "choice_made" in boundary_positions:
                position_info["after_choice_made"] = seq_pos >= boundary_positions["choice_made"]
            if "prompt_end" in boundary_positions:
                position_info["in_response"] = seq_pos > boundary_positions["prompt_end"]
            metadata["position"] = position_info

        # Add metrics based on probe type
        if is_classification:
            metadata["metrics"]["cv_accuracy"] = {
                "mean": tp.result.cv_accuracy_mean,
                "std": tp.result.cv_accuracy_std,
            }
            if tp.result.test_metrics:
                metadata["metrics"]["test"] = {
                    "accuracy": tp.result.test_metrics.accuracy,
                    "f1": tp.result.test_metrics.f1,
                }
        else:
            if tp.result.train_metrics:
                metadata["metrics"]["train"] = {
                    "mae_months": tp.result.train_metrics.mae,
                }
            if tp.result.test_metrics:
                metadata["metrics"]["test"] = {
                    "mae_months": tp.result.test_metrics.mae,
                    "rmse_months": tp.result.test_metrics.rmse,
                }

        # Build summary for index
        if is_classification:
            accuracy = tp.result.test_metrics.accuracy if tp.result.test_metrics else tp.result.cv_accuracy_mean
            summary = {
                "id": probe_id,
                "type": tp.probe_type.value,
                "layer": tp.layer,
                "position": tp.token_position_idx,
                "accuracy": round(accuracy, 4),
            }
        else:
            mae = tp.result.test_metrics.mae if tp.result.test_metrics else (
                tp.result.train_metrics.mae if tp.result.train_metrics else None
            )
            summary = {
                "id": probe_id,
                "type": tp.probe_type.value,
                "layer": tp.layer,
                "position": tp.token_position_idx,
                "mae_months": round(mae, 2) if mae else None,
            }
        # Add position flags to summary if available
        if "position" in metadata:
            pos = metadata["position"]
            summary["after_choices"] = pos.get("after_choices_presented", None)
            summary["after_horizon"] = pos.get("after_time_horizon", None)
            summary["after_choice_made"] = pos.get("after_choice_made", None)
            summary["in_response"] = pos.get("in_response", None)
        probe_summaries.append(summary)

        metadata_path = probes_dir / f"{probe_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        saved_paths[probe_id] = model_path
        print(f"  Saved: {probe_id}")

    # Save index file with better structure
    index = {
        "model": output.model_name,
        "created": datetime.now().isoformat(),

        "data": {
            "train_query_ids": output.train_data_query_ids,
        },

        "probes": probe_summaries,

        "best_by_type": {
            pt.value: f"{pt.value}_layer{r.layer}_pos{r.token_position_idx}"
            for pt, r in output.best_by_type.items()
        },
    }
    with open(probes_dir / "index.json", 'w') as f:
        json.dump(index, f, indent=2)

    return saved_paths


@dataclass
class LoadedProbe:
    """A probe loaded from disk with its metadata."""
    model: object  # sklearn model
    direction: np.ndarray  # Coefficient vector for steering
    intercept: Optional[np.ndarray]  # Intercept if available
    probe_type: ProbeType
    layer: int
    token_position_idx: int
    n_features: int
    model_name: str
    metadata: dict


def load_probe(probe_path: Path) -> LoadedProbe:
    """
    Load a single probe from disk.

    Args:
        probe_path: Path to the .joblib probe file

    Returns:
        LoadedProbe with model and metadata
    """
    probe_path = Path(probe_path)

    # Load sklearn model
    model = joblib.load(probe_path)

    # Load metadata
    metadata_path = probe_path.with_suffix('.json')
    with open(metadata_path) as f:
        metadata = json.load(f)

    # Handle both old and new format
    if "files" in metadata:
        # New format
        direction_file = metadata["files"]["direction"]
        intercept_file = metadata["files"].get("intercept")
        n_features = metadata["training"]["n_features"]
        model_name = metadata["model"]
    else:
        # Old format (backwards compatibility)
        direction_file = metadata["direction_file"]
        intercept_file = metadata.get("intercept_file")
        n_features = metadata["n_features"]
        model_name = metadata.get("model_name", metadata.get("model", "unknown"))

    # Load direction (coefficients)
    direction_path = probe_path.parent / direction_file
    direction = np.load(direction_path)

    # Load intercept if available
    intercept = None
    if intercept_file:
        intercept_path = probe_path.parent / intercept_file
        intercept = np.load(intercept_path)

    return LoadedProbe(
        model=model,
        direction=direction,
        intercept=intercept,
        probe_type=ProbeType(metadata["probe_type"]),
        layer=metadata["layer"],
        token_position_idx=metadata["token_position_idx"],
        n_features=n_features,
        model_name=model_name,
        metadata=metadata,
    )


def load_probes_from_dir(probes_dir: Path) -> dict[str, LoadedProbe]:
    """
    Load all probes from a directory.

    Args:
        probes_dir: Directory containing saved probes

    Returns:
        Dict mapping probe_id to LoadedProbe
    """
    probes_dir = Path(probes_dir)

    # Load index if available
    index_path = probes_dir / "index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        probes_list = index.get("probes", [])
        # Handle both old format (list of strings) and new format (list of dicts)
        if probes_list and isinstance(probes_list[0], dict):
            probe_ids = [p["id"] for p in probes_list]
        else:
            probe_ids = probes_list
    else:
        # Find all .joblib files
        probe_ids = [p.stem for p in probes_dir.glob("*.joblib")]

    loaded = {}
    for probe_id in probe_ids:
        probe_path = probes_dir / f"{probe_id}.joblib"
        if probe_path.exists():
            loaded[probe_id] = load_probe(probe_path)

    return loaded


def get_steering_direction(
    probes_dir: Path,
    probe_type: str = "choice",
    normalize: bool = True,
) -> tuple[np.ndarray, int, int]:
    """
    Get the steering direction vector for a probe.

    This is the main interface for steering experiments.
    Returns the probe coefficients which can be added/subtracted
    from activations to steer model behavior.

    Args:
        probes_dir: Directory containing saved probes
        probe_type: "choice", "time_horizon_category", or "time_horizon_value"
        normalize: Whether to normalize the direction vector

    Returns:
        Tuple of (direction_vector, layer, token_position_idx)
    """
    probes_dir = Path(probes_dir)

    # Load index to find best probe
    index_path = probes_dir / "index.json"
    with open(index_path) as f:
        index = json.load(f)

    probe_id = index["best_by_type"].get(probe_type)
    if not probe_id:
        raise ValueError(f"No best probe found for type: {probe_type}")

    # Load the probe
    probe = load_probe(probes_dir / f"{probe_id}.joblib")

    direction = probe.direction.flatten()
    if normalize:
        direction = direction / np.linalg.norm(direction)

    return direction, probe.layer, probe.token_position_idx
