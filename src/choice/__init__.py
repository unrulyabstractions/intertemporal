"""Choice analysis module.

This module provides tools for analyzing intertemporal choice behavior:
- Data loading and preprocessing
- Consistency analysis
- Model training and evaluation
- Results handling and visualization
"""

from .config import AnalysisConfig, load_analysis_config
from .data import (
    bucket_samples_by_horizon,
    class_balanced_train_test_split,
    extract_samples_with_horizons,
    filter_problematic_samples,
    find_preference_data_by_query_id,
    format_horizon,
    is_ambiguous_sample,
    load_samples_from_query_ids,
    serialize_problematic_samples,
)
from .consistency import (
    analyze_consistency,
    extract_constraints,
    filter_consistent_samples,
)
from .training import (
    test_model,
    train_and_test_by_horizon,
    train_model,
)
from .results import (
    AnalysisResult,
    print_summary,
    serialize_analysis_result,
)
from .types import (
    ALL_MODEL_SPECS,
    AmbiguousSample,
    ChoiceModelSpec,
    EvaluationMetrics,
    ExtractedSamples,
    HorizonModelResult,
    LoadedSamples,
    ProblematicSample,
    SampleWithHorizon,
    TestResult,
)

__all__ = [
    # Config
    "AnalysisConfig",
    "load_analysis_config",
    # Data
    "bucket_samples_by_horizon",
    "class_balanced_train_test_split",
    "extract_samples_with_horizons",
    "filter_problematic_samples",
    "find_preference_data_by_query_id",
    "format_horizon",
    "is_ambiguous_sample",
    "load_samples_from_query_ids",
    "serialize_problematic_samples",
    # Consistency
    "analyze_consistency",
    "extract_constraints",
    "filter_consistent_samples",
    # Training
    "test_model",
    "train_and_test_by_horizon",
    "train_model",
    # Results
    "AnalysisResult",
    "print_summary",
    "serialize_analysis_result",
    # Types
    "ALL_MODEL_SPECS",
    "AmbiguousSample",
    "ChoiceModelSpec",
    "EvaluationMetrics",
    "ExtractedSamples",
    "HorizonModelResult",
    "LoadedSamples",
    "ProblematicSample",
    "SampleWithHorizon",
    "TestResult",
]
