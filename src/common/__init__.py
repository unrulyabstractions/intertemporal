"""
Common utilities for the intertemporal project.

Provides I/O, schemas, profiling, and utilities.
"""

from .io import (
    ensure_dir,
    get_timestamp,
    load_dataset,
    load_json,
    load_jsonl,
    save_json,
    save_jsonl,
)
from .profiling import (
    OperationStats,
    Profiler,
    ProfilingSession,
    TimingRecord,
    get_profiler,
    load_profiling_session,
    measure,
    profile,
    reset_profiler,
    save_profiling_session,
    set_profiler,
)
from .schema_utils import SchemaClass, deterministic_id_from_dataclass
from .schemas import (
    SCHEMA_VERSION,
    ContextConfig,
    DatasetConfig,
    DatasetRunSpec,
    DatasetSpec,
    DecodingConfig,
    FormattingConfig,
    InternalsConfig,
    OptionRangeConfig,
    QueryConfig,
    SingleQuerySpec,
    StepType,
    TokenLocation,
    TokenPosition,
)

__all__ = [
    # io
    "ensure_dir",
    "get_timestamp",
    "load_dataset",
    "load_json",
    "load_jsonl",
    "save_json",
    "save_jsonl",
    # profiling
    "OperationStats",
    "Profiler",
    "ProfilingSession",
    "TimingRecord",
    "get_profiler",
    "load_profiling_session",
    "measure",
    "profile",
    "reset_profiler",
    "save_profiling_session",
    "set_profiler",
    # schema_utils
    "SchemaClass",
    "deterministic_id_from_dataclass",
    # schemas
    "SCHEMA_VERSION",
    "ContextConfig",
    "DatasetConfig",
    "DatasetRunSpec",
    "DatasetSpec",
    "DecodingConfig",
    "FormattingConfig",
    "InternalsConfig",
    "OptionRangeConfig",
    "QueryConfig",
    "SingleQuerySpec",
    "StepType",
    "TokenLocation",
    "TokenPosition",
]
