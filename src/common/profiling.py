"""
Profiling utilities for tracking performance metrics.

Provides decorators, context managers, and helpers for profiling
operations across the codebase.
"""

from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Optional, Any
from pathlib import Path

from .schema_utils import SchemaClass
from .schemas import SCHEMA_VERSION


# =============================================================================
# Profiling Schemas
# =============================================================================


@dataclass
class TimingRecord(SchemaClass):
    """
    Single timing measurement.

    Attributes:
        operation: Name of the operation
        duration_ms: Duration in milliseconds
        timestamp: When the operation started (epoch)
        metadata: Optional additional info
    """

    operation: str
    duration_ms: float
    timestamp: float
    metadata: dict = field(default_factory=dict)


@dataclass
class OperationStats(SchemaClass):
    """
    Aggregated statistics for an operation.

    Attributes:
        operation: Name of the operation
        call_count: Number of times called
        total_ms: Total time in milliseconds
        min_ms: Minimum duration
        max_ms: Maximum duration
        avg_ms: Average duration
    """

    operation: str
    call_count: int
    total_ms: float
    min_ms: float
    max_ms: float
    avg_ms: float


@dataclass
class ProfilingSession(SchemaClass):
    """
    Complete profiling session data.

    Attributes:
        version: Schema version
        session_id: Unique session identifier
        start_time: Session start timestamp
        end_time: Session end timestamp (0 if ongoing)
        records: List of timing records
        stats: Aggregated stats per operation
    """

    version: str
    session_id: str
    start_time: float
    end_time: float = 0.0
    records: list[TimingRecord] = field(default_factory=list)
    stats: dict[str, OperationStats] = field(default_factory=dict)


# =============================================================================
# Profiler Class
# =============================================================================


class Profiler:
    """
    Profiler for tracking operation timings.

    Usage:
        profiler = Profiler()

        # Using decorator
        @profiler.profile("my_operation")
        def my_function():
            ...

        # Using context manager
        with profiler.measure("my_operation"):
            ...

        # Get results
        session = profiler.get_session()
    """

    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize profiler.

        Args:
            session_id: Optional session identifier (auto-generated if not provided)
        """
        self._start_time = time.time()
        self._session_id = session_id or f"session_{int(self._start_time * 1000)}"
        self._records: list[TimingRecord] = []
        self._enabled = True

    def enable(self) -> None:
        """Enable profiling."""
        self._enabled = True

    def disable(self) -> None:
        """Disable profiling."""
        self._enabled = False

    @contextmanager
    def measure(self, operation: str, metadata: Optional[dict] = None):
        """
        Context manager for timing an operation.

        Args:
            operation: Name of the operation
            metadata: Optional metadata to attach

        Example:
            with profiler.measure("load_config"):
                config = load_config(path)
        """
        if not self._enabled:
            yield
            return

        start = time.time()
        try:
            yield
        finally:
            duration_ms = (time.time() - start) * 1000
            record = TimingRecord(
                operation=operation,
                duration_ms=duration_ms,
                timestamp=start,
                metadata=metadata or {},
            )
            self._records.append(record)

    def profile(self, operation: str, metadata: Optional[dict] = None) -> Callable:
        """
        Decorator for profiling a function.

        Args:
            operation: Name of the operation
            metadata: Optional metadata to attach

        Example:
            @profiler.profile("generate_samples")
            def generate_samples(config):
                ...
        """

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                with self.measure(operation, metadata):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    def record(
        self, operation: str, duration_ms: float, metadata: Optional[dict] = None
    ) -> None:
        """
        Manually record a timing.

        Args:
            operation: Name of the operation
            duration_ms: Duration in milliseconds
            metadata: Optional metadata
        """
        if not self._enabled:
            return

        record = TimingRecord(
            operation=operation,
            duration_ms=duration_ms,
            timestamp=time.time(),
            metadata=metadata or {},
        )
        self._records.append(record)

    def _compute_stats(self) -> dict[str, OperationStats]:
        """Compute aggregated statistics per operation."""
        from collections import defaultdict

        # Group by operation
        by_op: dict[str, list[float]] = defaultdict(list)
        for record in self._records:
            by_op[record.operation].append(record.duration_ms)

        # Compute stats
        stats = {}
        for op, durations in by_op.items():
            stats[op] = OperationStats(
                operation=op,
                call_count=len(durations),
                total_ms=sum(durations),
                min_ms=min(durations),
                max_ms=max(durations),
                avg_ms=sum(durations) / len(durations),
            )

        return stats

    def get_session(self) -> ProfilingSession:
        """
        Get complete profiling session.

        Returns:
            ProfilingSession with all records and computed stats
        """
        return ProfilingSession(
            version=SCHEMA_VERSION,
            session_id=self._session_id,
            start_time=self._start_time,
            end_time=time.time(),
            records=self._records.copy(),
            stats=self._compute_stats(),
        )

    def get_stats(self) -> dict[str, OperationStats]:
        """Get aggregated stats only."""
        return self._compute_stats()

    def get_records(self) -> list[TimingRecord]:
        """Get raw timing records."""
        return self._records.copy()

    def clear(self) -> None:
        """Clear all records."""
        self._records.clear()

    def summary(self) -> str:
        """
        Get human-readable summary.

        Returns:
            Formatted summary string
        """
        stats = self._compute_stats()
        if not stats:
            return "No profiling data recorded."

        lines = ["Profiling Summary", "=" * 50]

        # Sort by total time descending
        sorted_ops = sorted(stats.values(), key=lambda s: s.total_ms, reverse=True)

        for s in sorted_ops:
            lines.append(
                f"{s.operation}: {s.call_count} calls, "
                f"total={s.total_ms:.2f}ms, "
                f"avg={s.avg_ms:.2f}ms, "
                f"min={s.min_ms:.2f}ms, "
                f"max={s.max_ms:.2f}ms"
            )

        total_time = sum(s.total_ms for s in sorted_ops)
        lines.append("=" * 50)
        lines.append(f"Total profiled time: {total_time:.2f}ms")

        return "\n".join(lines)


# =============================================================================
# Global Profiler Instance
# =============================================================================

# Default global profiler (can be replaced or disabled)
_global_profiler: Optional[Profiler] = None


def get_profiler() -> Profiler:
    """
    Get the global profiler instance.

    Creates one if it doesn't exist.
    """
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = Profiler()
    return _global_profiler


def set_profiler(profiler: Optional[Profiler]) -> None:
    """Set the global profiler instance."""
    global _global_profiler
    _global_profiler = profiler


def reset_profiler() -> None:
    """Reset the global profiler to a new instance."""
    global _global_profiler
    _global_profiler = Profiler()


# =============================================================================
# Convenience Decorators
# =============================================================================


def profile(operation: str, metadata: Optional[dict] = None) -> Callable:
    """
    Decorator using global profiler.

    Args:
        operation: Name of the operation
        metadata: Optional metadata

    Example:
        @profile("load_config")
        def load_config(path):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            profiler = get_profiler()
            with profiler.measure(operation, metadata):
                return func(*args, **kwargs)

        return wrapper

    return decorator


@contextmanager
def measure(operation: str, metadata: Optional[dict] = None):
    """
    Context manager using global profiler.

    Args:
        operation: Name of the operation
        metadata: Optional metadata

    Example:
        with measure("process_data"):
            process(data)
    """
    profiler = get_profiler()
    with profiler.measure(operation, metadata):
        yield


# =============================================================================
# I/O Functions
# =============================================================================


def save_profiling_session(session: ProfilingSession, path: Path) -> None:
    """
    Save profiling session to JSON file.

    Sample JSON (see schema: ProfilingSession):
    {
        "version": "1.0",
        "session_id": "session_1703876543000",
        "start_time": 1703876543.0,
        "end_time": 1703876544.5,
        "records": [
            {
                "operation": "generate_samples",
                "duration_ms": 150.5,
                "timestamp": 1703876543.1,
                "metadata": {}
            },
            ...
        ],
        "stats": {
            "generate_samples": {
                "operation": "generate_samples",
                "call_count": 1,
                "total_ms": 150.5,
                "min_ms": 150.5,
                "max_ms": 150.5,
                "avg_ms": 150.5
            },
            ...
        }
    }
    """
    from dataclasses import asdict
    from .io import save_json, ensure_dir

    ensure_dir(path.parent)
    save_json(asdict(session), path, readable_text=False)


def load_profiling_session(path: Path) -> ProfilingSession:
    """
    Load profiling session from JSON file.

    See save_profiling_session for sample JSON format.
    """
    from .io import load_json

    data = load_json(path)

    # Validate version
    file_version = data.get("version", "unknown")
    if file_version != SCHEMA_VERSION:
        raise ValueError(
            f"Version mismatch: file has version '{file_version}', "
            f"expected '{SCHEMA_VERSION}'"
        )

    # Reconstruct records
    records = [TimingRecord(**r) for r in data.get("records", [])]

    # Reconstruct stats
    stats = {
        op: OperationStats(**s) for op, s in data.get("stats", {}).items()
    }

    return ProfilingSession(
        version=file_version,
        session_id=data["session_id"],
        start_time=data["start_time"],
        end_time=data.get("end_time", 0.0),
        records=records,
        stats=stats,
    )
