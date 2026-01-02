"""
Intertemporal preference research framework.

Submodules:
- common: I/O, schemas, profiling utilities
- choice_model: Discount and value functions
- probes: Probe training for model internals analysis
"""

# Re-export commonly used items for convenience
from .common.io import ensure_dir, get_timestamp, load_json, save_json
from .common.profiling import get_profiler
from .common.schemas import SCHEMA_VERSION

__all__ = [
    "ensure_dir",
    "get_timestamp",
    "load_json",
    "save_json",
    "get_profiler",
    "SCHEMA_VERSION",
]
