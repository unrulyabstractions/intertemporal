"""
Shared formatting-dependent boundary markers for visualization and steering.

These markers map formatting_id to keywords used to identify boundaries
in the prompt/response for:
- Token position selection (which probes to use)
- Heatmap boundary lines
- Steering experiments

Update FORMATTING_BOUNDARY_MARKERS when formatting configs change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class BoundaryMarkers:
    """
    Boundary markers for a specific formatting configuration.

    Each marker specifies:
    - text: The keyword/text pattern to search for
    - description: Human-readable description of what this boundary represents

    Boundaries define positions where model behavior changes:
    - choices_presented: Options are shown to the model (e.g., "CONSIDER:")
    - time_horizon: Time horizon context is injected (e.g., "ACTION:")
    - choice_made: Model has made its selection (e.g., "My reasoning:")
    """
    # Prompt boundaries
    choices_presented: str  # Text marking where options are presented
    time_horizon: str       # Text marking where time horizon is injected (after_time_horizon_spec)

    # Response boundaries
    choice_made: str        # Text marking where choice/reasoning begins

    # Token position text patterns for probe selection
    # These are the actual text snippets used in token_position configs
    after_horizon_token_text: str = "ACTION"  # Token position right after time horizon
    first_response_token_text: str = "I select"  # First meaningful response token


# Maps formatting_id -> boundary markers
# Update when formatting configs change or new ones are added
FORMATTING_BOUNDARY_MARKERS: dict[str, BoundaryMarkers] = {
    # default_formatting (0c3aae84eef4bcf3a62cb05ec7439583)
    "0c3aae84eef4bcf3a62cb05ec7439583": BoundaryMarkers(
        choices_presented="CONSIDER",
        time_horizon="ACTION",  # Has after_time_horizon_spec=true
        choice_made="reasoning",
        after_horizon_token_text="ACTION",
        first_response_token_text="I select",
    ),
}

# Default markers if formatting_id not found
DEFAULT_BOUNDARY_MARKERS = BoundaryMarkers(
    choices_presented="CONSIDER",
    time_horizon="ACTION",
    choice_made="reasoning",
    after_horizon_token_text="ACTION",
    first_response_token_text="I select",
)


def get_boundary_markers(formatting_id: str) -> BoundaryMarkers:
    """
    Get boundary markers for a formatting configuration.

    Args:
        formatting_id: The formatting config ID (hash)

    Returns:
        BoundaryMarkers for this formatting, or default if not found
    """
    return FORMATTING_BOUNDARY_MARKERS.get(formatting_id, DEFAULT_BOUNDARY_MARKERS)


def find_probe_indices_after_horizon(
    token_position_specs: list,
    markers: Optional[BoundaryMarkers] = None,
) -> list[int]:
    """
    Find token position indices that are after the time horizon injection.

    Args:
        token_position_specs: List of TokenPositionSpec from training
        markers: Boundary markers to use (default: DEFAULT_BOUNDARY_MARKERS)

    Returns:
        List of token position indices that are after time horizon
    """
    if markers is None:
        markers = DEFAULT_BOUNDARY_MARKERS

    after_horizon_indices = []
    found_horizon = False

    for i, spec in enumerate(token_position_specs):
        s = spec.spec if hasattr(spec, 'spec') else spec

        if isinstance(s, dict):
            # Check for after_time_horizon_spec marker
            if s.get("after_time_horizon_spec", False):
                found_horizon = True
                after_horizon_indices.append(i)
            elif found_horizon:
                # All positions after the marker
                after_horizon_indices.append(i)

            # Also check text match
            text = s.get("text", "")
            if markers.after_horizon_token_text in text:
                found_horizon = True
                if i not in after_horizon_indices:
                    after_horizon_indices.append(i)
        elif found_horizon:
            after_horizon_indices.append(i)

    return after_horizon_indices


def find_first_response_token_index(
    token_position_specs: list,
    markers: Optional[BoundaryMarkers] = None,
) -> Optional[int]:
    """
    Find the token position index for the first response/continuation token.

    Args:
        token_position_specs: List of TokenPositionSpec from training
        markers: Boundary markers to use (default: DEFAULT_BOUNDARY_MARKERS)

    Returns:
        Token position index, or None if not found
    """
    if markers is None:
        markers = DEFAULT_BOUNDARY_MARKERS

    for i, spec in enumerate(token_position_specs):
        s = spec.spec if hasattr(spec, 'spec') else spec

        if isinstance(s, dict):
            text = s.get("text", "")
            location = s.get("location", "")

            # Look for first continuation token
            if location == "continuation" or "continuation" in str(s):
                return i

            # Or look for response marker text
            if markers.first_response_token_text.lower() in text.lower():
                return i

        # Check for continuation_index or index (continuation position)
        if isinstance(s, dict):
            if "continuation_index" in s or ("index" in s and "prompt" not in str(s)):
                return i
        elif isinstance(s, int) and s >= 0:
            # Plain int typically means continuation index
            return i

    return None
