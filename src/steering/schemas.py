"""
Schemas for activation steering configuration.

Steering applies a learned direction vector to model activations during generation,
allowing manipulation of model behavior (e.g., shifting preference toward short-term
or long-term choices).

Example usage:
    config = SteeringConfig(
        direction=probe.direction,
        layer=26,
        strength=100.0,
        option=SteeringOption.APPLY_TO_ALL,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import numpy as np

from ..common.schema_utils import SchemaClass


class SteeringOption(Enum):
    """
    How to apply steering during generation.

    APPLY_TO_ALL: Apply steering vector to all token positions at every step.
        This is the standard approach for activation steering during autoregressive
        generation with KV caching. The steering is added to residual stream at
        every position, which affects the model's behavior throughout generation.

    APPLY_TO_TOKEN_POSITION: Apply steering only at a specific token position.
        Can target a fixed index or wait for a text pattern to appear, then
        apply steering only to the immediately following token.
    """

    APPLY_TO_ALL = "apply_to_all"
    APPLY_TO_TOKEN_POSITION = "apply_to_token_position"


@dataclass
class TokenPositionTarget(SchemaClass):
    """
    Specification for which token position to apply steering to.

    Supports three modes:
    1. Fixed index: Apply at a specific token position (0-indexed from prompt start)
    2. Multiple indices: Apply at multiple specific token positions
    3. Pattern trigger: Wait until a text pattern is generated, then apply to next token

    Examples:
        # Apply at token position 10
        TokenPositionTarget(index=10)

        # Apply at positions 5, 10, and 15
        TokenPositionTarget(indices=[5, 10, 15])

        # Apply after "I select:" is generated
        TokenPositionTarget(pattern="I select:")
    """

    # Fixed position index (0-indexed from start of sequence)
    index: Optional[int] = None

    # Multiple position indices
    indices: Optional[list[int]] = None

    # Pattern to wait for - steering applied to token AFTER this pattern
    pattern: Optional[str] = None

    def __post_init__(self):
        set_count = sum(x is not None for x in [self.index, self.indices, self.pattern])
        if set_count == 0:
            raise ValueError("TokenPositionTarget requires index, indices, or pattern")
        if set_count > 1:
            raise ValueError("TokenPositionTarget can only have one of: index, indices, pattern")
        super().__post_init__()

    @classmethod
    def from_value(cls, value: Union[int, str]) -> "TokenPositionTarget":
        """Create from int (index) or str (pattern)."""
        if isinstance(value, int):
            return cls(index=value)
        elif isinstance(value, str):
            return cls(pattern=value)
        else:
            raise TypeError(f"Expected int or str, got {type(value)}")

    def is_pattern_based(self) -> bool:
        """Whether this target uses pattern matching."""
        return self.pattern is not None


@dataclass
class SteeringConfig:
    """
    Configuration for activation steering during generation.

    This is the main config passed to model generation to enable steering.
    The direction vector is typically extracted from a trained probe.

    Attributes:
        direction: Steering direction vector (shape: [d_model,])
                  Typically the weight vector from a trained linear probe.
        layer: Which transformer layer to apply steering at (0-indexed)
        strength: Scaling factor for the steering vector. Positive values
                 steer toward the probe's positive class, negative toward
                 the negative class. Typical values: 50-200.
        option: How to apply steering (APPLY_TO_ALL or APPLY_TO_TOKEN_POSITION)
        token_position: Target position (required if option is APPLY_TO_TOKEN_POSITION)
    """

    direction: np.ndarray  # Shape: [d_model,]
    layer: int
    strength: float = 1.0
    option: SteeringOption = SteeringOption.APPLY_TO_ALL
    token_position: Optional[TokenPositionTarget] = None

    def __post_init__(self):
        # Normalize direction to unit vector
        norm = np.linalg.norm(self.direction)
        if norm > 0:
            self.direction = self.direction.flatten() / norm

        # Validate token_position requirement
        if self.option == SteeringOption.APPLY_TO_TOKEN_POSITION:
            if self.token_position is None:
                raise ValueError(
                    "token_position required when option is APPLY_TO_TOKEN_POSITION"
                )

    @property
    def hook_name(self) -> str:
        """Get the TransformerLens hook name for this layer."""
        return f"blocks.{self.layer}.hook_resid_post"
