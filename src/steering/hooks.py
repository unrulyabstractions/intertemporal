"""
TransformerLens hook implementations for activation steering.

Provides hooks that modify residual stream activations during generation
to steer model behavior toward desired outputs.

These hooks are designed for use with HookedTransformer.generate() which
uses KV caching for efficiency. During cached generation, only the newest
token's activations are computed at each step (shape: [batch, 1, d_model]),
so hooks must handle this appropriately.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from .schemas import SteeringConfig, SteeringOption, TokenPositionTarget


def create_steering_hook(
    config: SteeringConfig,
    model_dtype: torch.dtype,
    model_device: str,
    tokenizer: Optional[object] = None,
) -> tuple[Callable, Optional[object]]:
    """
    Create a steering hook based on the configuration.

    Args:
        config: Steering configuration specifying direction, strength, option
        model_dtype: Model's data type (e.g., torch.float16)
        model_device: Device to place tensors on (e.g., "cuda", "mps")
        tokenizer: Required for pattern-based token position targeting

    Returns:
        Tuple of (hook_function, state_object_or_none)
        The state object is used for pattern-based targeting to track generation.
    """
    # Convert direction to tensor
    direction_tensor = torch.tensor(
        config.direction,
        dtype=model_dtype,
        device=model_device,
    )

    if config.option == SteeringOption.APPLY_TO_ALL:
        return _create_apply_to_all_hook(direction_tensor, config.strength, debug=True), None
    else:
        return _create_token_position_hook(
            direction_tensor,
            config.strength,
            config.token_position,
            tokenizer,
        )


def _create_apply_to_all_hook(
    direction: torch.Tensor,
    strength: float,
    debug: bool = False,
) -> Callable:
    """
    Create hook that applies steering to all positions.

    This is the standard approach for activation steering during generation.
    During KV-cached generation, only the new token position is processed
    (activation shape: [batch, 1, d_model]), so we add the steering vector
    to all positions in the batch.

    Args:
        direction: Unit direction vector [d_model]
        strength: Scaling factor for steering
        debug: If True, print debug info on first call

    Returns:
        Hook function compatible with TransformerLens
    """
    scaled_direction = strength * direction
    call_count = [0]  # Use list for mutable closure

    def hook(activation: torch.Tensor, hook=None) -> torch.Tensor:
        # activation shape: [batch, seq_len, d_model]
        # During generation with KV cache, seq_len is typically 1
        # Add steering to ALL positions
        # Note: hook parameter is passed by TransformerLens but not used here
        if debug and call_count[0] == 0:
            print(f"        [HOOK] First call: activation shape={activation.shape}, "
                  f"direction shape={scaled_direction.shape}, "
                  f"steering mag={torch.norm(scaled_direction).item():.2f}")
        call_count[0] += 1
        activation[:, :, :] += scaled_direction
        return activation

    return hook


def _create_token_position_hook(
    direction: torch.Tensor,
    strength: float,
    target: TokenPositionTarget,
    tokenizer: Optional[object],
) -> tuple[Callable, Optional["PatternMatcher"]]:
    """
    Create hook that applies steering only at a specific position.

    For index-based targets: applies only when processing that specific position.
    For pattern-based targets: tracks generated text and applies after pattern match.

    Args:
        direction: Unit direction vector [d_model]
        strength: Scaling factor for steering
        target: Target position specification
        tokenizer: Required for pattern-based matching

    Returns:
        Tuple of (hook_function, pattern_matcher_or_none)
    """
    scaled_direction = strength * direction

    if target.index is not None:
        # Fixed index targeting
        target_index = target.index

        def hook(activation: torch.Tensor, hook=None) -> torch.Tensor:
            # Only apply if we're processing the target position
            # During full forward pass, check if target_index is within sequence
            # Note: hook parameter is passed by TransformerLens but not used here
            seq_len = activation.shape[1]
            if seq_len > 1:
                # Full sequence - apply at specific position
                if target_index < seq_len:
                    activation[:, target_index, :] += scaled_direction
            # During generation (seq_len=1), we can't know which position
            # we're at without additional state tracking, so skip
            return activation

        return hook, None

    else:
        # Pattern-based targeting
        if tokenizer is None:
            raise ValueError("tokenizer required for pattern-based steering")

        matcher = PatternMatcher(target.pattern, tokenizer)

        def hook(activation: torch.Tensor, hook=None) -> torch.Tensor:
            # Check if we should apply steering on this step
            # Note: hook parameter is passed by TransformerLens but not used here
            if matcher.should_apply_now():
                activation[:, :, :] += scaled_direction
                matcher.applied()
            return activation

        return hook, matcher


class PatternMatcher:
    """
    Tracks generated tokens and signals when pattern is matched.

    Used for pattern-based token position targeting. After the pattern
    is fully generated, signals that steering should be applied to the
    next token only.

    Usage:
        1. Create matcher with pattern and tokenizer
        2. Call update_generated() after each token is generated
        3. Check should_apply_now() before applying steering
        4. Call applied() after steering is applied
    """

    def __init__(self, pattern: str, tokenizer: object):
        """
        Initialize pattern matcher.

        Args:
            pattern: Text pattern to watch for
            tokenizer: Tokenizer for decoding generated tokens
        """
        self.pattern = pattern
        self.tokenizer = tokenizer
        self.generated_text = ""
        self._apply_next = False
        self._already_applied = False

    def update_generated(self, new_token_ids: torch.Tensor) -> None:
        """
        Update with newly generated tokens.

        Call this after each generation step with the new token(s).

        Args:
            new_token_ids: Tensor of new token IDs (shape: [n_tokens])
        """
        if self._already_applied:
            return

        # Decode new tokens and append to generated text
        new_text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)
        self.generated_text += new_text

        # Check if pattern just completed
        if not self._apply_next and self.pattern in self.generated_text:
            self._apply_next = True

    def should_apply_now(self) -> bool:
        """Whether steering should be applied on this step."""
        return self._apply_next and not self._already_applied

    def applied(self) -> None:
        """Mark that steering was applied."""
        self._apply_next = False
        self._already_applied = True

    def reset(self) -> None:
        """Reset matcher state for new generation."""
        self.generated_text = ""
        self._apply_next = False
        self._already_applied = False
