"""
Activation steering module.

Provides configuration schemas and hooks for manipulating model behavior
during generation by adding learned direction vectors to activations.

Main components:
    SteeringConfig: Configuration for steering (direction, layer, strength, option)
    SteeringOption: Enum for how to apply steering (APPLY_TO_ALL, APPLY_TO_TOKEN_POSITION)
    TokenPositionTarget: Target specification for position-based steering
    create_steering_hook: Factory function for creating steering hooks

Example:
    from src.steering import SteeringConfig, SteeringOption, create_steering_hook

    config = SteeringConfig(
        direction=probe.direction,
        layer=26,
        strength=100.0,
        option=SteeringOption.APPLY_TO_ALL,
    )

    hook, _ = create_steering_hook(
        config,
        model_dtype=model.cfg.dtype,
        model_device=model.cfg.device,
    )

    with model.hooks(fwd_hooks=[(config.hook_name, hook)]):
        output = model.generate(...)
"""

from .hooks import create_steering_hook, PatternMatcher
from .schemas import SteeringConfig, SteeringOption, TokenPositionTarget

__all__ = [
    "SteeringConfig",
    "SteeringOption",
    "TokenPositionTarget",
    "create_steering_hook",
    "PatternMatcher",
]
