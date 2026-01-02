"""
Discount functions for intertemporal choice models.

D(t; θ) : discount function over time (0 ≤ D(t) ≤ 1, decreasing)

Supported types:
    - Exponential: D(t) = exp(-θt)
    - Hyperbolic: D(t) = 1 / (1 + θt)
    - Quasi-hyperbolic: D(t) = β * δ^t (for t > 0), D(0) = 1
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..types import DiscountFunctionParams, DiscountType, TimeValue


# =============================================================================
# Abstract Base
# =============================================================================


class DiscountFunction(ABC):
    """
    Abstract base class for discount functions.

    D(t; θ) maps time to discount factor in [0, 1].
    """

    @abstractmethod
    def __call__(self, t: float) -> float:
        """
        Compute discount factor for time t.

        Args:
            t: Time value (in years)

        Returns:
            Discount factor in [0, 1]
        """
        pass

    def discount(self, time: TimeValue) -> float:
        """
        Compute discount factor for TimeValue.

        Args:
            time: TimeValue object

        Returns:
            Discount factor in [0, 1]
        """
        return self(time.to_years())

    @abstractmethod
    def get_params(self) -> DiscountFunctionParams:
        """Get parameters as schema object."""
        pass

    @classmethod
    def from_params(cls, params: DiscountFunctionParams) -> "DiscountFunction":
        """
        Create discount function from parameters.

        Args:
            params: DiscountFunctionParams schema

        Returns:
            DiscountFunction instance
        """
        if params.discount_type == DiscountType.EXPONENTIAL:
            return ExponentialDiscount(theta=params.theta)
        elif params.discount_type == DiscountType.HYPERBOLIC:
            return HyperbolicDiscount(theta=params.theta)
        elif params.discount_type == DiscountType.QUASI_HYPERBOLIC:
            return QuasiHyperbolicDiscount(beta=params.beta, delta=params.delta)
        elif params.discount_type == DiscountType.GENERALIZED_HYPERBOLIC:
            return GeneralizedHyperbolicDiscount(alpha=params.theta, gamma=params.gamma)
        elif params.discount_type == DiscountType.DOUBLE_EXPONENTIAL:
            return DoubleExponentialDiscount(omega=params.omega, r_fast=params.theta, r_slow=params.r_slow)
        elif params.discount_type == DiscountType.CONSTANT_SENSITIVITY:
            return ConstantSensitivityDiscount(r=params.theta, s=params.sensitivity)
        else:
            raise ValueError(f"Unknown discount type: {params.discount_type}")


# =============================================================================
# Implementations
# =============================================================================


@dataclass
class ExponentialDiscount(DiscountFunction):
    """
    Exponential discount function: D(t) = exp(-θt)

    Standard economic discounting with constant discount rate.

    Attributes:
        theta: Discount rate (higher = more impatient)
    """

    theta: float = 0.1

    def __call__(self, t: float) -> float:
        return math.exp(-self.theta * t)

    def get_params(self) -> DiscountFunctionParams:
        return DiscountFunctionParams(
            discount_type=DiscountType.EXPONENTIAL,
            theta=self.theta,
        )


@dataclass
class HyperbolicDiscount(DiscountFunction):
    """
    Hyperbolic discount function: D(t) = 1 / (1 + θt)

    Models decreasing impatience - discount rate decreases over time.

    Attributes:
        theta: Discount rate parameter
    """

    theta: float = 0.1

    def __call__(self, t: float) -> float:
        return 1.0 / (1.0 + self.theta * t)

    def get_params(self) -> DiscountFunctionParams:
        return DiscountFunctionParams(
            discount_type=DiscountType.HYPERBOLIC,
            theta=self.theta,
        )


@dataclass
class QuasiHyperbolicDiscount(DiscountFunction):
    """
    Quasi-hyperbolic (beta-delta) discount function.

    D(0) = 1
    D(t) = β * δ^t for t > 0

    Models present bias with β < 1.

    Attributes:
        beta: Present bias parameter (typically < 1)
        delta: Per-period discount factor (typically close to 1)
    """

    beta: float = 0.7
    delta: float = 0.99

    def __call__(self, t: float) -> float:
        if t <= 0:
            return 1.0
        return self.beta * (self.delta**t)

    def get_params(self) -> DiscountFunctionParams:
        return DiscountFunctionParams(
            discount_type=DiscountType.QUASI_HYPERBOLIC,
            beta=self.beta,
            delta=self.delta,
        )


@dataclass
class GeneralizedHyperbolicDiscount(DiscountFunction):
    """
    Generalized hyperbolic discount function (Loewenstein-Prelec).

    D(t) = 1 / (1 + α*t)^(γ/α)

    Special cases:
        - γ = α: Simple hyperbolic
        - α → 0: Exponential

    Attributes:
        alpha: Departure from exponential (α > 0)
        gamma: Controls rate of discounting (γ > 0)
    """

    alpha: float = 1.0
    gamma: float = 1.0

    def __call__(self, t: float) -> float:
        if self.alpha <= 0 or self.gamma <= 0:
            return math.exp(-self.gamma * t)  # Fallback to exponential
        return 1.0 / ((1.0 + self.alpha * t) ** (self.gamma / self.alpha))

    def get_params(self) -> DiscountFunctionParams:
        return DiscountFunctionParams(
            discount_type=DiscountType.GENERALIZED_HYPERBOLIC,
            theta=self.alpha,
            gamma=self.gamma,
        )


@dataclass
class DoubleExponentialDiscount(DiscountFunction):
    """
    Double exponential (dual-system) discount function.

    D(t) = ω*exp(-r_fast*t) + (1-ω)*exp(-r_slow*t)

    Models dual-process theory (System 1 vs System 2):
        - Fast system: High discount rate (impulsive)
        - Slow system: Low discount rate (deliberative)

    Attributes:
        omega: Weight on fast/impulsive system (0 < ω < 1)
        r_fast: Discount rate for fast system
        r_slow: Discount rate for slow system (r_slow < r_fast)
    """

    omega: float = 0.3
    r_fast: float = 1.0
    r_slow: float = 0.01

    def __call__(self, t: float) -> float:
        fast = self.omega * math.exp(-self.r_fast * t)
        slow = (1 - self.omega) * math.exp(-self.r_slow * t)
        return fast + slow

    def get_params(self) -> DiscountFunctionParams:
        return DiscountFunctionParams(
            discount_type=DiscountType.DOUBLE_EXPONENTIAL,
            theta=self.r_fast,
            omega=self.omega,
            r_slow=self.r_slow,
        )


@dataclass
class ConstantSensitivityDiscount(DiscountFunction):
    """
    Constant sensitivity (Ebert-Prelec) discount function.

    D(t) = exp(-(r*t)^s)

    Models subjective time perception:
        - s = 1: Exponential discounting
        - s < 1: Extended present (less sensitive to time)
        - s > 1: Compressed future

    Attributes:
        r: Baseline discount rate (r > 0)
        s: Time sensitivity parameter (s > 0)
    """

    r: float = 0.1
    s: float = 0.5

    def __call__(self, t: float) -> float:
        if t <= 0:
            return 1.0
        return math.exp(-((self.r * t) ** self.s))

    def get_params(self) -> DiscountFunctionParams:
        return DiscountFunctionParams(
            discount_type=DiscountType.CONSTANT_SENSITIVITY,
            theta=self.r,
            sensitivity=self.s,
        )


# =============================================================================
# Utility Functions
# =============================================================================


def compute_internal_horizon(
    discount: DiscountFunction,
    threshold: float = 0.5,
    max_time: float = 100.0,
) -> float:
    """
    Compute internal time horizon from discount function.

    t_internal = inf{t >= 0 : D(t) <= α} for threshold α

    Args:
        discount: Discount function
        threshold: Discount threshold α
        max_time: Maximum time to search

    Returns:
        Internal time horizon in years
    """
    # Binary search for threshold crossing
    low, high = 0.0, max_time

    # Check if threshold is ever reached
    if discount(max_time) > threshold:
        return max_time

    while high - low > 0.001:
        mid = (low + high) / 2
        if discount(mid) > threshold:
            low = mid
        else:
            high = mid

    return high
