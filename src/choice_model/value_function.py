"""
Value functions for intertemporal choice models.

U(o_i; θ) = u(r_i) * D(t_i; θ)

where:
    u(r) : utility of reward
    D(t; θ) : discount function over time

Predicted choice:
    o_chosen = argmax U(o_i; θ)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

from .discount_function import DiscountFunction, ExponentialDiscount
from ..types import (
    DiscountFunctionParams,
    DiscountType,
    IntertemporalOption,
    PreferencePair,
    PreferenceQuestion,
    TrainingResult,
    TrainingSample,
    UtilityType,
    ValueFunctionParams,
)


# =============================================================================
# Utility Functions
# =============================================================================


class UtilityFunction(ABC):
    """Abstract base for utility functions u(r)."""

    @abstractmethod
    def __call__(self, reward: float) -> float:
        """Compute utility of reward."""
        pass


@dataclass
class LinearUtility(UtilityFunction):
    """Linear utility: u(r) = r"""

    def __call__(self, reward: float) -> float:
        return reward


@dataclass
class LogUtility(UtilityFunction):
    """Logarithmic utility: u(r) = log(r)"""

    def __call__(self, reward: float) -> float:
        if reward <= 0:
            return float("-inf")
        return math.log(reward)


@dataclass
class PowerUtility(UtilityFunction):
    """Power utility: u(r) = r^α"""

    alpha: float = 0.5

    def __call__(self, reward: float) -> float:
        if reward < 0:
            return -((-reward) ** self.alpha)
        return reward**self.alpha


def create_utility(utility_type: UtilityType, alpha: float = 1.0) -> UtilityFunction:
    """Create utility function from type."""
    if utility_type == UtilityType.LINEAR:
        return LinearUtility()
    elif utility_type == UtilityType.LOG:
        return LogUtility()
    elif utility_type == UtilityType.POWER:
        return PowerUtility(alpha=alpha)
    else:
        raise ValueError(f"Unknown utility type: {utility_type}")


# =============================================================================
# Value Function
# =============================================================================


@dataclass
class ValueFunction:
    """
    Value function for intertemporal choice: U(o; θ) = u(r) * D(t; θ)

    Attributes:
        utility: Utility function u(r)
        discount: Discount function D(t; θ)
        params: Current parameters
    """

    utility: UtilityFunction = field(default_factory=LinearUtility)
    discount: DiscountFunction = field(default_factory=ExponentialDiscount)
    params: ValueFunctionParams = field(default_factory=ValueFunctionParams)

    @classmethod
    def from_params(cls, params: ValueFunctionParams) -> "ValueFunction":
        """Create value function from parameters."""
        utility = create_utility(params.utility_type, params.alpha)
        discount = DiscountFunction.from_params(params.discount)
        return cls(utility=utility, discount=discount, params=params)

    def value(self, option: IntertemporalOption) -> float:
        """
        Compute value of an intertemporal option.

        U(o) = u(r) * D(t)

        Args:
            option: IntertemporalOption with time and reward

        Returns:
            Discounted utility value
        """
        r = option.reward.value
        t = option.time.to_years()
        return self.utility(r) * self.discount(t)

    def predict(self, pair: PreferencePair) -> Literal["short_term", "long_term"]:
        """
        Predict which option would be chosen.

        o_chosen = argmax U(o_i; θ)

        Args:
            pair: PreferencePair with short_term and long_term options

        Returns:
            "short_term" or "long_term" indicating predicted choice
        """
        v_short = self.value(pair.short_term)
        v_long = self.value(pair.long_term)

        if v_short >= v_long:
            return "short_term"
        return "long_term"

    def predict_question(
        self, question: PreferenceQuestion
    ) -> Literal["short_term", "long_term"]:
        """
        Predict choice for a preference question.

        Args:
            question: PreferenceQuestion with pair and time_horizon

        Returns:
            "short_term" or "long_term" indicating predicted choice
        """
        return self.predict(question.pair)

    def choice_probability(
        self, pair: PreferencePair, temperature: float = 0.0
    ) -> tuple[float, float]:
        """
        Compute choice probabilities using softmax.

        P(short_term) = exp(U_s/τ) / (exp(U_s/τ) + exp(U_l/τ))

        Args:
            pair: PreferencePair
            temperature: Softmax temperature (higher = more random)

        Returns:
            Tuple of (P(short_term), P(long_term))
        """
        v_short = self.value(pair.short_term)
        v_long = self.value(pair.long_term)

        # Softmax with temperature
        if temperature <= 0:
            # Deterministic
            if v_short > v_long:
                return (1.0, 0.0)
            elif v_long > v_short:
                return (0.0, 1.0)
            return (0.5, 0.5)

        # Numerical stability
        max_v = max(v_short, v_long)
        exp_short = math.exp((v_short - max_v) / temperature)
        exp_long = math.exp((v_long - max_v) / temperature)
        total = exp_short + exp_long

        return (exp_short / total, exp_long / total)

    def train(
        self,
        samples: list[TrainingSample],
        learning_rate: float = 0.01,
        num_iterations: int = 100,
        temperature: float = 1.0,
    ) -> TrainingResult:
        """
        Train value function parameters to match observed choices.

        Maximizes:
            L(θ) = Σ log P(o_chosen | q_n, θ)

        Currently supports training theta parameter for exponential discount.

        Args:
            samples: List of TrainingSample with questions and observed choices
            learning_rate: Gradient descent learning rate
            num_iterations: Number of training iterations
            temperature: Softmax temperature for choice probabilities

        Returns:
            TrainingResult with fitted parameters and metrics
        """
        if not samples:
            return TrainingResult(
                params=self.params,
                loss=0.0,
                num_samples=0,
                accuracy=0.0,
            )

        # Use small positive temperature for training (gradient requires it)
        # Temperature <= 0 means "nearly deterministic"
        train_temp = max(temperature, 0.01)

        # Simple gradient descent on theta for exponential discount
        theta = self.params.discount.theta

        for _ in range(num_iterations):
            grad = 0.0
            for sample in samples:
                pair = sample.question.pair
                chosen = sample.chosen

                # Compute values
                v_short = self._value_with_theta(pair.short_term, theta)
                v_long = self._value_with_theta(pair.long_term, theta)

                # Choice probabilities
                p_short, p_long = self._softmax(v_short, v_long, train_temp)

                # Gradient of log-likelihood
                if chosen in ("short_term", "a", pair.short_term.label):
                    # Want to increase P(short_term)
                    t_short = pair.short_term.time.to_years()
                    t_long = pair.long_term.time.to_years()
                    grad += p_long * (t_long * v_long - t_short * v_short) / train_temp
                else:
                    # Want to increase P(long_term)
                    t_short = pair.short_term.time.to_years()
                    t_long = pair.long_term.time.to_years()
                    grad += (
                        p_short * (t_short * v_short - t_long * v_long) / train_temp
                    )

            # Update theta
            theta = max(0.001, theta + learning_rate * grad / len(samples))

        # Update parameters
        new_discount_params = DiscountFunctionParams(
            discount_type=self.params.discount.discount_type,
            theta=theta,
            beta=self.params.discount.beta,
            delta=self.params.discount.delta,
        )
        new_params = ValueFunctionParams(
            utility_type=self.params.utility_type,
            alpha=self.params.alpha,
            discount=new_discount_params,
        )

        # Update self
        self.params = new_params
        self.discount = DiscountFunction.from_params(new_discount_params)

        # Compute final loss and accuracy
        loss = self._compute_loss(samples, train_temp)
        accuracy = self._compute_accuracy(samples)

        return TrainingResult(
            params=new_params,
            loss=loss,
            num_samples=len(samples),
            accuracy=accuracy,
        )

    def _value_with_theta(self, option: IntertemporalOption, theta: float) -> float:
        """Compute value with specific theta, using the correct discount formula."""
        r = option.reward.value
        t = option.time.to_years()
        u = self.utility(r)

        # Use the correct discount formula based on discount type
        discount_type = self.params.discount.discount_type
        if discount_type == DiscountType.EXPONENTIAL:
            d = math.exp(-theta * t)
        elif discount_type == DiscountType.HYPERBOLIC:
            d = 1.0 / (1.0 + theta * t)
        elif discount_type == DiscountType.QUASI_HYPERBOLIC:
            # For quasi-hyperbolic, theta controls delta (per-period discount)
            # D(t) = β * δ^t, where we train δ = exp(-theta)
            # This maps theta to a decay rate similar to exponential
            beta = self.params.discount.beta
            delta = math.exp(-theta)  # Map theta to delta
            d = beta * (delta ** t) if t > 0 else 1.0
        else:
            d = math.exp(-theta * t)  # Default to exponential

        return u * d

    def _softmax(self, v1: float, v2: float, temperature: float) -> tuple[float, float]:
        """Compute softmax probabilities. Temperature=0 gives deterministic (argmax)."""
        if temperature <= 0:
            # Deterministic: 100% to higher value (with tie-breaking)
            if v1 > v2:
                return (1.0, 0.0)
            elif v2 > v1:
                return (0.0, 1.0)
            else:
                return (0.5, 0.5)  # Tie
        max_v = max(v1, v2)
        exp1 = math.exp((v1 - max_v) / temperature)
        exp2 = math.exp((v2 - max_v) / temperature)
        total = exp1 + exp2
        return (exp1 / total, exp2 / total)

    def _compute_loss(self, samples: list[TrainingSample], temperature: float) -> float:
        """Compute negative log-likelihood loss."""
        total_loss = 0.0
        for sample in samples:
            pair = sample.question.pair
            chosen = sample.chosen

            p_short, p_long = self.choice_probability(pair, temperature)

            if chosen in ("short_term", "a", pair.short_term.label):
                total_loss -= math.log(max(p_short, 1e-10))
            else:
                total_loss -= math.log(max(p_long, 1e-10))

        return total_loss / len(samples)

    def _compute_accuracy(self, samples: list[TrainingSample]) -> float:
        """Compute prediction accuracy."""
        correct = 0
        for sample in samples:
            predicted = self.predict(sample.question.pair)
            chosen = sample.chosen

            # Normalize chosen to short_term/long_term
            if chosen in ("short_term", "a", sample.question.pair.short_term.label):
                chosen_normalized = "short_term"
            elif chosen in ("long_term", "b", sample.question.pair.long_term.label):
                chosen_normalized = "long_term"
            else:
                # Unknown format, skip
                continue

            if predicted == chosen_normalized:
                correct += 1

        return correct / len(samples) if samples else 0.0

    def get_params(self) -> ValueFunctionParams:
        """Get current parameters."""
        return self.params
