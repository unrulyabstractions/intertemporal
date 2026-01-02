"""
Internal type definitions for intertemporal preference experiments.

These types are used internally by the codebase and are not directly
serialized to/from JSON config or output files.

Definitions from "Grounding Temporal-Awareness in Intertemporal Preference":
    x       : prompt
    y       : response
    t_i     : time value i
    r_i     : reward at time t_i (scalar or vector)
    t_h     : time horizon target
    s_c     : context string
    s_t     : trace string
    o_i = (t_i, r_i)    : intertemporal option
    q_x = (o_i, o_j)    : preference question for prompt x

Prompting Setup:
    Input/Prompt:   x = (t_h, q_x, s_x)
    Output/Response: y = (o_k, s_t)

Value Function:
    U(o_i; θ) = u(r_i) * D(t_i; θ)
    o_chosen = argmax U(o_i; θ)

Discount Function:
    D(t; θ) : discount function over time (0 ≤ D(t) ≤ 1, decreasing)
    Example: D = exp(-θt)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .common.schema_utils import SchemaClass


# =============================================================================
# Base Value Types
# =============================================================================


@dataclass
class TimeValue(SchemaClass):
    """
    A time value with unit: t_i

    Attributes:
        value: Numeric time value
        unit: Time unit (e.g., "months", "years")
    """

    value: float
    unit: str = "years"

    def to_months(self) -> float:
        """Convert to months for comparison."""
        if self.unit in ("month", "months"):
            return self.value
        elif self.unit in ("year", "years"):
            return self.value * 12
        elif self.unit in ("day", "days"):
            return self.value / 30
        else:
            raise ValueError(f"Unknown time unit: {self.unit}")

    def to_years(self) -> float:
        """Convert to years."""
        return self.to_months() / 12

    def to_list(self) -> list:
        """Convert to [value, unit] list format for JSON serialization."""
        # Use int if whole number
        val = int(self.value) if self.value == int(self.value) else self.value
        return [val, self.unit]

    def __str__(self) -> str:
        # Format value nicely (no decimal for whole numbers)
        if self.value == int(self.value):
            val_str = str(int(self.value))
        else:
            val_str = f"{self.value:.1f}"

        # Singular/plural unit
        unit = self.unit
        if self.value == 1:
            unit = unit.rstrip("s")  # "years" -> "year"

        return f"{val_str} {unit}"


@dataclass
class RewardValue(SchemaClass):
    """
    A reward value with unit: r_i

    Attributes:
        value: Numeric reward value
        unit: Reward unit (e.g., "dollars", "housing units")
    """

    value: float
    unit: str = ""

    def __str__(self) -> str:
        if self.unit:
            return f"{self.value:,.0f} {self.unit}"
        return f"{self.value:,.0f}"


# =============================================================================
# Intertemporal Option & Preference Schemas
# =============================================================================


@dataclass
class IntertemporalOption(SchemaClass):
    """
    An intertemporal option: o_i = (t_i, r_i)

    Represents a reward available at a specific time.

    Attributes:
        label: Option identifier (e.g., "a", "b")
        time: Time value t_i when reward is received
        reward: Reward value r_i
    """

    label: str
    time: TimeValue
    reward: RewardValue


@dataclass
class PreferencePair(SchemaClass):
    """
    A pair of intertemporal options for comparison.

    By convention, short_term has smaller time value than long_term.

    Attributes:
        short_term: The shorter-term option (smaller t_i)
        long_term: The longer-term option (larger t_j)
    """

    short_term: IntertemporalOption
    long_term: IntertemporalOption


@dataclass
class PreferenceQuestion(SchemaClass):
    """
    A preference question: q_x = (pair, t_h)

    Presents two intertemporal options with a target time horizon.

    Attributes:
        pair: The preference pair (short_term, long_term options)
        time_horizon: Target time horizon t_h for decision making (None = no constraint)
    """

    pair: PreferencePair
    time_horizon: Optional[TimeValue] = None  # None = no time horizon constraint


# =============================================================================
# Prompt & Response Types
# =============================================================================


@dataclass
class Prompt(SchemaClass):
    """
    Input prompt: x = (t_h, q_x, s_c)

    Attributes:
        question: Preference question q_x (includes pair and time_horizon)
        context: Context string s_c
        text: Full formatted prompt text
        response_format: Expected response format template
    """

    question: PreferenceQuestion
    context: str
    text: str = ""
    response_format: str = ""


@dataclass
class Response(SchemaClass):
    """
    Output response: y = (o_k, s_t)

    Attributes:
        chosen_option: The selected option o_k (label: "a" or "b")
        trace: Strategy/reasoning trace string s_t
        raw_text: Raw response text from model
    """

    chosen_option: str
    trace: str = ""
    raw_text: str = ""


# =============================================================================
# Dataset Internal Types
# =============================================================================


@dataclass
class DatasetSample(SchemaClass):
    """
    A complete dataset sample (internal representation).

    Attributes:
        id: Unique sample identifier
        prompt: The full prompt object
        response: Model response (populated after inference)
        domain: Domain category (e.g., "housing", "health", "climate")
    """

    id: int
    prompt: Prompt
    response: Optional[Response] = None
    domain: str = ""


@dataclass
class DatasetMetadata(SchemaClass):
    """Internal metadata for a generated dataset."""

    config_name: str
    domain: str
    num_samples: int
    time_horizons: list[TimeValue] = field(default_factory=list)
    seed: int = 42
    description: str = ""


# =============================================================================
# Value Function & Discount Function Types
# =============================================================================


class DiscountType(Enum):
    """Types of discount functions."""

    EXPONENTIAL = "exponential"  # D(t) = exp(-θt)
    HYPERBOLIC = "hyperbolic"  # D(t) = 1 / (1 + θt)
    QUASI_HYPERBOLIC = "quasi_hyperbolic"  # D(t) = β * δ^t
    GENERALIZED_HYPERBOLIC = "generalized_hyperbolic"  # D(t) = 1 / (1 + αt)^(γ/α)
    DOUBLE_EXPONENTIAL = "double_exponential"  # D(t) = ω*exp(-r*t) + (1-ω)*exp(-s*t)
    CONSTANT_SENSITIVITY = "constant_sensitivity"  # D(t) = exp(-(r*t)^s)


@dataclass
class DiscountFunctionParams(SchemaClass):
    """
    Parameters for discount function D(t; θ).

    D(t; θ) maps time to discount factor in [0, 1].

    Attributes:
        discount_type: Type of discount function
        theta: Primary discount rate parameter
        beta: Present bias parameter (for quasi-hyperbolic)
        delta: Per-period discount (for quasi-hyperbolic)
        gamma: Shape parameter (for generalized hyperbolic)
        omega: Weight parameter (for double exponential)
        r_slow: Slow discount rate (for double exponential)
        sensitivity: Time sensitivity (for constant sensitivity)
    """

    discount_type: DiscountType = DiscountType.EXPONENTIAL
    theta: float = 0.1  # Discount rate (also used as alpha in gen. hyperbolic, r_fast in double exp)
    beta: float = 1.0  # Present bias (quasi-hyperbolic)
    delta: float = 0.99  # Per-period discount (quasi-hyperbolic)
    gamma: float = 1.0  # Shape parameter (generalized hyperbolic)
    omega: float = 0.3  # Weight on fast system (double exponential)
    r_slow: float = 0.01  # Slow discount rate (double exponential)
    sensitivity: float = 1.0  # Time sensitivity s (constant sensitivity)


class UtilityType(Enum):
    """Types of utility functions."""

    LINEAR = "linear"  # u(r) = r
    LOG = "log"  # u(r) = log(r)
    POWER = "power"  # u(r) = r^α
    CRRA = "crra"  # u(c) = (c^(1-η) - 1) / (1-η), CRRA utility
    CARA = "cara"  # u(c) = 1 - exp(-α*c), CARA utility


@dataclass
class ValueFunctionParams(SchemaClass):
    """
    Parameters for value function U(o; θ) = u(r) * D(t; θ).

    Attributes:
        utility_type: Type of utility function u(r)
        alpha: Power parameter for power utility
        discount: Discount function parameters
    """

    utility_type: UtilityType = UtilityType.LINEAR
    alpha: float = 1.0  # Power utility exponent
    discount: DiscountFunctionParams = field(default_factory=DiscountFunctionParams)


# =============================================================================
# Training Types
# =============================================================================


@dataclass
class TrainingSample(SchemaClass):
    """
    A training sample with observed choice.

    Attributes:
        question: The preference question presented
        chosen: Label of chosen option ("a" or "b", or "short_term"/"long_term")
    """

    question: PreferenceQuestion
    chosen: str


@dataclass
class TrainingResult(SchemaClass):
    """
    Result from training a value function.

    Attributes:
        params: Fitted parameters
        loss: Final training loss
        num_samples: Number of training samples
        accuracy: Training accuracy
    """

    params: ValueFunctionParams
    loss: float
    num_samples: int
    accuracy: float


# =============================================================================
# Model Internals
# =============================================================================


@dataclass
class ModelInternals(SchemaClass):
    """
    Internal model states captured during inference.

    Attributes:
        hidden_states: Hidden states at key positions (optional)
        attention_weights: Attention weights (optional)
        logits: Output logits for choices (optional)
    """

    hidden_states: Optional[list[list[float]]] = None
    attention_weights: Optional[list[list[float]]] = None
    logits: Optional[dict[str, float]] = None


# =============================================================================
# Legacy Types
# =============================================================================


@dataclass
class GenerationConfig(SchemaClass):
    """Configuration for trajectory generation."""

    model: str
    base_prompt: str
    branching_points: list[str] = field(default_factory=list)
    temperature: float = 1.8
    top_p: float = 0.995
    top_k: int = 500
    estimation_temperature: float = 0.8
    seed: int = 42
