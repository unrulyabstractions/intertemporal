"""
Input config schemas for dataset generation.

This file contains schemas that correspond to input JSON config files:
- configs/dataset/*.json
- configs/formatting/*.json

VERSION: Schema version for config/output compatibility.
"""

from dataclasses import dataclass
from enum import Enum

from .schema_utils import SchemaClass
from ..types import TimeValue


# Schema version - increment when breaking changes are made
SCHEMA_VERSION = "1.0"


# =============================================================================
# Input Config Schemas
# =============================================================================


class StepType(Enum):
    """Types of stepping for grid generation."""

    LINEAR = "linear"
    LOGARITHMIC = "logarithmic"


@dataclass
class OptionRangeConfig(SchemaClass):
    """
    Configuration for an option's reward and time ranges.

    Sample JSON:
    {
        "reward_range": [2000, 5000],
        "time_range": [[3, "months"], [1, "years"]],
        "reward_steps": [1, "linear"],
        "time_steps": [1, "linear"]
    }
    """

    reward_range: tuple[float, float]  # (min, max)
    time_range: tuple[TimeValue, TimeValue]  # (min, max)
    reward_steps: tuple[int, StepType]  # (num_intervals, step_type)
    time_steps: tuple[int, StepType]  # (num_intervals, step_type)


@dataclass
class ContextConfig(SchemaClass):
    """
    Context configuration for dataset.

    Sample JSON:
    {
        "reward_unit": "housing units",
        "role": "the city administration",
        "situation": "Plan for housing in the city.",
        "action_in_question": "build",
        "reasoning_ask": "why choice was made",
        "domain": "housing",
        "labels": ["a)", "b)"],
        "method": "grid",
        "seed": 42
    }
    """

    reward_unit: str
    role: str
    situation: str
    action_in_question: str
    reasoning_ask: str
    domain: str
    labels: tuple[str, str] = ("a)", "b)")  # (left_label, right_label)
    method: str = "grid"  # "grid" or "random"
    seed: int = 42
    extra_situation: str = ""  # Optional extra context for situation


@dataclass
class DatasetConfig(SchemaClass):
    """
    Configuration for dataset generation.

    Loaded from src/configs/dataset/*.json

    Sample JSON:
    {
        "name": "cityhousing",
        "context": { <ContextConfig> },
        "options": {
            "short_term": { <OptionRangeConfig> },
            "long_term": { <OptionRangeConfig> }
        },
        "time_horizons": [null, [5, "months"], [15, "years"]],
        "add_formatting_variations": false
    }

    Note: labels, method, seed are in context.
    time_horizons can include null for samples without time horizon constraint.
    add_formatting_variations: If true, randomizes labels, order, time units, etc.
    """

    name: str
    context: ContextConfig
    options: dict[str, OptionRangeConfig]  # "short_term", "long_term"
    time_horizons: list[TimeValue | None]  # None = no time horizon constraint
    add_formatting_variations: bool = False  # Enable random formatting variations

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        from dataclasses import asdict

        config_dict = asdict(self)

        # Convert TimeValue objects to lists (None stays None)
        config_dict["time_horizons"] = [
            t.to_list() if t is not None else None for t in self.time_horizons
        ]

        # Convert labels tuple to list
        config_dict["context"]["labels"] = list(self.context.labels)

        # Convert option configs
        for key in ("short_term", "long_term"):
            opt = self.options[key]
            config_dict["options"][key]["time_range"] = [
                opt.time_range[0].to_list(),
                opt.time_range[1].to_list(),
            ]
            config_dict["options"][key]["reward_steps"] = [
                opt.reward_steps[0],
                opt.reward_steps[1].value,
            ]
            config_dict["options"][key]["time_steps"] = [
                opt.time_steps[0],
                opt.time_steps[1].value,
            ]

        return config_dict


@dataclass
class DatasetRunSpec(SchemaClass):
    """
    Specification for a dataset generation run.

    Used to compute dataset_run_id as a proper schema hash.
    Shared across all datasets generated in a single --config invocation.
    """
    config_names: tuple[str, ...]  # Sorted list of config names
    formatting_name: str


@dataclass
class FormattingConfig(SchemaClass):
    """
    Configuration for prompt formatting.

    Loaded from src/configs/formatting/*.json

    Sample JSON:
    {
        "question_template": "Situation: [SITUATION]\\nTask: You, [ROLE]...[TIME_HORIZON_SPEC]",
        "response_format": "I choose: [OPTION_LETTER]. The reasoning...",
        "choice_prefix": "I choose:",
        "time_horizon_spec": "You are concerned about outcome in [TIME_HORIZON].",
        "max_reasoning_length": "1-2 sentences"
    }

    Placeholders: [SITUATION], [ROLE], [ACTION_IN_QUESTION], [REWARD_UNITS],
                  [SHORT_TERM_REWARD], [SHORT_TERM_TIME], [LONG_TERM_REWARD],
                  [LONG_TERM_TIME], [TIME_HORIZON], [TIME_HORIZON_SPEC],
                  [REASONING_ASK], [MAX_REASONING_LENGTH],
                  [CHOICE_PREFIX], [REASONING_PREFIX]

    [TIME_HORIZON_SPEC] is replaced with time_horizon_spec template when
    time_horizon is set, or empty string when time_horizon is null.
    """

    question_template: str
    response_format: str
    choice_prefix: str = "I choose:"
    reasoning_prefix: str = "My reasoning:"
    time_horizon_spec: str = ""  # Template for time horizon, empty if not used
    max_reasoning_length: str = "1-2 sentences"


@dataclass
class DecodingConfig(SchemaClass):
    """
    Configuration for text generation decoding.

    Sample JSON:
    {
        "max_new_tokens": 256,
        "temperature": 0.0,
        "top_k": 0,
        "top_p": 1.0
    }
    """

    max_new_tokens: int = 256
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0


class TokenLocation(Enum):
    """Location of token position - prompt or continuation."""
    PROMPT = "prompt"
    CONTINUATION = "continuation"


@dataclass
class TokenPosition(SchemaClass):
    """
    Specification for a token position to capture activations from.

    Supports multiple formats:
    - {"text": "some text", "location": "prompt"} - find text in prompt
    - {"text": "some text", "location": "continuation"} - find text in continuation
    - {"text": "some text"} - find text (defaults to continuation)
    - {"index": N} - index into continuation (shorthand for continuation_index)
    - {"prompt_index": N} - index into prompt tokens (negative = from end)
    - {"continuation_index": N} - index into continuation tokens (negative = from end)
    - {"after_time_horizon_spec": true, ...} - resolve position AFTER time_horizon_spec

    Sample JSON entries:
        {"text": "I select:", "location": "continuation"}
        {"prompt_index": -1}
        {"continuation_index": 0}
        {"text": "ACTION:", "location": "prompt", "after_time_horizon_spec": true}
    """
    # Text-based position (find this text)
    text: str = None
    location: TokenLocation = None  # Where to search for text

    # Index-based positions
    index: int = None  # Shorthand for continuation_index
    prompt_index: int = None
    continuation_index: int = None

    # Position modifiers
    after_time_horizon_spec: bool = False  # Search only after [TIME_HORIZON_SPEC] in prompt

    @classmethod
    def from_dict(cls, data: dict) -> "TokenPosition":
        """Parse TokenPosition from JSON dict."""
        if isinstance(data, int):
            # Legacy: plain int means continuation index
            return cls(continuation_index=data)

        if isinstance(data, str):
            # Legacy: plain str means find text in continuation
            return cls(text=data, location=TokenLocation.CONTINUATION)

        location = None
        if "location" in data:
            location = TokenLocation(data["location"])

        return cls(
            text=data.get("text"),
            location=location,
            index=data.get("index"),
            prompt_index=data.get("prompt_index"),
            continuation_index=data.get("continuation_index"),
            after_time_horizon_spec=data.get("after_time_horizon_spec", False),
        )

    def is_prompt_position(self) -> bool:
        """Whether this position refers to prompt tokens."""
        if self.prompt_index is not None:
            return True
        if self.location == TokenLocation.PROMPT:
            return True
        return False

    def is_text_search(self) -> bool:
        """Whether this position uses text search."""
        return self.text is not None

    def get_index(self) -> int | None:
        """Get the index value (if index-based position)."""
        if self.prompt_index is not None:
            return self.prompt_index
        if self.continuation_index is not None:
            return self.continuation_index
        if self.index is not None:
            return self.index
        return None


@dataclass
class InternalsConfig(SchemaClass):
    """
    Configuration for which internals to capture.

    Sample JSON:
    {
        "resid_post": {"layers": [0, -1]}
    }
    """

    activations: dict = None
    token_positions: list[TokenPosition] = None

    def __post_init__(self):
        if self.activations is None:
            self.activations = {}
        if self.token_positions is None:
            self.token_positions = []
        super().__post_init__()


@dataclass
class DatasetSpec(SchemaClass):
    """Specification for a dataset to query."""
    name: str  # Expected dataset name (for validation)
    dataset_id: str  # Dataset ID (primary lookup key)


@dataclass
class SingleQuerySpec(SchemaClass):
    """
    Specification for a single query (one dataset + model + formatting + query params).

    Used to compute query_id as a proper schema hash.
    Includes all parameters that affect query results.
    """
    dataset_id: str
    model: str
    formatting_id: str
    decoding: DecodingConfig
    internals: InternalsConfig
    subsample: float


@dataclass
class QueryConfig(SchemaClass):
    """
    Configuration for LLM querying.

    Loaded from configs/query/*.json

    Sample JSON:
    {
        "models": ["gpt2", "meta-llama/Meta-Llama-3-8B-Instruct"],
        "datasets": [
            {"name": "cityhousing", "dataset_id": "b841a6a419433f1ef8ab0c6d3aa77b43"}
        ],
        "formatting": {
            "name": "default_formatting"
        },
        "decoding": {
            "max_new_tokens": 256,
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0
        },
        "internals": {
            "resid_post": {"layers": [0, -1]}
        },
        "token_positions": [0, "I choose:"],
        "device": null,
        "limit": 0,
        "subsample": 1.0
    }
    """

    models: list
    datasets: list[DatasetSpec]
    formatting_name: str
    formatting_id: str
    decoding: DecodingConfig
    internals: InternalsConfig
    device: str = None
    limit: int = 0
    subsample: float = 1.0  # Fraction of samples to use (0.1 = 10%, 1.0 = all)
