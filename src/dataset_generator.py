"""
Dataset generator class for intertemporal preference experiments.

Generates datasets from config files with support for:
- Grid and random sampling methods
- Linear and logarithmic stepping
- Separate dataset and formatting configs
- Optional formatting variations (labels, time units, number spelling)
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from common.formatting_variation import FormattingVariation

from .common.profiling import get_profiler
from .common.schemas import (
    ContextConfig,
    DatasetConfig,
    FormattingConfig,
    OptionRangeConfig,
    StepType,
)
from .types import (
    DatasetMetadata,
    DatasetSample,
    IntertemporalOption,
    PreferencePair,
    PreferenceQuestion,
    Prompt,
    RewardValue,
    TimeValue,
)


class DatasetGenerator:
    """
    Generator for intertemporal preference datasets.

    Reads config and generates samples with varying time horizons and options.
    Supports grid and random sampling with linear/logarithmic stepping.
    """

    def __init__(
        self,
        dataset_config: DatasetConfig,
        formatting_config: FormattingConfig,
        seed: int = 42,
    ):
        """
        Initialize generator with configs.

        Args:
            dataset_config: Dataset configuration
            formatting_config: Prompt formatting configuration
            seed: Random seed for reproducibility
        """
        self.dataset_config = dataset_config
        self.formatting_config = formatting_config
        self.seed = seed
        random.seed(seed)

    @classmethod
    def from_config_files(
        cls,
        dataset_config_path: Path,
        formatting_config_path: Path,
    ) -> "DatasetGenerator":
        """
        Create generator from config files.

        Args:
            dataset_config_path: Path to dataset config JSON
            formatting_config_path: Path to formatting config JSON

        Returns:
            Configured DatasetGenerator instance
        """
        profiler = get_profiler()
        with profiler.measure("load_dataset_config", {"path": str(dataset_config_path)}):
            dataset_config = cls.load_dataset_config(dataset_config_path)
        with profiler.measure("load_formatting_config", {"path": str(formatting_config_path)}):
            formatting_config = cls.load_formatting_config(formatting_config_path)
        return cls(dataset_config, formatting_config, dataset_config.context.seed)

    @staticmethod
    def parse_time_value(time_data) -> TimeValue:
        """
        Parse time value from various formats.

        Handles:
        - [value, unit] arrays: [5, "months"]
        - "value unit" strings: "5 months"
        - Dict with value/unit keys

        Robust to singular/plural: "1 year" == "1 years"
        """
        if isinstance(time_data, list) and len(time_data) == 2:
            value = float(time_data[0])
            unit = time_data[1]
        elif isinstance(time_data, str):
            parts = time_data.lower().strip().split()
            if len(parts) != 2:
                raise ValueError(f"Invalid time format: {time_data}")
            value = float(parts[0])
            unit = parts[1]
        elif isinstance(time_data, dict):
            value = float(time_data["value"])
            unit = time_data["unit"]
        else:
            raise ValueError(f"Unknown time format: {time_data}")

        # Normalize unit (robust to singular/plural)
        unit_lower = unit.lower()
        if unit_lower in ("month", "months"):
            unit = "months"
        elif unit_lower in ("year", "years"):
            unit = "years"
        elif unit_lower in ("day", "days"):
            unit = "days"
        elif unit_lower in ("week", "weeks"):
            unit = "weeks"
        else:
            unit = unit_lower

        return TimeValue(value=value, unit=unit)

    @classmethod
    def load_dataset_config(cls, path: Path) -> DatasetConfig:
        """
        Load and parse dataset config from JSON file.

        Sample JSON (see schemas: DatasetConfig, ContextConfig, OptionRangeConfig):
        {
            "name": "cityhousing",
            "context": {
                "reward_unit": "housing units",
                "role": "the city administration",
                "situation": "Plan for housing in the city.",
                "action_in_question": "build",
                "reasoning_ask": "why choice was made",
                "domain": "housing",
                "labels": ["a)", "b)"],
                "method": "grid",
                "seed": 42
            },
            "options": {
                "short_term": {
                    "reward_range": [2000, 5000],
                    "time_range": [[3, "months"], [1, "years"]],
                    "reward_steps": [1, "linear"],
                    "time_steps": [1, "linear"]
                },
                "long_term": { ... }  // same structure as short_term
            },
            "time_horizons": [[5, "months"], [15, "years"]]
        }
        """
        with open(path) as f:
            data = json.load(f)

        # Parse context (includes labels, method, seed)
        ctx = data["context"]
        labels = ctx.get("labels", ["a)", "b)"])
        if isinstance(labels, list):
            labels = tuple(labels)

        context = ContextConfig(
            reward_unit=ctx["reward_unit"],
            role=ctx["role"],
            situation=ctx["situation"],
            action_in_question=ctx["action_in_question"],
            reasoning_ask=ctx["reasoning_ask"],
            domain=ctx["domain"],
            labels=labels,
            method=ctx.get("method", "grid"),
            seed=ctx.get("seed", 42),
        )

        # Parse options
        options = {}
        for key in ("short_term", "long_term"):
            opt = data["options"][key]
            options[key] = OptionRangeConfig(
                reward_range=tuple(opt["reward_range"]),
                time_range=(
                    cls.parse_time_value(opt["time_range"][0]),
                    cls.parse_time_value(opt["time_range"][1]),
                ),
                reward_steps=(
                    opt["reward_steps"][0],
                    StepType(opt["reward_steps"][1]),
                ),
                time_steps=(
                    opt["time_steps"][0],
                    StepType(opt["time_steps"][1]),
                ),
            )

        # Parse time horizons (null/None = no time horizon constraint)
        time_horizons = [
            cls.parse_time_value(t) if t is not None else None
            for t in data["time_horizons"]
        ]

        # Parse optional formatting variations flag
        add_formatting_variations = data.get("add_formatting_variations", False)

        return DatasetConfig(
            name=data["name"],
            context=context,
            options=options,
            time_horizons=time_horizons,
            add_formatting_variations=add_formatting_variations,
        )

    @staticmethod
    def load_formatting_config(path: Path) -> FormattingConfig:
        """
        Load formatting config from JSON file.

        Sample JSON (see schema: FormattingConfig):
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
                      [REASONING_ASK], [CHOICE_PREFIX], [MAX_REASONING_LENGTH]
        """
        with open(path) as f:
            data = json.load(f)

        return FormattingConfig(
            question_template=data["question_template"],
            response_format=data["response_format"],
            choice_prefix=data.get("choice_prefix", "I choose:"),
            time_horizon_spec=data.get("time_horizon_spec", ""),
            max_reasoning_length=data.get("max_reasoning_length", "1-2 sentences"),
        )

    def generate_steps(
        self,
        min_val: float,
        max_val: float,
        num_intervals: int,
        step_type: StepType,
    ) -> list[float]:
        """
        Generate stepped values between min and max.

        Args:
            min_val: Minimum value
            max_val: Maximum value
            num_intervals: Number of intervals (0 = midpoint only, 1 = endpoints, 2 = 3 values, etc.)
            step_type: LINEAR or LOGARITHMIC

        Returns:
            List of values (num_intervals + 1 values, or 1 value if num_intervals=0)
        """
        if num_intervals == 0:
            # Return midpoint
            if step_type == StepType.LINEAR:
                return [(min_val + max_val) / 2]
            else:  # LOGARITHMIC
                if min_val <= 0:
                    raise ValueError("Logarithmic stepping requires positive values")
                return [math.exp((math.log(min_val) + math.log(max_val)) / 2)]

        num_values = num_intervals + 1

        if step_type == StepType.LINEAR:
            step = (max_val - min_val) / num_intervals
            return [min_val + i * step for i in range(num_values)]
        else:  # LOGARITHMIC
            if min_val <= 0:
                raise ValueError("Logarithmic stepping requires positive values")
            log_min = math.log(min_val)
            log_max = math.log(max_val)
            log_step = (log_max - log_min) / num_intervals
            return [math.exp(log_min + i * log_step) for i in range(num_values)]

    def generate_time_steps(
        self,
        min_time: TimeValue,
        max_time: TimeValue,
        num_intervals: int,
        step_type: StepType,
    ) -> list[TimeValue]:
        """
        Generate stepped time values.

        Args:
            min_time: Minimum time
            max_time: Maximum time
            num_intervals: Number of intervals (0 = midpoint only)
            step_type: LINEAR or LOGARITHMIC

        Returns:
            List of TimeValue objects
        """
        # Convert to common unit (months) for stepping
        min_months = min_time.to_months()
        max_months = max_time.to_months()

        month_values = self.generate_steps(min_months, max_months, num_intervals, step_type)

        # Convert back, using appropriate unit based on magnitude
        result = []
        for months in month_values:
            if months >= 12:
                # Use years for 12+ months
                years = months / 12
                result.append(TimeValue(value=round(years, 1), unit="years"))
            else:
                result.append(TimeValue(value=round(months, 1), unit="months"))

        return result

    def generate_option_grid(
        self, option_key: str
    ) -> list[tuple[float, TimeValue]]:
        """
        Generate grid of (reward, time) combinations for an option.

        Args:
            option_key: "short_term" or "long_term"

        Returns:
            List of (reward_value, time_value) tuples
        """
        opt = self.dataset_config.options[option_key]

        # Generate reward steps
        rewards = self.generate_steps(
            opt.reward_range[0],
            opt.reward_range[1],
            opt.reward_steps[0],
            opt.reward_steps[1],
        )

        # Generate time steps
        times = self.generate_time_steps(
            opt.time_range[0],
            opt.time_range[1],
            opt.time_steps[0],
            opt.time_steps[1],
        )

        # Create grid
        grid = []
        for reward in rewards:
            for time in times:
                grid.append((reward, time))

        return grid

    def format_prompt(
        self,
        left_option: IntertemporalOption,
        right_option: IntertemporalOption,
        time_horizon: Optional[TimeValue],
        labels: tuple[str, str],
        left_time_str: Optional[str] = None,
        right_time_str: Optional[str] = None,
        horizon_time_str: Optional[str] = None,
    ) -> str:
        """
        Format prompt text using template and context.

        Args:
            left_option: Option displayed on left (first)
            right_option: Option displayed on right (second)
            time_horizon: Decision time horizon (None = no constraint)
            labels: (left_label, right_label) tuple
            left_time_str: Optional formatted time string for left option
            right_time_str: Optional formatted time string for right option
            horizon_time_str: Optional formatted time string for horizon

        Returns:
            Formatted prompt text
        """
        ctx = self.dataset_config.context
        template = self.formatting_config.question_template

        # Use provided time strings or default to str(time)
        left_time = left_time_str if left_time_str else str(left_option.time)
        right_time = right_time_str if right_time_str else str(right_option.time)

        # Replace placeholders
        prompt = template.replace("[SITUATION]", ctx.situation)
        prompt = prompt.replace("[EXTRA_SITUATION]", ctx.extra_situation)
        prompt = prompt.replace("[ROLE]", ctx.role)
        prompt = prompt.replace("[ACTION_IN_QUESTION]", ctx.action_in_question)
        prompt = prompt.replace("[REWARD_UNITS]", ctx.reward_unit)
        prompt = prompt.replace("[REASONING_ASK]", ctx.reasoning_ask)
        prompt = prompt.replace("[MAX_REASONING_LENGTH]", self.formatting_config.max_reasoning_length)

        # Handle time horizon spec (conditional inclusion)
        if time_horizon is not None:
            horizon_str = horizon_time_str if horizon_time_str else str(time_horizon)
            time_horizon_spec = self.formatting_config.time_horizon_spec.replace(
                "[TIME_HORIZON]", horizon_str
            )
            prompt = prompt.replace("[TIME_HORIZON_SPEC]", time_horizon_spec)
            # Also support direct [TIME_HORIZON] placeholder for backwards compatibility
            prompt = prompt.replace("[TIME_HORIZON]", horizon_str)
        else:
            # No time horizon - remove the spec placeholder entirely
            prompt = prompt.replace("[TIME_HORIZON_SPEC]", "")
            prompt = prompt.replace("[TIME_HORIZON]", "")

        # Left/right option placeholders (new format)
        prompt = prompt.replace("[LEFT_TERM_LABEL]", labels[0])
        prompt = prompt.replace("[LEFT_TERM_REWARD]", f"{round(left_option.reward.value):,}")
        prompt = prompt.replace("[LEFT_TERM_TIME]", left_time)
        prompt = prompt.replace("[RIGHT_TERM_LABEL]", labels[1])
        prompt = prompt.replace("[RIGHT_TERM_REWARD]", f"{round(right_option.reward.value):,}")
        prompt = prompt.replace("[RIGHT_TERM_TIME]", right_time)

        # Legacy short/long term placeholders (for backwards compatibility)
        # These are deprecated - use LEFT/RIGHT instead
        prompt = prompt.replace("[SHORT_TERM_REWARD]", f"{round(left_option.reward.value):,}")
        prompt = prompt.replace("[SHORT_TERM_TIME]", left_time)
        prompt = prompt.replace("[LONG_TERM_REWARD]", f"{round(right_option.reward.value):,}")
        prompt = prompt.replace("[LONG_TERM_TIME]", right_time)

        # Validate no unreplaced placeholders remain
        self._validate_no_unreplaced_placeholders(prompt, "question_template")

        return prompt

    def _validate_no_unreplaced_placeholders(self, text: str, context: str = "") -> None:
        """
        Validate that no [PLACEHOLDER] patterns remain in text.

        Args:
            text: Text to check
            context: Description of where this text came from (for error messages)

        Raises:
            ValueError: If unreplaced placeholders are found
        """
        import re
        # Find [WORD] patterns that look like placeholders:
        # - Must contain underscore OR be longer than 2 chars
        # - This excludes labels like [A], [B], [1], [2] which are intentional
        all_brackets = re.findall(r'\[[A-Z][A-Z0-9_]*\]', text)
        placeholders = [p for p in all_brackets if '_' in p or len(p) > 4]  # [XX] = 4 chars
        if placeholders:
            unique = sorted(set(placeholders))
            ctx = f" in {context}" if context else ""
            raise ValueError(
                f"Unreplaced placeholders found{ctx}: {', '.join(unique)}\n"
                f"Text snippet: {text[:200]}..."
            )

    def _get_formatting_variation(self) -> "FormattingVariation":
        """Get a formatting variation (random if enabled, default otherwise)."""
        # Lazy import to avoid circular dependencies
        import sys
        scripts_dir = Path(__file__).parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from common.formatting_variation import FormattingVariation

        if self.dataset_config.add_formatting_variations:
            return FormattingVariation.random(allow_all=True)
        return FormattingVariation.default()

    def _apply_time_variation(
        self, tv: TimeValue, variation: "FormattingVariation"
    ) -> tuple[TimeValue, str]:
        """Apply time variation to a TimeValue."""
        import sys
        scripts_dir = Path(__file__).parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from common.formatting_variation import apply_time_variation
        return apply_time_variation(tv, variation)

    def create_sample(
        self,
        sample_id: int,
        short_term_data: tuple[float, TimeValue],
        long_term_data: tuple[float, TimeValue],
        time_horizon: Optional[TimeValue],
    ) -> DatasetSample:
        """
        Create a dataset sample from option data.

        Randomly assigns short_term to left or right position.
        Applies formatting variations if enabled in config.

        Args:
            sample_id: Unique sample ID
            short_term_data: (reward, time) for short-term option
            long_term_data: (reward, time) for long-term option
            time_horizon: Decision time horizon (None = no constraint)

        Returns:
            DatasetSample instance
        """
        ctx = self.dataset_config.context

        # Get formatting variation (random if enabled, default otherwise)
        variation = self._get_formatting_variation()

        # Use variation labels if enabled, otherwise use config labels
        if self.dataset_config.add_formatting_variations:
            labels = variation.labels
        else:
            labels = ctx.labels

        # Randomly assign short_term to left (index 0) or right (index 1)
        # The variation.flip_order adds another layer of randomization
        short_on_left = random.choice([True, False])
        if variation.flip_order:
            short_on_left = not short_on_left

        if short_on_left:
            left_label, right_label = labels[0], labels[1]
            short_term_label, long_term_label = left_label, right_label
        else:
            left_label, right_label = labels[0], labels[1]
            short_term_label, long_term_label = right_label, left_label

        short_term = IntertemporalOption(
            label=short_term_label,
            time=short_term_data[1],
            reward=RewardValue(value=round(short_term_data[0]), unit=ctx.reward_unit),
        )

        long_term = IntertemporalOption(
            label=long_term_label,
            time=long_term_data[1],
            reward=RewardValue(value=round(long_term_data[0]), unit=ctx.reward_unit),
        )

        pair = PreferencePair(short_term=short_term, long_term=long_term)
        question = PreferenceQuestion(pair=pair, time_horizon=time_horizon)

        # Determine which option goes on left/right for formatting
        if short_on_left:
            left_option, right_option = short_term, long_term
        else:
            left_option, right_option = long_term, short_term

        # Apply time variations for prompt formatting
        _, left_time_str = self._apply_time_variation(left_option.time, variation)
        _, right_time_str = self._apply_time_variation(right_option.time, variation)

        # Apply time variation to horizon if present
        horizon_time_str = None
        if time_horizon is not None:
            _, horizon_time_str = self._apply_time_variation(time_horizon, variation)

        prompt_text = self.format_prompt(
            left_option, right_option, time_horizon, labels,
            left_time_str=left_time_str,
            right_time_str=right_time_str,
            horizon_time_str=horizon_time_str,
        )

        # Format response_format with the actual labels
        response_format = self.formatting_config.response_format
        response_format = response_format.replace("[LEFT_TERM_LABEL]", labels[0])
        response_format = response_format.replace("[RIGHT_TERM_LABEL]", labels[1])
        response_format = response_format.replace("[CHOICE_PREFIX]", self.formatting_config.choice_prefix)
        response_format = response_format.replace("[MAX_REASONING_LENGTH]", self.formatting_config.max_reasoning_length)

        prompt = Prompt(
            question=question,
            context=ctx.situation,
            text=prompt_text,
            response_format=response_format,
        )

        return DatasetSample(
            id=sample_id,
            prompt=prompt,
            response=None,
            domain=ctx.domain,
        )

    def generate_grid(self) -> list[DatasetSample]:
        """
        Generate samples using grid method.

        Creates all combinations of:
        - Short-term option grid
        - Long-term option grid
        - Time horizons

        Returns:
            List of DatasetSample objects
        """
        profiler = get_profiler()

        with profiler.measure("generate_option_grid_short"):
            short_term_grid = self.generate_option_grid("short_term")
        with profiler.measure("generate_option_grid_long"):
            long_term_grid = self.generate_option_grid("long_term")

        time_horizons = self.dataset_config.time_horizons

        samples = []
        sample_id = 0

        with profiler.measure("create_samples", {
            "num_combinations": len(short_term_grid) * len(long_term_grid) * len(time_horizons)
        }):
            for time_horizon in time_horizons:
                for short_data in short_term_grid:
                    for long_data in long_term_grid:
                        sample = self.create_sample(
                            sample_id, short_data, long_data, time_horizon
                        )
                        samples.append(sample)
                        sample_id += 1

        return samples

    def generate_random(self, num_samples: int = 100) -> list[DatasetSample]:
        """
        Generate samples using random sampling.

        Args:
            num_samples: Number of samples to generate

        Returns:
            List of DatasetSample objects
        """
        ctx = self.dataset_config.context
        time_horizons = self.dataset_config.time_horizons

        samples = []

        for sample_id in range(num_samples):
            # Random time horizon
            time_horizon = random.choice(time_horizons)

            # Random short-term option
            st_opt = self.dataset_config.options["short_term"]
            st_reward = random.uniform(*st_opt.reward_range)
            st_time_months = random.uniform(
                st_opt.time_range[0].to_months(),
                st_opt.time_range[1].to_months(),
            )
            if st_time_months >= 12:
                st_time = TimeValue(value=round(st_time_months / 12, 1), unit="years")
            else:
                st_time = TimeValue(value=round(st_time_months, 1), unit="months")

            # Random long-term option
            lt_opt = self.dataset_config.options["long_term"]
            lt_reward = random.uniform(*lt_opt.reward_range)
            lt_time_months = random.uniform(
                lt_opt.time_range[0].to_months(),
                lt_opt.time_range[1].to_months(),
            )
            if lt_time_months >= 12:
                lt_time = TimeValue(value=round(lt_time_months / 12, 1), unit="years")
            else:
                lt_time = TimeValue(value=round(lt_time_months, 1), unit="months")

            sample = self.create_sample(
                sample_id,
                (st_reward, st_time),
                (lt_reward, lt_time),
                time_horizon,
            )
            samples.append(sample)

        return samples

    def generate(
        self, num_random_samples: Optional[int] = None
    ) -> tuple[list[DatasetSample], DatasetMetadata]:
        """
        Generate dataset samples and metadata.

        Args:
            num_random_samples: Number of samples for random method (ignored for grid)

        Returns:
            Tuple of (samples, metadata)
        """
        profiler = get_profiler()

        with profiler.measure("generate_dataset", {"method": self.dataset_config.context.method}):
            if self.dataset_config.context.method == "grid":
                samples = self.generate_grid()
            else:  # random
                samples = self.generate_random(num_random_samples or 100)

        metadata = DatasetMetadata(
            config_name=self.dataset_config.name,
            domain=self.dataset_config.context.domain,
            num_samples=len(samples),
            time_horizons=self.dataset_config.time_horizons,
            seed=self.seed,
            description=f"Generated from {self.dataset_config.name} config",
        )

        return samples, metadata
