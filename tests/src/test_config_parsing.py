"""Comprehensive tests for config parsing - covering all keys, edge cases, and validation."""

import json
import pytest
from pathlib import Path

from src.dataset_generator import DatasetGenerator
from src.common.schemas import (
    ContextConfig,
    DatasetConfig,
    DecodingConfig,
    FormattingConfig,
    InternalsConfig,
    OptionRangeConfig,
    SingleQuerySpec,
    StepType,
)
from src.types import TimeValue


class TestDatasetConfigKeys:
    """Tests for all dataset config keys."""

    def test_name_required(self, tmp_path):
        """name key is required."""
        config = {
            "context": {
                "reward_unit": "units",
                "role": "tester",
                "situation": "test",
                "action_in_question": "choose",
                "reasoning_ask": "why",
                "domain": "test",
            },
            "options": {
                "short_term": {
                    "reward_range": [100, 200],
                    "time_range": [[1, "months"], [3, "months"]],
                    "reward_steps": [1, "linear"],
                    "time_steps": [1, "linear"],
                },
                "long_term": {
                    "reward_range": [500, 1000],
                    "time_range": [[1, "years"], [2, "years"]],
                    "reward_steps": [1, "linear"],
                    "time_steps": [1, "linear"],
                },
            },
            "time_horizons": [None],
        }
        config_path = tmp_path / "missing_name.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        with pytest.raises(KeyError):
            DatasetGenerator.load_dataset_config(config_path)

    def test_name_variations(self, tmp_path):
        """name can be any string."""
        for name in ["simple", "with-dashes", "with_underscores", "CamelCase", "123numeric"]:
            config = self._minimal_config(name=name)
            config_path = tmp_path / f"name_{name}.json"
            with open(config_path, "w") as f:
                json.dump(config, f)

            result = DatasetGenerator.load_dataset_config(config_path)
            assert result.name == name

    def _minimal_config(self, **overrides):
        """Create minimal valid config with optional overrides."""
        config = {
            "name": overrides.get("name", "test"),
            "context": {
                "reward_unit": overrides.get("reward_unit", "units"),
                "role": overrides.get("role", "tester"),
                "situation": overrides.get("situation", "test situation"),
                "action_in_question": overrides.get("action_in_question", "choose"),
                "reasoning_ask": overrides.get("reasoning_ask", "why"),
                "domain": overrides.get("domain", "test"),
                "labels": overrides.get("labels", ["a)", "b)"]),
                "method": overrides.get("method", "grid"),
                "seed": overrides.get("seed", 42),
            },
            "options": {
                "short_term": {
                    "reward_range": overrides.get("short_reward_range", [100, 200]),
                    "time_range": overrides.get("short_time_range", [[1, "months"], [3, "months"]]),
                    "reward_steps": overrides.get("short_reward_steps", [1, "linear"]),
                    "time_steps": overrides.get("short_time_steps", [1, "linear"]),
                },
                "long_term": {
                    "reward_range": overrides.get("long_reward_range", [500, 1000]),
                    "time_range": overrides.get("long_time_range", [[1, "years"], [2, "years"]]),
                    "reward_steps": overrides.get("long_reward_steps", [1, "linear"]),
                    "time_steps": overrides.get("long_time_steps", [1, "linear"]),
                },
            },
            "time_horizons": overrides.get("time_horizons", [None]),
        }
        if "add_formatting_variations" in overrides:
            config["add_formatting_variations"] = overrides["add_formatting_variations"]
        return config


class TestContextConfigKeys:
    """Tests for context config keys."""

    def _make_config(self, tmp_path, context_overrides=None, **kwargs):
        """Create config with context overrides."""
        config = {
            "name": "test",
            "context": {
                "reward_unit": "dollars",
                "role": "a decision maker",
                "situation": "Test scenario",
                "action_in_question": "select",
                "reasoning_ask": "your reasoning",
                "domain": "finance",
                "labels": ["a)", "b)"],
                "method": "grid",
                "seed": 42,
            },
            "options": {
                "short_term": {
                    "reward_range": [100, 200],
                    "time_range": [[1, "months"], [3, "months"]],
                    "reward_steps": [0, "linear"],
                    "time_steps": [0, "linear"],
                },
                "long_term": {
                    "reward_range": [500, 1000],
                    "time_range": [[1, "years"], [2, "years"]],
                    "reward_steps": [0, "linear"],
                    "time_steps": [0, "linear"],
                },
            },
            "time_horizons": [None],
        }
        if context_overrides:
            config["context"].update(context_overrides)
        config_path = tmp_path / "test.json"
        with open(config_path, "w") as f:
            json.dump(config, f)
        return config_path

    def test_reward_unit_required(self, tmp_path):
        """reward_unit is required."""
        config_path = self._make_config(tmp_path)
        with open(config_path) as f:
            config = json.load(f)
        del config["context"]["reward_unit"]
        with open(config_path, "w") as f:
            json.dump(config, f)

        with pytest.raises(KeyError):
            DatasetGenerator.load_dataset_config(config_path)

    def test_reward_unit_variations(self, tmp_path):
        """reward_unit can be any descriptive string."""
        units = ["dollars", "housing units", "points", "tokens", "items"]
        for unit in units:
            config_path = self._make_config(tmp_path, {"reward_unit": unit})
            result = DatasetGenerator.load_dataset_config(config_path)
            assert result.context.reward_unit == unit

    def test_labels_default(self, tmp_path):
        """labels defaults to ["a)", "b)"]."""
        config_path = self._make_config(tmp_path)
        with open(config_path) as f:
            config = json.load(f)
        del config["context"]["labels"]
        with open(config_path, "w") as f:
            json.dump(config, f)

        result = DatasetGenerator.load_dataset_config(config_path)
        assert result.context.labels == ("a)", "b)")

    def test_labels_custom(self, tmp_path):
        """labels can be customized."""
        custom_labels = [
            ["1)", "2)"],
            ["x)", "y)"],
            ["[A]", "[B]"],
            ["Option 1", "Option 2"],
        ]
        for labels in custom_labels:
            config_path = self._make_config(tmp_path, {"labels": labels})
            result = DatasetGenerator.load_dataset_config(config_path)
            assert result.context.labels == tuple(labels)

    def test_method_default(self, tmp_path):
        """method defaults to 'grid'."""
        config_path = self._make_config(tmp_path)
        with open(config_path) as f:
            config = json.load(f)
        del config["context"]["method"]
        with open(config_path, "w") as f:
            json.dump(config, f)

        result = DatasetGenerator.load_dataset_config(config_path)
        assert result.context.method == "grid"

    def test_method_grid(self, tmp_path):
        """method 'grid' generates all combinations."""
        config_path = self._make_config(tmp_path, {"method": "grid"})
        result = DatasetGenerator.load_dataset_config(config_path)
        assert result.context.method == "grid"

    def test_method_random(self, tmp_path):
        """method 'random' generates random samples."""
        config_path = self._make_config(tmp_path, {"method": "random"})
        result = DatasetGenerator.load_dataset_config(config_path)
        assert result.context.method == "random"

    def test_seed_default(self, tmp_path):
        """seed defaults to 42."""
        config_path = self._make_config(tmp_path)
        with open(config_path) as f:
            config = json.load(f)
        del config["context"]["seed"]
        with open(config_path, "w") as f:
            json.dump(config, f)

        result = DatasetGenerator.load_dataset_config(config_path)
        assert result.context.seed == 42

    def test_seed_custom(self, tmp_path):
        """seed can be customized."""
        for seed in [0, 1, 123, 99999]:
            config_path = self._make_config(tmp_path, {"seed": seed})
            result = DatasetGenerator.load_dataset_config(config_path)
            assert result.context.seed == seed


class TestOptionRangeConfigKeys:
    """Tests for option range config keys."""

    def _make_config(self, tmp_path, short_overrides=None, long_overrides=None):
        """Create config with option overrides."""
        config = {
            "name": "test",
            "context": {
                "reward_unit": "units",
                "role": "tester",
                "situation": "test",
                "action_in_question": "choose",
                "reasoning_ask": "why",
                "domain": "test",
            },
            "options": {
                "short_term": {
                    "reward_range": [100, 200],
                    "time_range": [[1, "months"], [3, "months"]],
                    "reward_steps": [1, "linear"],
                    "time_steps": [1, "linear"],
                },
                "long_term": {
                    "reward_range": [500, 1000],
                    "time_range": [[1, "years"], [2, "years"]],
                    "reward_steps": [1, "linear"],
                    "time_steps": [1, "linear"],
                },
            },
            "time_horizons": [None],
        }
        if short_overrides:
            config["options"]["short_term"].update(short_overrides)
        if long_overrides:
            config["options"]["long_term"].update(long_overrides)
        config_path = tmp_path / "test.json"
        with open(config_path, "w") as f:
            json.dump(config, f)
        return config_path

    def test_reward_range_required(self, tmp_path):
        """reward_range is required for both options."""
        config_path = self._make_config(tmp_path)
        with open(config_path) as f:
            config = json.load(f)
        del config["options"]["short_term"]["reward_range"]
        with open(config_path, "w") as f:
            json.dump(config, f)

        with pytest.raises(KeyError):
            DatasetGenerator.load_dataset_config(config_path)

    def test_reward_range_values(self, tmp_path):
        """reward_range accepts various numeric ranges."""
        ranges = [
            [0, 100],
            [100, 100],  # equal min/max
            [0.5, 1.5],  # floats
            [1, 1000000],  # large range
        ]
        for rng in ranges:
            config_path = self._make_config(tmp_path, short_overrides={"reward_range": rng})
            result = DatasetGenerator.load_dataset_config(config_path)
            assert result.options["short_term"].reward_range == tuple(rng)

    def test_time_range_formats(self, tmp_path):
        """time_range accepts [value, unit] format."""
        ranges = [
            [[1, "days"], [7, "days"]],
            [[1, "weeks"], [4, "weeks"]],
            [[1, "months"], [12, "months"]],
            [[1, "years"], [10, "years"]],
        ]
        for rng in ranges:
            config_path = self._make_config(tmp_path, short_overrides={"time_range": rng})
            result = DatasetGenerator.load_dataset_config(config_path)
            assert result.options["short_term"].time_range[0].value == rng[0][0]
            assert result.options["short_term"].time_range[1].value == rng[1][0]

    def test_time_range_singular_plural(self, tmp_path):
        """time_range handles singular/plural units."""
        # Singular forms
        config_path = self._make_config(tmp_path, short_overrides={
            "time_range": [[1, "month"], [1, "year"]]
        })
        result = DatasetGenerator.load_dataset_config(config_path)
        # Should normalize to plural
        assert result.options["short_term"].time_range[0].unit == "months"
        assert result.options["short_term"].time_range[1].unit == "years"

    def test_step_type_linear(self, tmp_path):
        """step_type 'linear' works for rewards and times."""
        config_path = self._make_config(tmp_path, short_overrides={
            "reward_steps": [3, "linear"],
            "time_steps": [3, "linear"],
        })
        result = DatasetGenerator.load_dataset_config(config_path)
        assert result.options["short_term"].reward_steps == (3, StepType.LINEAR)
        assert result.options["short_term"].time_steps == (3, StepType.LINEAR)

    def test_step_type_logarithmic(self, tmp_path):
        """step_type 'logarithmic' works for rewards and times."""
        config_path = self._make_config(tmp_path, long_overrides={
            "reward_steps": [3, "logarithmic"],
            "time_steps": [3, "logarithmic"],
        })
        result = DatasetGenerator.load_dataset_config(config_path)
        assert result.options["long_term"].reward_steps == (3, StepType.LOGARITHMIC)
        assert result.options["long_term"].time_steps == (3, StepType.LOGARITHMIC)

    def test_steps_zero_gives_midpoint(self, tmp_path):
        """0 intervals gives midpoint only."""
        config_path = self._make_config(tmp_path, short_overrides={
            "reward_steps": [0, "linear"],
            "time_steps": [0, "linear"],
        })
        result = DatasetGenerator.load_dataset_config(config_path)
        assert result.options["short_term"].reward_steps[0] == 0


class TestTimeHorizonKeys:
    """Tests for time_horizons config key."""

    def _make_config(self, tmp_path, time_horizons):
        """Create config with specific time horizons."""
        config = {
            "name": "test",
            "context": {
                "reward_unit": "units",
                "role": "tester",
                "situation": "test",
                "action_in_question": "choose",
                "reasoning_ask": "why",
                "domain": "test",
            },
            "options": {
                "short_term": {
                    "reward_range": [100, 200],
                    "time_range": [[1, "months"], [3, "months"]],
                    "reward_steps": [0, "linear"],
                    "time_steps": [0, "linear"],
                },
                "long_term": {
                    "reward_range": [500, 1000],
                    "time_range": [[1, "years"], [2, "years"]],
                    "reward_steps": [0, "linear"],
                    "time_steps": [0, "linear"],
                },
            },
            "time_horizons": time_horizons,
        }
        config_path = tmp_path / "test.json"
        with open(config_path, "w") as f:
            json.dump(config, f)
        return config_path

    def test_time_horizons_required(self, tmp_path):
        """time_horizons key is required."""
        config_path = self._make_config(tmp_path, [None])
        with open(config_path) as f:
            config = json.load(f)
        del config["time_horizons"]
        with open(config_path, "w") as f:
            json.dump(config, f)

        with pytest.raises(KeyError):
            DatasetGenerator.load_dataset_config(config_path)

    def test_time_horizons_null(self, tmp_path):
        """null in time_horizons means no constraint."""
        config_path = self._make_config(tmp_path, [None])
        result = DatasetGenerator.load_dataset_config(config_path)
        assert result.time_horizons == [None]

    def test_time_horizons_single_value(self, tmp_path):
        """Single time horizon value."""
        config_path = self._make_config(tmp_path, [[5, "years"]])
        result = DatasetGenerator.load_dataset_config(config_path)
        assert len(result.time_horizons) == 1
        assert result.time_horizons[0].value == 5
        assert result.time_horizons[0].unit == "years"

    def test_time_horizons_multiple(self, tmp_path):
        """Multiple time horizons including null."""
        horizons = [None, [6, "months"], [2, "years"], [10, "years"]]
        config_path = self._make_config(tmp_path, horizons)
        result = DatasetGenerator.load_dataset_config(config_path)
        assert len(result.time_horizons) == 4
        assert result.time_horizons[0] is None
        assert result.time_horizons[1].value == 6
        assert result.time_horizons[2].value == 2
        assert result.time_horizons[3].value == 10

    def test_time_horizons_empty_list(self, tmp_path):
        """Empty time_horizons list is valid but produces no samples."""
        config_path = self._make_config(tmp_path, [])
        result = DatasetGenerator.load_dataset_config(config_path)
        assert result.time_horizons == []


class TestAddFormattingVariations:
    """Tests for add_formatting_variations key."""

    def _make_config(self, tmp_path, add_variations=None):
        """Create config with optional formatting variations."""
        config = {
            "name": "test",
            "context": {
                "reward_unit": "units",
                "role": "tester",
                "situation": "test",
                "action_in_question": "choose",
                "reasoning_ask": "why",
                "domain": "test",
            },
            "options": {
                "short_term": {
                    "reward_range": [100, 200],
                    "time_range": [[1, "months"], [3, "months"]],
                    "reward_steps": [0, "linear"],
                    "time_steps": [0, "linear"],
                },
                "long_term": {
                    "reward_range": [500, 1000],
                    "time_range": [[1, "years"], [2, "years"]],
                    "reward_steps": [0, "linear"],
                    "time_steps": [0, "linear"],
                },
            },
            "time_horizons": [None],
        }
        if add_variations is not None:
            config["add_formatting_variations"] = add_variations
        config_path = tmp_path / "test.json"
        with open(config_path, "w") as f:
            json.dump(config, f)
        return config_path

    def test_defaults_to_false(self, tmp_path):
        """add_formatting_variations defaults to False."""
        config_path = self._make_config(tmp_path)
        result = DatasetGenerator.load_dataset_config(config_path)
        assert result.add_formatting_variations is False

    def test_explicit_false(self, tmp_path):
        """add_formatting_variations=false works."""
        config_path = self._make_config(tmp_path, add_variations=False)
        result = DatasetGenerator.load_dataset_config(config_path)
        assert result.add_formatting_variations is False

    def test_explicit_true(self, tmp_path):
        """add_formatting_variations=true works."""
        config_path = self._make_config(tmp_path, add_variations=True)
        result = DatasetGenerator.load_dataset_config(config_path)
        assert result.add_formatting_variations is True


class TestFormattingConfigKeys:
    """Tests for formatting config keys."""

    def _make_config(self, tmp_path, **overrides):
        """Create formatting config with overrides."""
        config = {
            "question_template": overrides.get(
                "question_template",
                "Situation: [SITUATION]\nYou are [ROLE]. Choose: [LEFT_TERM_LABEL] or [RIGHT_TERM_LABEL]"
            ),
            "response_format": overrides.get(
                "response_format",
                "[CHOICE_PREFIX] <choice>"
            ),
        }
        if "choice_prefix" in overrides:
            config["choice_prefix"] = overrides["choice_prefix"]
        if "time_horizon_spec" in overrides:
            config["time_horizon_spec"] = overrides["time_horizon_spec"]
        if "max_reasoning_length" in overrides:
            config["max_reasoning_length"] = overrides["max_reasoning_length"]

        config_path = tmp_path / "formatting.json"
        with open(config_path, "w") as f:
            json.dump(config, f)
        return config_path

    def test_question_template_required(self, tmp_path):
        """question_template is required."""
        config_path = self._make_config(tmp_path)
        with open(config_path) as f:
            config = json.load(f)
        del config["question_template"]
        with open(config_path, "w") as f:
            json.dump(config, f)

        with pytest.raises(KeyError):
            DatasetGenerator.load_formatting_config(config_path)

    def test_response_format_required(self, tmp_path):
        """response_format is required."""
        config_path = self._make_config(tmp_path)
        with open(config_path) as f:
            config = json.load(f)
        del config["response_format"]
        with open(config_path, "w") as f:
            json.dump(config, f)

        with pytest.raises(KeyError):
            DatasetGenerator.load_formatting_config(config_path)

    def test_choice_prefix_default(self, tmp_path):
        """choice_prefix defaults to 'I choose:'."""
        config_path = self._make_config(tmp_path)
        result = DatasetGenerator.load_formatting_config(config_path)
        assert result.choice_prefix == "I choose:"

    def test_choice_prefix_custom(self, tmp_path):
        """choice_prefix can be customized."""
        prefixes = ["I select:", "My answer:", "Option:", "Choice:"]
        for prefix in prefixes:
            config_path = self._make_config(tmp_path, choice_prefix=prefix)
            result = DatasetGenerator.load_formatting_config(config_path)
            assert result.choice_prefix == prefix

    def test_time_horizon_spec_default(self, tmp_path):
        """time_horizon_spec defaults to empty string."""
        config_path = self._make_config(tmp_path)
        result = DatasetGenerator.load_formatting_config(config_path)
        assert result.time_horizon_spec == ""

    def test_time_horizon_spec_custom(self, tmp_path):
        """time_horizon_spec can include [TIME_HORIZON] placeholder."""
        spec = "Consider outcome in [TIME_HORIZON]."
        config_path = self._make_config(tmp_path, time_horizon_spec=spec)
        result = DatasetGenerator.load_formatting_config(config_path)
        assert result.time_horizon_spec == spec

    def test_max_reasoning_length_default(self, tmp_path):
        """max_reasoning_length defaults to '1-2 sentences'."""
        config_path = self._make_config(tmp_path)
        result = DatasetGenerator.load_formatting_config(config_path)
        assert result.max_reasoning_length == "1-2 sentences"

    def test_max_reasoning_length_custom(self, tmp_path):
        """max_reasoning_length can be customized."""
        lengths = ["1 sentence", "2-3 sentences", "a brief paragraph", "50 words"]
        for length in lengths:
            config_path = self._make_config(tmp_path, max_reasoning_length=length)
            result = DatasetGenerator.load_formatting_config(config_path)
            assert result.max_reasoning_length == length


class TestDecodingConfigKeys:
    """Tests for decoding config keys (used in query configs)."""

    def test_max_new_tokens_default(self):
        """max_new_tokens defaults to 256."""
        config = DecodingConfig()
        assert config.max_new_tokens == 256

    def test_max_new_tokens_values(self):
        """max_new_tokens accepts various values."""
        for tokens in [1, 10, 100, 256, 1024, 4096]:
            config = DecodingConfig(max_new_tokens=tokens)
            assert config.max_new_tokens == tokens

    def test_temperature_default(self):
        """temperature defaults to 0.0 (deterministic)."""
        config = DecodingConfig()
        assert config.temperature == 0.0

    def test_temperature_range(self):
        """temperature accepts values in valid range."""
        for temp in [0.0, 0.1, 0.5, 0.7, 1.0, 1.5, 2.0]:
            config = DecodingConfig(temperature=temp)
            assert config.temperature == temp

    def test_top_k_default(self):
        """top_k defaults to 0 (disabled)."""
        config = DecodingConfig()
        assert config.top_k == 0

    def test_top_k_values(self):
        """top_k accepts various values."""
        for k in [0, 1, 10, 50, 100]:
            config = DecodingConfig(top_k=k)
            assert config.top_k == k

    def test_top_p_default(self):
        """top_p defaults to 1.0 (disabled)."""
        config = DecodingConfig()
        assert config.top_p == 1.0

    def test_top_p_range(self):
        """top_p accepts values in [0, 1]."""
        for p in [0.0, 0.1, 0.5, 0.9, 0.95, 1.0]:
            config = DecodingConfig(top_p=p)
            assert config.top_p == p


class TestTimeValueParsing:
    """Tests for TimeValue parsing edge cases."""

    def test_parse_list_format(self):
        """Parse [value, unit] list format."""
        tv = DatasetGenerator.parse_time_value([5, "months"])
        assert tv.value == 5
        assert tv.unit == "months"

    def test_parse_string_format(self):
        """Parse 'value unit' string format."""
        tv = DatasetGenerator.parse_time_value("5 months")
        assert tv.value == 5
        assert tv.unit == "months"

    def test_parse_dict_format(self):
        """Parse {value, unit} dict format."""
        tv = DatasetGenerator.parse_time_value({"value": 5, "unit": "months"})
        assert tv.value == 5
        assert tv.unit == "months"

    def test_parse_float_value(self):
        """Parse float values."""
        tv = DatasetGenerator.parse_time_value([2.5, "years"])
        assert tv.value == 2.5
        assert tv.unit == "years"

    def test_parse_normalizes_singular(self):
        """Parse normalizes singular to plural."""
        singular_units = ["month", "year", "day", "week"]
        plural_units = ["months", "years", "days", "weeks"]

        for singular, plural in zip(singular_units, plural_units):
            tv = DatasetGenerator.parse_time_value([1, singular])
            assert tv.unit == plural

    def test_parse_case_insensitive(self):
        """Parse is case insensitive for units."""
        for unit in ["MONTHS", "Months", "months", "YEARS", "Years"]:
            tv = DatasetGenerator.parse_time_value([1, unit])
            assert tv.unit in ["months", "years"]

    def test_parse_invalid_format(self):
        """Parse raises for invalid formats."""
        with pytest.raises(ValueError):
            DatasetGenerator.parse_time_value("invalid")

        with pytest.raises(ValueError):
            DatasetGenerator.parse_time_value("5")

        with pytest.raises(ValueError):
            DatasetGenerator.parse_time_value(12345)


class TestSingleQuerySpec:
    """Tests for SingleQuerySpec query_id generation."""

    def test_different_models_produce_different_query_ids(self):
        """Different models should produce different query_ids."""
        base_spec = {
            "dataset_id": "test_dataset_id",
            "formatting_id": "test_formatting_id",
            "decoding": DecodingConfig(),
            "internals": InternalsConfig(),
            "subsample": 1.0,
        }

        spec1 = SingleQuerySpec(model="model-A", **base_spec)
        spec2 = SingleQuerySpec(model="model-B", **base_spec)

        assert spec1.get_id() != spec2.get_id(), \
            "Different models should produce different query_ids"

    def test_same_config_produces_same_query_id(self):
        """Same configuration should produce the same query_id."""
        spec1 = SingleQuerySpec(
            dataset_id="test_dataset",
            model="test-model",
            formatting_id="test_formatting",
            decoding=DecodingConfig(),
            internals=InternalsConfig(),
            subsample=1.0,
        )
        spec2 = SingleQuerySpec(
            dataset_id="test_dataset",
            model="test-model",
            formatting_id="test_formatting",
            decoding=DecodingConfig(),
            internals=InternalsConfig(),
            subsample=1.0,
        )

        assert spec1.get_id() == spec2.get_id(), \
            "Same configuration should produce the same query_id"

    def test_different_datasets_produce_different_query_ids(self):
        """Different datasets should produce different query_ids."""
        base_spec = {
            "model": "test-model",
            "formatting_id": "test_formatting",
            "decoding": DecodingConfig(),
            "internals": InternalsConfig(),
            "subsample": 1.0,
        }

        spec1 = SingleQuerySpec(dataset_id="dataset-A", **base_spec)
        spec2 = SingleQuerySpec(dataset_id="dataset-B", **base_spec)

        assert spec1.get_id() != spec2.get_id(), \
            "Different datasets should produce different query_ids"

    def test_different_decoding_produces_different_query_ids(self):
        """Different decoding configs should produce different query_ids."""
        base_spec = {
            "dataset_id": "test_dataset",
            "model": "test-model",
            "formatting_id": "test_formatting",
            "internals": InternalsConfig(),
            "subsample": 1.0,
        }

        spec1 = SingleQuerySpec(decoding=DecodingConfig(temperature=0.0), **base_spec)
        spec2 = SingleQuerySpec(decoding=DecodingConfig(temperature=0.7), **base_spec)

        assert spec1.get_id() != spec2.get_id(), \
            "Different decoding configs should produce different query_ids"

    def test_different_subsample_produces_different_query_ids(self):
        """Different subsample values should produce different query_ids."""
        base_spec = {
            "dataset_id": "test_dataset",
            "model": "test-model",
            "formatting_id": "test_formatting",
            "decoding": DecodingConfig(),
            "internals": InternalsConfig(),
        }

        spec1 = SingleQuerySpec(subsample=1.0, **base_spec)
        spec2 = SingleQuerySpec(subsample=0.5, **base_spec)

        assert spec1.get_id() != spec2.get_id(), \
            "Different subsample values should produce different query_ids"

    def test_different_internals_produces_different_query_ids(self):
        """Different internals configs should produce different query_ids."""
        base_spec = {
            "dataset_id": "test_dataset",
            "model": "test-model",
            "formatting_id": "test_formatting",
            "decoding": DecodingConfig(),
            "subsample": 1.0,
        }

        spec1 = SingleQuerySpec(internals=InternalsConfig(), **base_spec)
        spec2 = SingleQuerySpec(
            internals=InternalsConfig(activations={"resid_post": {"layers": [0, -1]}}),
            **base_spec
        )

        assert spec1.get_id() != spec2.get_id(), \
            "Different internals configs should produce different query_ids"
