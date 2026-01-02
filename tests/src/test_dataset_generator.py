"""Tests for src/dataset_generator.py - Dataset generation."""

import json
import pytest
from pathlib import Path

from src.dataset_generator import DatasetGenerator
from src.common.schemas import DatasetConfig, FormattingConfig


class TestDatasetGenerator:
    """Tests for DatasetGenerator class."""

    @pytest.fixture
    def test_config_path(self, project_root):
        """Path to a test dataset config."""
        return project_root / "scripts" / "configs" / "dataset" / "test" / "minimal.json"

    @pytest.fixture
    def formatting_config_path(self, project_root):
        """Path to test formatting config."""
        return project_root / "scripts" / "configs" / "formatting" / "test" / "minimal.json"

    def test_load_dataset_config(self, test_config_path):
        """Can load dataset config from JSON."""
        if not test_config_path.exists():
            pytest.skip("Test config not found")
        config = DatasetGenerator.load_dataset_config(test_config_path)
        assert isinstance(config, DatasetConfig)
        assert config.name is not None

    def test_load_formatting_config(self, formatting_config_path):
        """Can load formatting config from JSON."""
        if not formatting_config_path.exists():
            pytest.skip("Test formatting config not found")
        config = DatasetGenerator.load_formatting_config(formatting_config_path)
        assert isinstance(config, FormattingConfig)

    def test_generate_samples_count(self, test_config_path, formatting_config_path):
        """Generates correct number of samples."""
        if not test_config_path.exists() or not formatting_config_path.exists():
            pytest.skip("Test configs not found")

        # Load configs first, then create generator
        dataset_config = DatasetGenerator.load_dataset_config(test_config_path)
        formatting_config = DatasetGenerator.load_formatting_config(formatting_config_path)

        gen = DatasetGenerator(dataset_config, formatting_config)
        samples, metadata = gen.generate()

        # Number of samples should match grid size
        assert len(samples) > 0

    def test_samples_have_required_fields(self, test_config_path, formatting_config_path):
        """Generated samples have all required fields."""
        if not test_config_path.exists() or not formatting_config_path.exists():
            pytest.skip("Test configs not found")

        dataset_config = DatasetGenerator.load_dataset_config(test_config_path)
        formatting_config = DatasetGenerator.load_formatting_config(formatting_config_path)

        gen = DatasetGenerator(dataset_config, formatting_config)
        samples, metadata = gen.generate()

        for sample in samples:
            assert sample.id is not None
            assert sample.prompt is not None
            assert sample.prompt.question is not None
            assert sample.prompt.question.pair is not None
            assert sample.prompt.question.pair.short_term is not None
            assert sample.prompt.question.pair.long_term is not None


class TestDatasetGeneratorWithVariations:
    """Tests for formatting variations."""

    def test_variations_produce_different_labels(self, project_root, tmp_path):
        """With variations enabled, labels should vary."""
        # Create a temp config with variations
        config = {
            "name": "test_variations",
            "context": {
                "reward_unit": "dollars",
                "role": "a tester",
                "situation": "Test situation",
                "action_in_question": "choose",
                "reasoning_ask": "why",
                "domain": "test",
                "labels": ["a)", "b)"],
                "method": "grid",
                "seed": 42
            },
            "options": {
                "short_term": {
                    "reward_range": [100, 200],
                    "time_range": [[1, "months"], [3, "months"]],
                    "reward_steps": [1, "linear"],
                    "time_steps": [1, "linear"]
                },
                "long_term": {
                    "reward_range": [500, 600],
                    "time_range": [[1, "years"], [2, "years"]],
                    "reward_steps": [1, "linear"],
                    "time_steps": [1, "linear"]
                }
            },
            "time_horizons": [None],
            "add_formatting_variations": True
        }

        config_path = tmp_path / "test_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        formatting_path = project_root / "scripts" / "configs" / "formatting" / "default_formatting.json"
        if not formatting_path.exists():
            pytest.skip("Default formatting config not found")

        # Load configs and create generator
        dataset_config = DatasetGenerator.load_dataset_config(config_path)
        formatting_config = DatasetGenerator.load_formatting_config(formatting_path)

        gen = DatasetGenerator(dataset_config, formatting_config)
        samples, metadata = gen.generate()

        # With variations, we should see different labels across samples
        labels_seen = set()
        for sample in samples:
            pair = sample.prompt.question.pair
            labels_seen.add(pair.short_term.label)
            labels_seen.add(pair.long_term.label)

        # Should have more than just the default labels
        assert len(labels_seen) >= 2


class TestDatasetConfigParsing:
    """Tests for config parsing edge cases."""

    def test_null_time_horizon(self, project_root):
        """Can parse config with null time horizon."""
        config_path = project_root / "scripts" / "configs" / "dataset" / "test" / "no_time_horizon.json"
        if not config_path.exists():
            pytest.skip("Test config not found")

        config = DatasetGenerator.load_dataset_config(config_path)
        assert None in config.time_horizons

    def test_logarithmic_steps(self, project_root):
        """Can parse config with logarithmic steps."""
        config_path = project_root / "scripts" / "configs" / "dataset" / "test" / "logarithmic_steps.json"
        if not config_path.exists():
            pytest.skip("Test config not found")

        config = DatasetGenerator.load_dataset_config(config_path)
        # Check that logarithmic step type is correctly parsed
        from src.common.schemas import StepType
        assert config.options["long_term"].time_steps[1] == StepType.LOGARITHMIC
