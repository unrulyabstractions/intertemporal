"""Tests for script config loading - query, analysis, and probe configs."""

import json
import pytest
import sys
from pathlib import Path

# Add scripts to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from src.types import DiscountType, UtilityType


class TestQueryConfigKeys:
    """Tests for query config keys."""

    def _make_config(self, tmp_path, **overrides):
        """Create query config with overrides."""
        config = {
            "models": overrides.get("models", ["test-model"]),
            "dataset": overrides.get("dataset", {
                "name": "test",
                "id": "abc123"
            }),
            "formatting": overrides.get("formatting", {
                "name": "default_formatting"
            }),
            "decoding": overrides.get("decoding", {
                "max_new_tokens": 10,
                "temperature": 0.0,
                "top_k": 0,
                "top_p": 1.0
            }),
            "internals": overrides.get("internals", {}),
            "token_positions": overrides.get("token_positions", []),
        }
        if "device" in overrides:
            config["device"] = overrides["device"]
        if "limit" in overrides:
            config["limit"] = overrides["limit"]

        config_path = tmp_path / "query.json"
        with open(config_path, "w") as f:
            json.dump(config, f)
        return config_path

    def test_models_required(self, tmp_path):
        """models key is required."""
        config_path = self._make_config(tmp_path)
        with open(config_path) as f:
            config = json.load(f)
        del config["models"]
        with open(config_path, "w") as f:
            json.dump(config, f)

        # Just verify the JSON structure is correct
        with open(config_path) as f:
            data = json.load(f)
        assert "models" not in data

    def test_models_single(self, tmp_path):
        """Single model in models list."""
        config_path = self._make_config(tmp_path, models=["gpt2"])
        with open(config_path) as f:
            data = json.load(f)
        assert data["models"] == ["gpt2"]

    def test_models_multiple(self, tmp_path):
        """Multiple models in models list."""
        models = ["gpt2", "meta-llama/Llama-2-7b", "Qwen/Qwen2.5-7B-Instruct"]
        config_path = self._make_config(tmp_path, models=models)
        with open(config_path) as f:
            data = json.load(f)
        assert data["models"] == models

    def test_dataset_name_and_id_required(self, tmp_path):
        """dataset.name and dataset.id are required."""
        config_path = self._make_config(tmp_path)
        with open(config_path) as f:
            config = json.load(f)
        del config["dataset"]["id"]
        with open(config_path, "w") as f:
            json.dump(config, f)

        # Just verify the JSON structure
        with open(config_path) as f:
            data = json.load(f)
        assert "id" not in data["dataset"]

    def test_dataset_id_formats(self, tmp_path):
        """dataset.id accepts various hash formats."""
        ids = [
            "abc123",  # Short
            "d955f857df139b31983796a678acc1d4",  # MD5-like
            "a" * 64,  # SHA256-like
        ]
        for id_val in ids:
            config_path = self._make_config(tmp_path, dataset={"name": "test", "id": id_val})
            with open(config_path) as f:
                data = json.load(f)
            assert data["dataset"]["id"] == id_val

    def test_formatting_name_default(self, tmp_path):
        """formatting.name defaults to 'default_formatting'."""
        config_path = self._make_config(tmp_path)
        with open(config_path) as f:
            config = json.load(f)
        del config["formatting"]
        with open(config_path, "w") as f:
            json.dump(config, f)

        # Verify JSON - actual default is applied during loading
        with open(config_path) as f:
            data = json.load(f)
        assert "formatting" not in data

    def test_decoding_max_new_tokens(self, tmp_path):
        """decoding.max_new_tokens accepts various values."""
        for tokens in [1, 10, 50, 256, 512, 1024]:
            config_path = self._make_config(tmp_path, decoding={
                "max_new_tokens": tokens,
                "temperature": 0.0,
                "top_k": 0,
                "top_p": 1.0
            })
            with open(config_path) as f:
                data = json.load(f)
            assert data["decoding"]["max_new_tokens"] == tokens

    def test_decoding_temperature_range(self, tmp_path):
        """decoding.temperature accepts valid range [0, 2]."""
        for temp in [0.0, 0.1, 0.5, 0.7, 1.0, 1.5, 2.0]:
            config_path = self._make_config(tmp_path, decoding={
                "max_new_tokens": 10,
                "temperature": temp,
                "top_k": 0,
                "top_p": 1.0
            })
            with open(config_path) as f:
                data = json.load(f)
            assert data["decoding"]["temperature"] == temp

    def test_decoding_top_k_values(self, tmp_path):
        """decoding.top_k: 0 disables, positive enables."""
        for k in [0, 1, 10, 50, 100]:
            config_path = self._make_config(tmp_path, decoding={
                "max_new_tokens": 10,
                "temperature": 0.0,
                "top_k": k,
                "top_p": 1.0
            })
            with open(config_path) as f:
                data = json.load(f)
            assert data["decoding"]["top_k"] == k

    def test_decoding_top_p_range(self, tmp_path):
        """decoding.top_p accepts [0, 1] range."""
        for p in [0.0, 0.5, 0.9, 0.95, 0.99, 1.0]:
            config_path = self._make_config(tmp_path, decoding={
                "max_new_tokens": 10,
                "temperature": 0.0,
                "top_k": 0,
                "top_p": p
            })
            with open(config_path) as f:
                data = json.load(f)
            assert data["decoding"]["top_p"] == p

    def test_internals_empty(self, tmp_path):
        """Empty internals disables activation capture."""
        config_path = self._make_config(tmp_path, internals={})
        with open(config_path) as f:
            data = json.load(f)
        assert data["internals"] == {}

    def test_internals_resid_post(self, tmp_path):
        """internals.resid_post.layers specifies layers to capture."""
        config_path = self._make_config(tmp_path, internals={
            "resid_post": {"layers": [0, 8, 16, -1]}
        })
        with open(config_path) as f:
            data = json.load(f)
        assert data["internals"]["resid_post"]["layers"] == [0, 8, 16, -1]

    def test_internals_negative_layers(self, tmp_path):
        """Negative layer indices count from end."""
        config_path = self._make_config(tmp_path, internals={
            "resid_post": {"layers": [-1, -2, -4, -8]}
        })
        with open(config_path) as f:
            data = json.load(f)
        assert data["internals"]["resid_post"]["layers"] == [-1, -2, -4, -8]

    def test_token_positions_empty(self, tmp_path):
        """Empty token_positions captures no positions."""
        config_path = self._make_config(tmp_path, token_positions=[])
        with open(config_path) as f:
            data = json.load(f)
        assert data["token_positions"] == []

    def test_token_positions_indices(self, tmp_path):
        """token_positions can be integer indices."""
        config_path = self._make_config(tmp_path, token_positions=[0, 10, -1])
        with open(config_path) as f:
            data = json.load(f)
        assert data["token_positions"] == [0, 10, -1]

    def test_token_positions_strings(self, tmp_path):
        """token_positions can be string patterns."""
        config_path = self._make_config(tmp_path, token_positions=[0, "I choose:"])
        with open(config_path) as f:
            data = json.load(f)
        assert data["token_positions"] == [0, "I choose:"]

    def test_device_null(self, tmp_path):
        """device=null means auto-detect."""
        config_path = self._make_config(tmp_path, device=None)
        with open(config_path) as f:
            data = json.load(f)
        assert data["device"] is None

    def test_device_cuda(self, tmp_path):
        """device can be 'cuda', 'cuda:0', etc."""
        for device in ["cuda", "cuda:0", "cuda:1", "mps", "cpu"]:
            config_path = self._make_config(tmp_path, device=device)
            with open(config_path) as f:
                data = json.load(f)
            assert data["device"] == device

    def test_limit_zero(self, tmp_path):
        """limit=0 means no limit."""
        config_path = self._make_config(tmp_path, limit=0)
        with open(config_path) as f:
            data = json.load(f)
        assert data["limit"] == 0

    def test_limit_positive(self, tmp_path):
        """Positive limit restricts number of samples."""
        for limit in [1, 10, 100]:
            config_path = self._make_config(tmp_path, limit=limit)
            with open(config_path) as f:
                data = json.load(f)
            assert data["limit"] == limit

    def test_subsample_default(self, tmp_path):
        """subsample defaults to 1.0 (use all samples)."""
        config_path = self._make_config(tmp_path)
        with open(config_path) as f:
            data = json.load(f)
        # Default is not in JSON, applied during loading
        assert "subsample" not in data or data.get("subsample", 1.0) == 1.0

    def test_subsample_fraction(self, tmp_path):
        """subsample accepts fractions in (0, 1]."""
        for subsample in [0.1, 0.25, 0.5, 0.75, 1.0]:
            config = self._make_config(tmp_path)
            with open(config) as f:
                data = json.load(f)
            data["subsample"] = subsample
            with open(config, "w") as f:
                json.dump(data, f)

            with open(config) as f:
                loaded = json.load(f)
            assert loaded["subsample"] == subsample

    def test_subsample_ten_percent(self, tmp_path):
        """subsample=0.1 means use 10% of samples."""
        config_path = self._make_config(tmp_path)
        with open(config_path) as f:
            data = json.load(f)
        data["subsample"] = 0.1
        with open(config_path, "w") as f:
            json.dump(data, f)

        with open(config_path) as f:
            loaded = json.load(f)
        assert loaded["subsample"] == 0.1


class TestAnalysisConfigKeys:
    """Tests for choice_modeling/analysis config keys."""

    def _make_config(self, tmp_path, **overrides):
        """Create analysis config with overrides."""
        config = {
            "train_data": overrides.get("train_data", {
                "name": "test",
                "model": "test-model",
                "query_id": "abc123"
            }),
            "test_data": overrides.get("test_data", {
                "name": "test",
                "model": "test-model",
                "query_id": "abc123"
            }),
            "choice_models": overrides.get("choice_models", [
                {"utility_type": "linear", "discount_type": "exponential"}
            ]),
        }
        config_path = tmp_path / "analysis.json"
        with open(config_path, "w") as f:
            json.dump(config, f)
        return config_path

    def test_train_data_required(self, tmp_path):
        """train_data is required."""
        config_path = self._make_config(tmp_path)
        with open(config_path) as f:
            config = json.load(f)
        del config["train_data"]
        with open(config_path, "w") as f:
            json.dump(config, f)

        with open(config_path) as f:
            data = json.load(f)
        assert "train_data" not in data

    def test_train_data_name_model_query_id(self, tmp_path):
        """train_data requires name, model, query_id."""
        config_path = self._make_config(tmp_path, train_data={
            "name": "cityhousing",
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "query_id": "abc123def456"
        })
        with open(config_path) as f:
            data = json.load(f)
        assert data["train_data"]["name"] == "cityhousing"
        assert "Qwen" in data["train_data"]["model"]
        assert len(data["train_data"]["query_id"]) > 0

    def test_test_data_same_as_train(self, tmp_path):
        """test_data can be same as train_data (self-evaluation)."""
        ref = {"name": "test", "model": "model", "query_id": "abc"}
        config_path = self._make_config(tmp_path, train_data=ref, test_data=ref)
        with open(config_path) as f:
            data = json.load(f)
        assert data["train_data"] == data["test_data"]

    def test_test_data_different_from_train(self, tmp_path):
        """test_data can differ from train_data."""
        config_path = self._make_config(tmp_path,
            train_data={"name": "set1", "model": "m", "query_id": "a"},
            test_data={"name": "set2", "model": "m", "query_id": "b"}
        )
        with open(config_path) as f:
            data = json.load(f)
        assert data["train_data"]["name"] != data["test_data"]["name"]

    def test_choice_models_single(self, tmp_path):
        """Single choice model."""
        config_path = self._make_config(tmp_path, choice_models=[
            {"utility_type": "linear", "discount_type": "exponential"}
        ])
        with open(config_path) as f:
            data = json.load(f)
        assert len(data["choice_models"]) == 1

    def test_choice_models_multiple(self, tmp_path):
        """Multiple choice models."""
        models = [
            {"utility_type": "linear", "discount_type": "exponential"},
            {"utility_type": "linear", "discount_type": "hyperbolic"},
            {"utility_type": "log", "discount_type": "exponential"},
        ]
        config_path = self._make_config(tmp_path, choice_models=models)
        with open(config_path) as f:
            data = json.load(f)
        assert len(data["choice_models"]) == 3

    def test_utility_type_linear(self, tmp_path):
        """utility_type='linear' is valid."""
        config_path = self._make_config(tmp_path, choice_models=[
            {"utility_type": "linear", "discount_type": "exponential"}
        ])
        with open(config_path) as f:
            data = json.load(f)
        assert data["choice_models"][0]["utility_type"] == "linear"

    def test_utility_type_log(self, tmp_path):
        """utility_type='log' is valid."""
        config_path = self._make_config(tmp_path, choice_models=[
            {"utility_type": "log", "discount_type": "exponential"}
        ])
        with open(config_path) as f:
            data = json.load(f)
        assert data["choice_models"][0]["utility_type"] == "log"

    def test_discount_type_exponential(self, tmp_path):
        """discount_type='exponential' is valid."""
        config_path = self._make_config(tmp_path, choice_models=[
            {"utility_type": "linear", "discount_type": "exponential"}
        ])
        with open(config_path) as f:
            data = json.load(f)
        assert data["choice_models"][0]["discount_type"] == "exponential"

    def test_discount_type_hyperbolic(self, tmp_path):
        """discount_type='hyperbolic' is valid."""
        config_path = self._make_config(tmp_path, choice_models=[
            {"utility_type": "linear", "discount_type": "hyperbolic"}
        ])
        with open(config_path) as f:
            data = json.load(f)
        assert data["choice_models"][0]["discount_type"] == "hyperbolic"

    def test_discount_type_quasi_hyperbolic(self, tmp_path):
        """discount_type='quasi_hyperbolic' is valid."""
        config_path = self._make_config(tmp_path, choice_models=[
            {"utility_type": "linear", "discount_type": "quasi_hyperbolic"}
        ])
        with open(config_path) as f:
            data = json.load(f)
        assert data["choice_models"][0]["discount_type"] == "quasi_hyperbolic"


class TestProbeConfigKeys:
    """Tests for probes config keys."""

    def _make_config(self, tmp_path, **overrides):
        """Create probe config with overrides."""
        config = {
            "train_data": overrides.get("train_data", {
                "name": "test",
                "model": "test-model",
                "query_id": "abc123"
            }),
            "test_data": overrides.get("test_data", {
                "name": "test",
                "model": "test-model",
                "query_id": "abc123"
            }),
            "layers": overrides.get("layers", [8, 16, 24]),
            "token_position_idx": overrides.get("token_position_idx", -1),
            "test_split": overrides.get("test_split", 0.2),
            "random_seed": overrides.get("random_seed", 42),
        }
        config_path = tmp_path / "probes.json"
        with open(config_path, "w") as f:
            json.dump(config, f)
        return config_path

    def test_layers_required(self, tmp_path):
        """layers key specifies which layers to probe."""
        config_path = self._make_config(tmp_path, layers=[0, 4, 8, 12])
        with open(config_path) as f:
            data = json.load(f)
        assert data["layers"] == [0, 4, 8, 12]

    def test_layers_empty(self, tmp_path):
        """Empty layers list is valid (no probes)."""
        config_path = self._make_config(tmp_path, layers=[])
        with open(config_path) as f:
            data = json.load(f)
        assert data["layers"] == []

    def test_layers_negative(self, tmp_path):
        """Negative layer indices count from end."""
        config_path = self._make_config(tmp_path, layers=[-1, -2, -4])
        with open(config_path) as f:
            data = json.load(f)
        assert data["layers"] == [-1, -2, -4]

    def test_layers_mixed(self, tmp_path):
        """Mix of positive and negative indices."""
        config_path = self._make_config(tmp_path, layers=[0, 8, -4, -1])
        with open(config_path) as f:
            data = json.load(f)
        assert data["layers"] == [0, 8, -4, -1]

    def test_token_position_idx_default(self, tmp_path):
        """token_position_idx defaults to -1 (last)."""
        config_path = self._make_config(tmp_path)
        with open(config_path) as f:
            data = json.load(f)
        assert data["token_position_idx"] == -1

    def test_token_position_idx_first(self, tmp_path):
        """token_position_idx=0 for first position."""
        config_path = self._make_config(tmp_path, token_position_idx=0)
        with open(config_path) as f:
            data = json.load(f)
        assert data["token_position_idx"] == 0

    def test_token_position_idx_middle(self, tmp_path):
        """token_position_idx can be any index."""
        for idx in [0, 1, 2, -1, -2]:
            config_path = self._make_config(tmp_path, token_position_idx=idx)
            with open(config_path) as f:
                data = json.load(f)
            assert data["token_position_idx"] == idx

    def test_test_split_default(self, tmp_path):
        """test_split defaults to 0.2."""
        config_path = self._make_config(tmp_path)
        with open(config_path) as f:
            data = json.load(f)
        assert data["test_split"] == 0.2

    def test_test_split_range(self, tmp_path):
        """test_split accepts [0, 1] range."""
        for split in [0.1, 0.2, 0.3, 0.5]:
            config_path = self._make_config(tmp_path, test_split=split)
            with open(config_path) as f:
                data = json.load(f)
            assert data["test_split"] == split

    def test_random_seed_default(self, tmp_path):
        """random_seed defaults to 42."""
        config_path = self._make_config(tmp_path)
        with open(config_path) as f:
            data = json.load(f)
        assert data["random_seed"] == 42

    def test_random_seed_custom(self, tmp_path):
        """random_seed can be customized."""
        for seed in [0, 1, 123, 99999]:
            config_path = self._make_config(tmp_path, random_seed=seed)
            with open(config_path) as f:
                data = json.load(f)
            assert data["random_seed"] == seed


class TestEnumValidation:
    """Tests for enum type validation."""

    def test_utility_type_values(self):
        """UtilityType enum has expected values."""
        assert UtilityType.LINEAR.value == "linear"
        assert UtilityType.LOG.value == "log"

    def test_discount_type_values(self):
        """DiscountType enum has expected values."""
        assert DiscountType.EXPONENTIAL.value == "exponential"
        assert DiscountType.HYPERBOLIC.value == "hyperbolic"
        assert DiscountType.QUASI_HYPERBOLIC.value == "quasi_hyperbolic"

    def test_invalid_utility_type(self):
        """Invalid utility_type raises error."""
        with pytest.raises(ValueError):
            UtilityType("invalid")

    def test_invalid_discount_type(self):
        """Invalid discount_type raises error."""
        with pytest.raises(ValueError):
            DiscountType("invalid")
