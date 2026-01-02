"""Tests for output JSON schemas - dataset, preference, analysis outputs."""

import json
import pytest
import sys
from pathlib import Path

# Add scripts to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from common.schemas import (
    DatasetOutput,
    DatasetOutputMetadata,
    OptionOutput,
    PreferencePairOutput,
    QuestionOutput,
    PreferenceDataOutput,
    PreferenceDataMetadata,
    PreferenceItem,
    PreferenceDebugInfo,
    InternalsReference,
    EvaluationMetrics,
    ModelResultOutput,
    AnalysisOutput,
    ProbeResultOutput,
    ProbeTrainingOutput,
)


class TestOptionOutput:
    """Tests for OptionOutput schema."""

    def test_required_fields(self):
        """OptionOutput requires label, time, reward."""
        opt = OptionOutput(label="a)", time=[3, "months"], reward=100.0)
        assert opt.label == "a)"
        assert opt.time == [3, "months"]
        assert opt.reward == 100.0

    def test_label_variations(self):
        """label can be any string."""
        for label in ["a)", "1)", "[A]", "Option A", ""]:
            opt = OptionOutput(label=label, time=[1, "years"], reward=100.0)
            assert opt.label == label

    def test_time_format(self):
        """time must be [value, unit] list."""
        times = [
            [1, "days"],
            [2, "weeks"],
            [3, "months"],
            [5, "years"],
            [0.5, "years"],
        ]
        for time in times:
            opt = OptionOutput(label="a)", time=time, reward=100.0)
            assert opt.time == time

    def test_reward_numeric(self):
        """reward is numeric (int or float)."""
        for reward in [0, 100, 1000.5, 99999]:
            opt = OptionOutput(label="a)", time=[1, "years"], reward=reward)
            assert opt.reward == reward


class TestPreferencePairOutput:
    """Tests for PreferencePairOutput schema."""

    def test_required_fields(self):
        """PreferencePairOutput requires short_term and long_term."""
        short = OptionOutput(label="a)", time=[1, "months"], reward=100.0)
        long = OptionOutput(label="b)", time=[1, "years"], reward=500.0)
        pair = PreferencePairOutput(short_term=short, long_term=long)
        assert pair.short_term.reward == 100.0
        assert pair.long_term.reward == 500.0

    def test_short_term_before_long_term(self):
        """short_term time is typically less than long_term."""
        short = OptionOutput(label="a)", time=[1, "months"], reward=100.0)
        long = OptionOutput(label="b)", time=[12, "months"], reward=500.0)
        pair = PreferencePairOutput(short_term=short, long_term=long)
        # Verify short time is before long time
        assert pair.short_term.time[0] < pair.long_term.time[0]


class TestQuestionOutput:
    """Tests for QuestionOutput schema."""

    def test_required_fields(self):
        """QuestionOutput requires sample_id, question_text, preference_pair."""
        short = OptionOutput(label="a)", time=[1, "months"], reward=100.0)
        long = OptionOutput(label="b)", time=[1, "years"], reward=500.0)
        pair = PreferencePairOutput(short_term=short, long_term=long)

        q = QuestionOutput(
            sample_id=0,
            question_text="Choose an option",
            time_horizon=None,
            preference_pair=pair
        )
        assert q.sample_id == 0
        assert q.question_text == "Choose an option"

    def test_time_horizon_null(self):
        """time_horizon can be null (no constraint)."""
        short = OptionOutput(label="a)", time=[1, "months"], reward=100.0)
        long = OptionOutput(label="b)", time=[1, "years"], reward=500.0)
        pair = PreferencePairOutput(short_term=short, long_term=long)

        q = QuestionOutput(
            sample_id=0,
            question_text="Choose",
            time_horizon=None,
            preference_pair=pair
        )
        assert q.time_horizon is None

    def test_time_horizon_value(self):
        """time_horizon can be [value, unit] list."""
        short = OptionOutput(label="a)", time=[1, "months"], reward=100.0)
        long = OptionOutput(label="b)", time=[1, "years"], reward=500.0)
        pair = PreferencePairOutput(short_term=short, long_term=long)

        q = QuestionOutput(
            sample_id=0,
            question_text="Choose",
            time_horizon=[5, "years"],
            preference_pair=pair
        )
        assert q.time_horizon == [5, "years"]

    def test_sample_id_increments(self):
        """sample_id should be unique incrementing integers."""
        short = OptionOutput(label="a)", time=[1, "months"], reward=100.0)
        long = OptionOutput(label="b)", time=[1, "years"], reward=500.0)
        pair = PreferencePairOutput(short_term=short, long_term=long)

        for i in range(10):
            q = QuestionOutput(
                sample_id=i,
                question_text="Choose",
                time_horizon=None,
                preference_pair=pair
            )
            assert q.sample_id == i


class TestDatasetOutputMetadata:
    """Tests for DatasetOutputMetadata schema."""

    def test_required_fields(self):
        """Metadata requires version, dataset_id, dataset_run_id, config, num_questions."""
        meta = DatasetOutputMetadata(
            version="1.0",
            dataset_id="abc123",
            dataset_run_id="run123",
            config={"name": "test"},
            num_questions=10,
            timestamp="20251230_120000"
        )
        assert meta.version == "1.0"
        assert meta.dataset_id == "abc123"
        assert meta.dataset_run_id == "run123"
        assert meta.num_questions == 10

    def test_version_format(self):
        """version follows semver-like format."""
        for version in ["1.0", "1.1", "2.0"]:
            meta = DatasetOutputMetadata(
                version=version,
                dataset_id="abc",
                dataset_run_id="run",
                config={},
                num_questions=0
            )
            assert meta.version == version

    def test_dataset_id_hash(self):
        """dataset_id is a hash string."""
        ids = ["abc", "abc123def456", "a" * 32]
        for id_val in ids:
            meta = DatasetOutputMetadata(
                version="1.0",
                dataset_id=id_val,
                dataset_run_id="run",
                config={},
                num_questions=0
            )
            assert meta.dataset_id == id_val

    def test_timestamp_optional(self):
        """timestamp defaults to empty string."""
        meta = DatasetOutputMetadata(
            version="1.0",
            dataset_id="abc",
            dataset_run_id="run",
            config={},
            num_questions=0
        )
        assert meta.timestamp == ""


class TestPreferenceItem:
    """Tests for PreferenceItem schema."""

    def test_required_fields(self):
        """PreferenceItem requires sample_id, preference_pair, choice."""
        short = OptionOutput(label="a)", time=[1, "months"], reward=100.0)
        long = OptionOutput(label="b)", time=[1, "years"], reward=500.0)
        pair = PreferencePairOutput(short_term=short, long_term=long)

        item = PreferenceItem(
            sample_id=0,
            time_horizon=None,
            preference_pair=pair,
            choice="short_term"
        )
        assert item.sample_id == 0
        assert item.choice == "short_term"

    def test_choice_short_term(self):
        """choice='short_term' when short option selected."""
        short = OptionOutput(label="a)", time=[1, "months"], reward=100.0)
        long = OptionOutput(label="b)", time=[1, "years"], reward=500.0)
        pair = PreferencePairOutput(short_term=short, long_term=long)

        item = PreferenceItem(
            sample_id=0,
            time_horizon=None,
            preference_pair=pair,
            choice="short_term"
        )
        assert item.choice == "short_term"

    def test_choice_long_term(self):
        """choice='long_term' when long option selected."""
        short = OptionOutput(label="a)", time=[1, "months"], reward=100.0)
        long = OptionOutput(label="b)", time=[1, "years"], reward=500.0)
        pair = PreferencePairOutput(short_term=short, long_term=long)

        item = PreferenceItem(
            sample_id=0,
            time_horizon=None,
            preference_pair=pair,
            choice="long_term"
        )
        assert item.choice == "long_term"

    def test_choice_unknown(self):
        """choice='unknown' when parse failed."""
        short = OptionOutput(label="a)", time=[1, "months"], reward=100.0)
        long = OptionOutput(label="b)", time=[1, "years"], reward=500.0)
        pair = PreferencePairOutput(short_term=short, long_term=long)

        item = PreferenceItem(
            sample_id=0,
            time_horizon=None,
            preference_pair=pair,
            choice="unknown"
        )
        assert item.choice == "unknown"

    def test_internals_optional(self):
        """internals is optional (can be None)."""
        short = OptionOutput(label="a)", time=[1, "months"], reward=100.0)
        long = OptionOutput(label="b)", time=[1, "years"], reward=500.0)
        pair = PreferencePairOutput(short_term=short, long_term=long)

        item = PreferenceItem(
            sample_id=0,
            time_horizon=None,
            preference_pair=pair,
            choice="short_term",
            internals=None
        )
        assert item.internals is None

    def test_debug_optional(self):
        """debug is optional (can be None)."""
        short = OptionOutput(label="a)", time=[1, "months"], reward=100.0)
        long = OptionOutput(label="b)", time=[1, "years"], reward=500.0)
        pair = PreferencePairOutput(short_term=short, long_term=long)

        item = PreferenceItem(
            sample_id=0,
            time_horizon=None,
            preference_pair=pair,
            choice="short_term",
            debug=None
        )
        assert item.debug is None


class TestInternalsReference:
    """Tests for InternalsReference schema."""

    def test_required_fields(self):
        """InternalsReference requires file_path, activations, token_positions, tokens."""
        ref = InternalsReference(
            file_path="out/internals/test_sample_0.pt",
            activations=["blocks.8.hook_resid_post_pos45"],
            token_positions=[0, 45],
            tokens=["<bos>", "choose"]
        )
        assert "out/internals" in ref.file_path
        assert len(ref.activations) == 1
        assert len(ref.token_positions) == 2

    def test_file_path_relative(self):
        """file_path is relative to repo root."""
        ref = InternalsReference(
            file_path="out/internals/dataset_model_query_sample_0.pt",
            activations=[],
            token_positions=[],
            tokens=[]
        )
        assert ref.file_path.startswith("out/")

    def test_activations_naming(self):
        """activations follow blocks.N.hook_resid_post_posM pattern."""
        activations = [
            "blocks.0.hook_resid_post_pos0",
            "blocks.8.hook_resid_post_pos45",
            "blocks.24.hook_resid_post_pos100",
        ]
        ref = InternalsReference(
            file_path="test.pt",
            activations=activations,
            token_positions=[0, 45, 100],
            tokens=["a", "b", "c"]
        )
        for act in ref.activations:
            assert "blocks." in act
            assert "hook_resid_post_pos" in act


class TestPreferenceDebugInfo:
    """Tests for PreferenceDebugInfo schema."""

    def test_required_fields(self):
        """Debug info requires raw_prompt, raw_continuation."""
        debug = PreferenceDebugInfo(
            raw_prompt="Choose: a) or b)",
            raw_continuation="I choose a) because..."
        )
        assert debug.raw_prompt == "Choose: a) or b)"
        assert debug.raw_continuation == "I choose a) because..."

    def test_parsed_label_optional(self):
        """parsed_label is optional."""
        debug = PreferenceDebugInfo(
            raw_prompt="prompt",
            raw_continuation="response",
            parsed_label="a)"
        )
        assert debug.parsed_label == "a)"

    def test_parsed_label_none(self):
        """parsed_label can be None if parsing failed."""
        debug = PreferenceDebugInfo(
            raw_prompt="prompt",
            raw_continuation="unclear response",
            parsed_label=None
        )
        assert debug.parsed_label is None

    def test_token_counts(self):
        """Token counts are non-negative integers."""
        debug = PreferenceDebugInfo(
            raw_prompt="prompt",
            raw_continuation="response",
            num_prompt_tokens=150,
            num_continuation_tokens=45
        )
        assert debug.num_prompt_tokens == 150
        assert debug.num_continuation_tokens == 45


class TestEvaluationMetrics:
    """Tests for EvaluationMetrics schema."""

    def test_required_fields(self):
        """Metrics require accuracy, num_samples, num_correct."""
        metrics = EvaluationMetrics(
            accuracy=0.85,
            num_samples=100,
            num_correct=85
        )
        assert metrics.accuracy == 0.85
        assert metrics.num_samples == 100
        assert metrics.num_correct == 85

    def test_accuracy_range(self):
        """accuracy is in [0, 1] range."""
        for acc in [0.0, 0.5, 0.85, 1.0]:
            metrics = EvaluationMetrics(
                accuracy=acc,
                num_samples=100,
                num_correct=int(acc * 100)
            )
            assert 0.0 <= metrics.accuracy <= 1.0

    def test_short_long_term_accuracy(self):
        """Per-class accuracies are optional."""
        metrics = EvaluationMetrics(
            accuracy=0.85,
            num_samples=100,
            num_correct=85,
            short_term_accuracy=0.9,
            long_term_accuracy=0.8
        )
        assert metrics.short_term_accuracy == 0.9
        assert metrics.long_term_accuracy == 0.8

    def test_per_horizon_accuracy(self):
        """per_horizon_accuracy maps horizon strings to accuracies."""
        metrics = EvaluationMetrics(
            accuracy=0.85,
            num_samples=100,
            num_correct=85,
            per_horizon_accuracy={
                "no_horizon": 0.82,
                "5 years": 0.88,
                "10 years": 0.85
            }
        )
        assert "no_horizon" in metrics.per_horizon_accuracy
        assert metrics.per_horizon_accuracy["5 years"] == 0.88


class TestProbeResultOutput:
    """Tests for ProbeResultOutput schema."""

    def test_required_fields(self):
        """Probe result requires layer, accuracies, sample counts."""
        result = ProbeResultOutput(
            layer=8,
            cv_accuracy_mean=0.85,
            cv_accuracy_std=0.05,
            test_accuracy=0.9,
            n_train=100,
            n_test=20,
            n_features=768,
            confusion_matrix=[[10, 2], [1, 7]]
        )
        assert result.layer == 8
        assert result.cv_accuracy_mean == 0.85
        assert result.test_accuracy == 0.9

    def test_layer_values(self):
        """layer can be any valid layer index."""
        for layer in [0, 4, 8, 16, 24, 32]:
            result = ProbeResultOutput(
                layer=layer,
                cv_accuracy_mean=0.8,
                cv_accuracy_std=0.1,
                test_accuracy=0.85,
                n_train=80,
                n_test=20,
                n_features=768,
                confusion_matrix=[[8, 2], [1, 9]]
            )
            assert result.layer == layer

    def test_n_features_model_dependent(self):
        """n_features depends on model hidden size."""
        for n_features in [768, 1024, 2048, 4096]:
            result = ProbeResultOutput(
                layer=0,
                cv_accuracy_mean=0.8,
                cv_accuracy_std=0.1,
                test_accuracy=0.85,
                n_train=80,
                n_test=20,
                n_features=n_features,
                confusion_matrix=[[8, 2], [1, 9]]
            )
            assert result.n_features == n_features

    def test_confusion_matrix_2x2(self):
        """confusion_matrix is 2x2 for binary classification."""
        result = ProbeResultOutput(
            layer=0,
            cv_accuracy_mean=0.8,
            cv_accuracy_std=0.1,
            test_accuracy=0.85,
            n_train=80,
            n_test=20,
            n_features=768,
            confusion_matrix=[[15, 5], [3, 17]]
        )
        assert len(result.confusion_matrix) == 2
        assert len(result.confusion_matrix[0]) == 2
        assert len(result.confusion_matrix[1]) == 2


class TestOutputFileLoading:
    """Tests for loading output files from out/*/test/ fixtures."""

    @pytest.fixture
    def project_root(self):
        return PROJECT_ROOT

    def test_load_dataset_output(self, project_root):
        """Can load dataset output from test fixture."""
        fixture_path = project_root / "out" / "datasets" / "test" / "minimal.json"
        if not fixture_path.exists():
            pytest.skip("Test fixture not found")

        with open(fixture_path) as f:
            data = json.load(f)

        assert "metadata" in data
        assert "questions" in data
        assert data["metadata"]["version"] == "1.0"

    def test_load_preference_data_output(self, project_root):
        """Can load preference data output from test fixture."""
        fixture_path = project_root / "out" / "preference_data" / "test" / "minimal.json"
        if not fixture_path.exists():
            pytest.skip("Test fixture not found")

        with open(fixture_path) as f:
            data = json.load(f)

        assert "metadata" in data
        assert "preferences" in data

    def test_load_analysis_output(self, project_root):
        """Can load analysis output from test fixture."""
        fixture_path = project_root / "out" / "choice_modeling" / "test" / "analysis_minimal.json"
        if not fixture_path.exists():
            pytest.skip("Test fixture not found")

        with open(fixture_path) as f:
            data = json.load(f)

        # Check expected keys (actual format uses "models" and "train_data")
        assert "models" in data or "results" in data
        assert "train_data" in data or "train_data_path" in data


class TestSchemaValidation:
    """Tests for schema validation edge cases."""

    def test_empty_questions_list(self):
        """Dataset with empty questions list is valid."""
        meta = DatasetOutputMetadata(
            version="1.0",
            dataset_id="test",
            dataset_run_id="test_run",
            config={},
            num_questions=0
        )
        output = DatasetOutput(metadata=meta, questions=[])
        assert len(output.questions) == 0

    def test_empty_preferences_list(self):
        """Preference data with empty preferences is valid."""
        meta = PreferenceDataMetadata(
            version="1.0",
            dataset_id="test",
            formatting_id="test",
            query_run_id="test_run",
            query_id="test",
            model="test",
            query_config={}
        )
        output = PreferenceDataOutput(metadata=meta, preferences=[])
        assert len(output.preferences) == 0

    def test_large_sample_id(self):
        """Large sample_id values are valid."""
        short = OptionOutput(label="a)", time=[1, "months"], reward=100.0)
        long = OptionOutput(label="b)", time=[1, "years"], reward=500.0)
        pair = PreferencePairOutput(short_term=short, long_term=long)

        q = QuestionOutput(
            sample_id=999999,
            question_text="Choose",
            time_horizon=None,
            preference_pair=pair
        )
        assert q.sample_id == 999999

    def test_very_long_question_text(self):
        """Very long question_text is valid."""
        short = OptionOutput(label="a)", time=[1, "months"], reward=100.0)
        long = OptionOutput(label="b)", time=[1, "years"], reward=500.0)
        pair = PreferencePairOutput(short_term=short, long_term=long)

        long_text = "Choose " * 1000  # Very long prompt
        q = QuestionOutput(
            sample_id=0,
            question_text=long_text,
            time_horizon=None,
            preference_pair=pair
        )
        assert len(q.question_text) > 5000
