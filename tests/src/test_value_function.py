"""Tests for src/value_function.py - Value functions and choice prediction."""

import pytest
from src.choice_model.value_function import (
    LinearUtility,
    LogUtility,
    PowerUtility,
    ValueFunction,
    create_utility,
)
from src.choice_model.discount_function import ExponentialDiscount
from src.types import (
    DiscountFunctionParams,
    DiscountType,
    IntertemporalOption,
    PreferencePair,
    PreferenceQuestion,
    RewardValue,
    TimeValue,
    TrainingSample,
    UtilityType,
    ValueFunctionParams,
)


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_linear_utility(self):
        """Linear utility returns reward unchanged."""
        u = LinearUtility()
        assert u(100) == 100
        assert u(0) == 0
        assert u(-50) == -50

    def test_log_utility(self):
        """Log utility returns log of reward."""
        import math
        u = LogUtility()
        assert abs(u(math.e) - 1.0) < 1e-10
        assert u(1) == 0

    def test_log_utility_negative(self):
        """Log utility returns -inf for non-positive."""
        u = LogUtility()
        assert u(0) == float("-inf")
        assert u(-1) == float("-inf")

    def test_power_utility(self):
        """Power utility returns r^alpha."""
        u = PowerUtility(alpha=0.5)
        assert abs(u(4) - 2.0) < 1e-10
        assert abs(u(9) - 3.0) < 1e-10

    def test_create_utility_linear(self):
        """Factory creates linear utility."""
        u = create_utility(UtilityType.LINEAR)
        assert isinstance(u, LinearUtility)

    def test_create_utility_log(self):
        """Factory creates log utility."""
        u = create_utility(UtilityType.LOG)
        assert isinstance(u, LogUtility)


class TestValueFunction:
    """Tests for ValueFunction class."""

    @pytest.fixture
    def sample_pair(self):
        """Create a sample preference pair."""
        short = IntertemporalOption(
            label="a)",
            time=TimeValue(1, "months"),
            reward=RewardValue(100)
        )
        long = IntertemporalOption(
            label="b)",
            time=TimeValue(1, "years"),
            reward=RewardValue(500)
        )
        return PreferencePair(short_term=short, long_term=long)

    def test_value_immediate_reward(self):
        """Value at t=0 equals utility."""
        discount_params = DiscountFunctionParams(
            discount_type=DiscountType.EXPONENTIAL,
            theta=0.1,
        )
        params = ValueFunctionParams(
            utility_type=UtilityType.LINEAR,
            discount=discount_params,
        )
        vf = ValueFunction.from_params(params)
        opt = IntertemporalOption(
            label="a)",
            time=TimeValue(0, "years"),
            reward=RewardValue(100)
        )
        # At t=0, discount = 1, so value = utility(100) = 100
        assert vf.value(opt) == 100

    def test_predict_returns_valid_choice(self, sample_pair):
        """Predicts choice as short_term or long_term."""
        discount_params = DiscountFunctionParams(
            discount_type=DiscountType.EXPONENTIAL,
            theta=0.1,
        )
        params = ValueFunctionParams(
            utility_type=UtilityType.LINEAR,
            discount=discount_params,
        )
        vf = ValueFunction.from_params(params)
        prediction = vf.predict(sample_pair)
        assert prediction in ["short_term", "long_term"]

    def test_high_theta_prefers_short_term(self, sample_pair):
        """With very high theta, short-term is preferred."""
        discount_params = DiscountFunctionParams(
            discount_type=DiscountType.EXPONENTIAL,
            theta=100.0,
        )
        params = ValueFunctionParams(
            utility_type=UtilityType.LINEAR,
            discount=discount_params,
        )
        vf = ValueFunction.from_params(params)
        assert vf.predict(sample_pair) == "short_term"

    def test_low_theta_prefers_long_term(self, sample_pair):
        """With very low theta, long-term is preferred."""
        discount_params = DiscountFunctionParams(
            discount_type=DiscountType.EXPONENTIAL,
            theta=0.001,
        )
        params = ValueFunctionParams(
            utility_type=UtilityType.LINEAR,
            discount=discount_params,
        )
        vf = ValueFunction.from_params(params)
        assert vf.predict(sample_pair) == "long_term"


class TestValueFunctionTraining:
    """Tests for training value functions."""

    @pytest.fixture
    def training_samples(self):
        """Create sample training data."""
        samples = []
        # Create samples where choice clearly depends on theta
        for choice in ["short_term", "long_term"]:
            short = IntertemporalOption(
                label="a)",
                time=TimeValue(1, "months"),
                reward=RewardValue(100)
            )
            long = IntertemporalOption(
                label="b)",
                time=TimeValue(5, "years"),
                reward=RewardValue(1000)
            )
            pair = PreferencePair(short_term=short, long_term=long)
            question = PreferenceQuestion(pair=pair, time_horizon=None)
            samples.append(TrainingSample(question=question, chosen=choice))
        return samples

    def test_training_returns_result(self, training_samples):
        """Training returns a TrainingResult."""
        discount_params = DiscountFunctionParams(
            discount_type=DiscountType.EXPONENTIAL,
            theta=0.1,
        )
        params = ValueFunctionParams(
            utility_type=UtilityType.LINEAR,
            discount=discount_params,
        )
        vf = ValueFunction.from_params(params)
        result = vf.train(
            training_samples,
            learning_rate=0.01,
            num_iterations=10,
        )
        assert result is not None
        assert hasattr(result, "params")
        assert hasattr(result, "loss")
