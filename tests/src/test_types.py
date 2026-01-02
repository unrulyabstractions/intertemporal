"""Tests for src/types.py - Core type definitions."""

import pytest
from src.types import TimeValue, RewardValue, IntertemporalOption, PreferencePair


class TestTimeValue:
    """Tests for TimeValue class."""

    def test_to_years_from_years(self):
        """Years converts to years correctly."""
        tv = TimeValue(2.5, "years")
        assert tv.to_years() == 2.5

    def test_to_years_from_months(self):
        """Months convert to years correctly."""
        tv = TimeValue(12, "months")
        assert tv.to_years() == 1.0

    def test_to_years_from_days(self):
        """Days convert to years correctly."""
        tv = TimeValue(30, "days")  # 1 month
        # 30 days = 1 month = 1/12 year
        assert abs(tv.to_years() - 1/12) < 0.01

    def test_to_list(self):
        """to_list returns correct format."""
        tv = TimeValue(3, "months")
        assert tv.to_list() == [3, "months"]

    def test_str_representation(self):
        """String representation is readable."""
        tv = TimeValue(6, "months")
        assert "6" in str(tv)
        assert "months" in str(tv)

    def test_singular_unit_for_one(self):
        """Unit is singular for value 1."""
        tv = TimeValue(1, "years")
        assert "year" in str(tv)
        assert "years" not in str(tv)


class TestRewardValue:
    """Tests for RewardValue class."""

    def test_creation(self):
        """RewardValue stores value and unit."""
        rv = RewardValue(100, "dollars")
        assert rv.value == 100
        assert rv.unit == "dollars"

    def test_default_unit(self):
        """Default unit is empty string."""
        rv = RewardValue(50)
        assert rv.unit == ""

    def test_str_with_unit(self):
        """String representation includes unit."""
        rv = RewardValue(1000, "dollars")
        assert "1,000" in str(rv)
        assert "dollars" in str(rv)


class TestIntertemporalOption:
    """Tests for IntertemporalOption class."""

    def test_creation(self):
        """IntertemporalOption stores all fields."""
        opt = IntertemporalOption(
            label="a)",
            time=TimeValue(3, "months"),
            reward=RewardValue(100, "dollars")
        )
        assert opt.label == "a)"
        assert opt.time.value == 3
        assert opt.reward.value == 100


class TestPreferencePair:
    """Tests for PreferencePair class."""

    def test_creation(self):
        """PreferencePair stores short and long term options."""
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
        pair = PreferencePair(short_term=short, long_term=long)
        assert pair.short_term.reward.value == 100
        assert pair.long_term.reward.value == 500
