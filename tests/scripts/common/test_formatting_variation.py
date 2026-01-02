"""Tests for scripts/common/formatting_variation.py - Formatting variations."""

import pytest
from common.formatting_variation import (
    LABEL_STYLES,
    TIME_UNITS,
    FormattingVariation,
    convert_time_value,
    format_time_spelled,
    get_all_label_styles,
    get_random_labels,
    get_sensible_units_for_time,
    spell_number,
)
from src.types import TimeValue


class TestLabelStyles:
    """Tests for label style utilities."""

    def test_all_label_styles_distinct(self):
        """All label pairs have distinct left/right labels."""
        for left, right in LABEL_STYLES:
            assert left != right, f"Same label in pair: ({left}, {right})"

    def test_get_random_labels_returns_valid_pair(self):
        """get_random_labels returns a valid label pair."""
        labels = get_random_labels()
        assert isinstance(labels, tuple)
        assert len(labels) == 2
        assert labels[0] != labels[1]

    def test_get_all_label_styles_returns_copy(self):
        """get_all_label_styles returns a copy, not original."""
        styles1 = get_all_label_styles()
        styles2 = get_all_label_styles()
        assert styles1 is not styles2


class TestTimeUnitConversion:
    """Tests for time unit conversion."""

    def test_convert_years_to_months(self):
        """1 year = 12 months."""
        tv = TimeValue(1, "years")
        result = convert_time_value(tv, "months")
        assert result.unit == "months"
        assert result.value == 12

    def test_convert_years_to_days(self):
        """1 year ~ 365 days."""
        tv = TimeValue(1, "years")
        result = convert_time_value(tv, "days")
        assert result.unit == "days"
        # Allow for leap year adjustment
        assert abs(result.value - 365.25) < 1

    def test_convert_months_to_weeks(self):
        """1 month ~ 4.35 weeks."""
        tv = TimeValue(1, "months")
        result = convert_time_value(tv, "weeks")
        assert result.unit == "weeks"
        assert abs(result.value - 4.35) < 0.1

    def test_convert_decades_to_years(self):
        """1 decade = 10 years."""
        tv = TimeValue(10, "years")  # Use years since TimeValue.to_years() knows years
        result = convert_time_value(tv, "decades")
        assert result.unit == "decades"
        assert result.value == 1.0

    def test_get_sensible_units(self):
        """Sensible units are returned for various time values."""
        # 1 year should include months, years
        tv = TimeValue(1, "years")
        units = get_sensible_units_for_time(tv)
        assert "years" in units
        assert "months" in units


class TestSpellNumber:
    """Tests for number spelling."""

    def test_spell_integers(self):
        """Can spell common integers."""
        assert spell_number(1) == "one"
        assert spell_number(2) == "two"
        assert spell_number(5) == "five"
        assert spell_number(10) == "ten"
        assert spell_number(12) == "twelve"

    def test_spell_half(self):
        """Can spell 0.5 as 'half a'."""
        assert spell_number(0.5) == "half a"

    def test_spell_quarter(self):
        """Can spell 0.25 as 'a quarter of a'."""
        assert spell_number(0.25) == "a quarter of a"

    def test_spell_large_number(self):
        """Can spell larger integers."""
        assert spell_number(100) == "one hundred"
        assert spell_number(25) == "twenty-five"

    def test_spell_unsupported_returns_none(self):
        """Unsupported numbers return None."""
        assert spell_number(0.33) is None
        assert spell_number(123) is None


class TestFormatTimeSpelled:
    """Tests for spelled time formatting."""

    def test_format_one_year(self):
        """1 year -> 'one year'."""
        tv = TimeValue(1, "years")
        result = format_time_spelled(tv)
        assert result == "one year"

    def test_format_two_months(self):
        """2 months -> 'two months'."""
        tv = TimeValue(2, "months")
        result = format_time_spelled(tv)
        assert result == "two months"

    def test_format_half_decade(self):
        """0.5 decades -> 'half a decade'."""
        tv = TimeValue(0.5, "decades")
        result = format_time_spelled(tv)
        assert result == "half a decade"

    def test_format_unsupported_returns_none(self):
        """Unsupported values return None."""
        tv = TimeValue(123, "years")
        result = format_time_spelled(tv)
        assert result is None


class TestFormattingVariation:
    """Tests for FormattingVariation dataclass."""

    def test_default_variation(self):
        """Default variation has no changes."""
        v = FormattingVariation.default()
        assert v.labels == ("a)", "b)")
        assert v.flip_order is False
        assert v.time_unit_variation is False
        assert v.spell_numbers is False

    def test_random_variation(self):
        """Random variation produces valid settings."""
        v = FormattingVariation.random()
        assert isinstance(v.labels, tuple)
        assert len(v.labels) == 2
        assert isinstance(v.flip_order, bool)
        assert isinstance(v.time_unit_variation, bool)
        assert isinstance(v.spell_numbers, bool)

    def test_random_variations_differ(self):
        """Multiple random variations are likely different."""
        variations = [FormattingVariation.random() for _ in range(10)]
        # At least some should differ
        labels_seen = set(v.labels for v in variations)
        assert len(labels_seen) > 1  # Not all the same
