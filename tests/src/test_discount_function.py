"""Tests for src/discount_function.py - Discount functions."""

import math
import pytest
from src.choice_model.discount_function import (
    ExponentialDiscount,
    HyperbolicDiscount,
    QuasiHyperbolicDiscount,
    DiscountFunction,
)
from src.types import TimeValue


class TestExponentialDiscount:
    """Tests for exponential discount function D(t) = exp(-theta * t)."""

    def test_discount_at_zero(self):
        """Discount at t=0 should be 1."""
        df = ExponentialDiscount(theta=0.1)
        assert df(0) == 1.0

    def test_discount_decreases_with_time(self):
        """Discount should decrease as time increases."""
        df = ExponentialDiscount(theta=0.1)
        assert df(1) < df(0)
        assert df(5) < df(1)
        assert df(10) < df(5)

    def test_higher_theta_more_discounting(self):
        """Higher theta means more discounting."""
        low_theta = ExponentialDiscount(theta=0.05)
        high_theta = ExponentialDiscount(theta=0.5)
        t = 5.0
        assert high_theta(t) < low_theta(t)

    def test_known_value(self):
        """Test against known exponential value."""
        df = ExponentialDiscount(theta=1.0)
        expected = math.exp(-1.0)
        assert abs(df(1.0) - expected) < 1e-10

    def test_discount_with_time_value(self):
        """Test discount method with TimeValue."""
        df = ExponentialDiscount(theta=0.1)
        tv = TimeValue(12, "months")  # 1 year
        assert abs(df.discount(tv) - df(1.0)) < 1e-10


class TestHyperbolicDiscount:
    """Tests for hyperbolic discount function D(t) = 1 / (1 + theta * t)."""

    def test_discount_at_zero(self):
        """Discount at t=0 should be 1."""
        df = HyperbolicDiscount(theta=0.5)
        assert df(0) == 1.0

    def test_discount_decreases_with_time(self):
        """Discount should decrease as time increases."""
        df = HyperbolicDiscount(theta=0.5)
        assert df(1) < df(0)
        assert df(5) < df(1)

    def test_known_value(self):
        """Test against known hyperbolic value."""
        df = HyperbolicDiscount(theta=1.0)
        expected = 1 / (1 + 1.0)  # 0.5
        assert abs(df(1.0) - expected) < 1e-10


class TestQuasiHyperbolicDiscount:
    """Tests for quasi-hyperbolic discount function."""

    def test_discount_at_zero(self):
        """Discount at t=0 should be 1."""
        df = QuasiHyperbolicDiscount(beta=0.7, delta=0.95)
        assert df(0) == 1.0

    def test_immediate_drop(self):
        """There's an immediate drop by factor beta for t > 0."""
        df = QuasiHyperbolicDiscount(beta=0.7, delta=0.99)
        # At small t, discount should be close to beta
        assert abs(df(0.01) - 0.7 * (0.99 ** 0.01)) < 0.01
