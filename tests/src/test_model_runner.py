"""Tests for src/model_runner.py - Label token extraction."""

import pytest
from src.model_runner import extract_flip_tokens


class TestExtractFlipTokens:
    """Tests for extract_flip_tokens function."""

    def test_parenthesis_labels(self):
        """Labels like a) and b) should extract 'a' and 'b'."""
        labels = ("a)", "b)")
        flip1, flip2 = extract_flip_tokens(labels)
        assert flip1 == "a"
        assert flip2 == "b"

    def test_dot_labels(self):
        """Labels like A. and B. should extract 'A' and 'B'."""
        labels = ("A.", "B.")
        flip1, flip2 = extract_flip_tokens(labels)
        assert flip1 == "A"
        assert flip2 == "B"

    def test_numbered_parenthesis_labels(self):
        """Labels like (1) and (2) should extract '1' and '2'."""
        labels = ("(1)", "(2)")
        flip1, flip2 = extract_flip_tokens(labels)
        assert flip1 == "1"
        assert flip2 == "2"

    def test_bracket_labels(self):
        """Labels like [a] and [b] should extract 'a' and 'b'."""
        labels = ("[a]", "[b]")
        flip1, flip2 = extract_flip_tokens(labels)
        assert flip1 == "a"
        assert flip2 == "b"

    def test_colon_labels(self):
        """Labels like 1: and 2: should extract '1' and '2'."""
        labels = ("1:", "2:")
        flip1, flip2 = extract_flip_tokens(labels)
        assert flip1 == "1"
        assert flip2 == "2"

    def test_uppercase_parenthesis(self):
        """Labels like A) and B) should extract 'A' and 'B'."""
        labels = ("A)", "B)")
        flip1, flip2 = extract_flip_tokens(labels)
        assert flip1 == "A"
        assert flip2 == "B"

    def test_option_prefix_labels(self):
        """Labels like 'Option A:' and 'Option B:' should extract 'A' and 'B'."""
        labels = ("Option A:", "Option B:")
        flip1, flip2 = extract_flip_tokens(labels)
        assert flip1 == "A"
        assert flip2 == "B"

    def test_single_char_labels(self):
        """Single character labels should return themselves."""
        labels = ("a", "b")
        flip1, flip2 = extract_flip_tokens(labels)
        assert flip1 == "a"
        assert flip2 == "b"

    def test_multi_diff_chars(self):
        """When multiple chars differ, return the differing substring."""
        labels = ("Choice 1", "Choice 2")
        flip1, flip2 = extract_flip_tokens(labels)
        assert flip1 == "1"
        assert flip2 == "2"
