"""Tests for scripts/common/utils.py - Response parsing utilities."""

import pytest
from common.utils import (
    parse_label_from_response,
    determine_choice,
    strip_markdown,
    format_response_format,
    _strip_label_punctuation,
)


class TestParseLabel:
    """Tests for label parsing from responses."""

    def test_parse_explicit_choice_prefix(self):
        """Parses 'I choose: a)' format."""
        text = "I choose: a). This is because..."
        result = parse_label_from_response(text, ["a)", "b)"], "I choose:")
        assert result == "a)"

    def test_parse_option_prefix(self):
        """Parses 'option a)' format."""
        text = "I think option a) is better because..."
        result = parse_label_from_response(text, ["a)", "b)"], "I choose:")
        assert result == "a)"

    def test_parse_label_at_start(self):
        """Parses label at start of response."""
        text = "a). The short-term option..."
        result = parse_label_from_response(text, ["a)", "b)"], "I choose:")
        assert result == "a)"

    def test_parse_returns_none_for_no_match(self):
        """Returns None when no label found."""
        text = "I don't know which to choose."
        result = parse_label_from_response(text, ["a)", "b)"], "I choose:")
        assert result is None

    def test_parse_case_insensitive(self):
        """Parsing is case insensitive."""
        text = "I CHOOSE: A)"
        result = parse_label_from_response(text, ["a)", "b)"], "I choose:")
        assert result == "a)"

    def test_parse_alternative_labels(self):
        """Parses alternative label styles."""
        text = "I select: Option 1:. The reasoning..."
        result = parse_label_from_response(
            text, ["Option 1:", "Option 2:"], "I select:"
        )
        assert result == "option 1:"

    def test_parse_x_y_labels(self):
        """Parses x) y) label style."""
        text = "I choose: y). Because the long-term..."
        result = parse_label_from_response(text, ["x)", "y)"], "I choose:")
        assert result == "y)"


class TestDetermineChoice:
    """Tests for choice determination from parsed labels."""

    def test_short_term_match(self):
        """Returns 'short_term' when label matches."""
        result = determine_choice("a)", "a)", "b)")
        assert result == "short_term"

    def test_long_term_match(self):
        """Returns 'long_term' when label matches."""
        result = determine_choice("b)", "a)", "b)")
        assert result == "long_term"

    def test_unknown_for_none(self):
        """Returns 'unknown' for None label."""
        result = determine_choice(None, "a)", "b)")
        assert result == "unknown"

    def test_unknown_for_no_match(self):
        """Returns 'unknown' when no label matches."""
        result = determine_choice("c)", "a)", "b)")
        assert result == "unknown"

    def test_case_insensitive_matching(self):
        """Matching is case insensitive."""
        result = determine_choice("A)", "a)", "b)")
        assert result == "short_term"


class TestStripMarkdown:
    """Tests for markdown stripping."""

    def test_strip_bold_asterisks(self):
        """Strips **bold** formatting."""
        text = "I choose **a)**"
        result = strip_markdown(text)
        assert result == "I choose a)"

    def test_strip_bold_underscores(self):
        """Strips __bold__ formatting."""
        text = "I choose __a)__"
        result = strip_markdown(text)
        assert result == "I choose a)"

    def test_strip_italic(self):
        """Strips *italic* formatting."""
        text = "I choose *a)*"
        result = strip_markdown(text)
        assert result == "I choose a)"

    def test_preserves_plain_text(self):
        """Leaves plain text unchanged."""
        text = "I choose a)"
        result = strip_markdown(text)
        assert result == text


class TestStripLabelPunctuation:
    """Tests for label punctuation stripping."""

    def test_strip_square_brackets(self):
        """Strips square brackets from labels like [A]."""
        assert _strip_label_punctuation("[A]") == "A"
        assert _strip_label_punctuation("[B]") == "B"
        assert _strip_label_punctuation("[1]") == "1"
        assert _strip_label_punctuation("[2]") == "2"

    def test_strip_parentheses(self):
        """Strips parentheses from labels like (1)."""
        assert _strip_label_punctuation("(1)") == "1"
        assert _strip_label_punctuation("(2)") == "2"
        assert _strip_label_punctuation("(A)") == "A"

    def test_strip_trailing_punctuation(self):
        """Strips trailing punctuation."""
        assert _strip_label_punctuation("a)") == "a"
        assert _strip_label_punctuation("Option 1:") == "Option 1"
        assert _strip_label_punctuation("FIRST.") == "FIRST"

    def test_strip_roman_numerals(self):
        """Handles roman numeral labels."""
        assert _strip_label_punctuation("[I]") == "I"
        assert _strip_label_punctuation("[II]") == "II"
        assert _strip_label_punctuation("[III]") == "III"

    def test_preserves_core_content(self):
        """Preserves the core label content."""
        assert _strip_label_punctuation("Option A") == "Option A"
        assert _strip_label_punctuation("SECOND") == "SECOND"


class TestParseLabelWithBrackets:
    """Tests for parsing labels with various bracket styles.

    These tests prevent regression of a bug where labels like [A], [1], (2)
    weren't matched because the regex only looked for stripped versions.
    """

    def test_parse_square_bracket_labels(self):
        """Parses [A] and [B] style labels."""
        text = "I select: [A]. This is my choice."
        result = parse_label_from_response(text, ["[A]", "[B]"], "I select:")
        assert result == "[a]"

    def test_parse_numeric_bracket_labels(self):
        """Parses [1] and [2] style labels."""
        text = "I select: [1] 24,662 housing units in ten years."
        result = parse_label_from_response(text, ["[2]", "[1]"], "I select:")
        assert result == "[1]"

    def test_parse_parenthesis_labels(self):
        """Parses (1) and (2) style labels."""
        text = "I select: (2) 2,000 housing units in 1 year."
        result = parse_label_from_response(text, ["(2)", "(1)"], "I select:")
        assert result == "(2)"

    def test_parse_roman_numeral_labels(self):
        """Parses [I] and [II] style labels."""
        text = "I select: [II]. This option is better."
        result = parse_label_from_response(text, ["[II]", "[I]"], "I select:")
        assert result == "[ii]"

    def test_parse_label_without_brackets_matches_bracketed(self):
        """Model outputs A. but labels are [A], [B] - should still match."""
        text = "I select: A. 10,000 housing units"
        result = parse_label_from_response(text, ["[A]", "[B]"], "I select:")
        assert result == "[a]"

    def test_parse_label_with_brackets_different_format(self):
        """Model outputs [A]. but labels are A), B) - doesn't match (different formats).

        Note: The parser matches labels against text, not the reverse. If the prompt
        uses A)/B) labels but model outputs [A], they won't match. This is expected
        since prompts should use consistent label formats.
        """
        text = "I select: [A]. This is my choice."
        result = parse_label_from_response(text, ["A)", "B)"], "I select:")
        # This returns None because [A] in text doesn't match A) or a patterns
        assert result is None


class TestDetermineChoiceWithBrackets:
    """Tests for choice determination with bracketed labels."""

    def test_bracketed_labels_short_term(self):
        """Returns 'short_term' for bracketed label match."""
        result = determine_choice("[a]", "[A]", "[B]")
        assert result == "short_term"

    def test_bracketed_labels_long_term(self):
        """Returns 'long_term' for bracketed label match."""
        result = determine_choice("[b]", "[A]", "[B]")
        assert result == "long_term"

    def test_numeric_bracketed_labels(self):
        """Handles numeric labels like [1], [2]."""
        result = determine_choice("[1]", "[1]", "[2]")
        assert result == "short_term"
        result = determine_choice("[2]", "[1]", "[2]")
        assert result == "long_term"
