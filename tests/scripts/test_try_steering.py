"""Tests for scripts/try_steering.py - Steering experiment utilities."""

import pytest
from unittest.mock import MagicMock, patch
from try_steering import build_prompt_from_question


class TestBuildPromptFromQuestion:
    """Tests for prompt building from questions.

    These tests prevent regression of a bug where label placeholders
    [LEFT_TERM_LABEL] and [RIGHT_TERM_LABEL] weren't being replaced.
    """

    @pytest.fixture
    def mock_question(self):
        """Create a mock question object."""
        question = MagicMock()
        question.question_text = "Which option do you prefer?\n[A] Short option\n[B] Long option\n"
        question.preference_pair.short_term.label = "[A]"
        question.preference_pair.long_term.label = "[B]"
        return question

    @pytest.fixture
    def mock_formatting_config(self):
        """Create a mock formatting config."""
        config = MagicMock()
        config.response_format = (
            "Respond in this format:\n"
            "[CHOICE_PREFIX] <[LEFT_TERM_LABEL] or [RIGHT_TERM_LABEL]>.\n"
            "[REASONING_PREFIX] <your reasoning in [MAX_REASONING_LENGTH]>"
        )
        config.choice_prefix = "I select:"
        config.reasoning_prefix = "My reasoning:"
        config.max_reasoning_length = "2-3 sentences"
        return config

    def test_replaces_left_term_label(self, mock_question, mock_formatting_config):
        """Ensures [LEFT_TERM_LABEL] is replaced with actual label."""
        with patch("src.dataset_generator.DatasetGenerator.load_formatting_config") as mock_load:
            mock_load.return_value = mock_formatting_config
            prompt = build_prompt_from_question(mock_question)

        assert "[LEFT_TERM_LABEL]" not in prompt
        assert "[A]" in prompt

    def test_replaces_right_term_label(self, mock_question, mock_formatting_config):
        """Ensures [RIGHT_TERM_LABEL] is replaced with actual label."""
        with patch("src.dataset_generator.DatasetGenerator.load_formatting_config") as mock_load:
            mock_load.return_value = mock_formatting_config
            prompt = build_prompt_from_question(mock_question)

        assert "[RIGHT_TERM_LABEL]" not in prompt
        assert "[B]" in prompt

    def test_replaces_choice_prefix(self, mock_question, mock_formatting_config):
        """Ensures [CHOICE_PREFIX] is replaced."""
        with patch("src.dataset_generator.DatasetGenerator.load_formatting_config") as mock_load:
            mock_load.return_value = mock_formatting_config
            prompt = build_prompt_from_question(mock_question)

        assert "[CHOICE_PREFIX]" not in prompt
        assert "I select:" in prompt

    def test_replaces_reasoning_prefix(self, mock_question, mock_formatting_config):
        """Ensures [REASONING_PREFIX] is replaced."""
        with patch("src.dataset_generator.DatasetGenerator.load_formatting_config") as mock_load:
            mock_load.return_value = mock_formatting_config
            prompt = build_prompt_from_question(mock_question)

        assert "[REASONING_PREFIX]" not in prompt
        assert "My reasoning:" in prompt

    def test_prompt_includes_question_text(self, mock_question, mock_formatting_config):
        """Ensures prompt includes the original question text."""
        with patch("src.dataset_generator.DatasetGenerator.load_formatting_config") as mock_load:
            mock_load.return_value = mock_formatting_config
            prompt = build_prompt_from_question(mock_question)

        assert "Which option do you prefer?" in prompt

    def test_full_prompt_format(self, mock_question, mock_formatting_config):
        """Tests the complete prompt format with all substitutions."""
        with patch("src.dataset_generator.DatasetGenerator.load_formatting_config") as mock_load:
            mock_load.return_value = mock_formatting_config
            prompt = build_prompt_from_question(mock_question)

        # Should contain the expected format
        assert "I select: <[A] or [B]>" in prompt
        assert "My reasoning:" in prompt
