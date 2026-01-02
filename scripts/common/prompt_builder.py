"""
Prompt building utilities for LLM experiments.

This module provides a single interface for constructing prompts from questions
and formatting configurations. It consolidates the prompt-building logic that
was previously duplicated across multiple scripts.

The key function is `build_prompt_from_question()` which combines:
- Question text (from dataset generation)
- Response format (from formatting config)
- Label substitution (short_term/long_term labels)
- Model-specific adjustments (from model quirks)

Usage:
    from common.prompt_builder import build_prompt_from_question

    prompt = build_prompt_from_question(
        question=question,
        formatting_config=formatting_config,
        model_name="Qwen/Qwen2.5-7B-Instruct",
    )
"""

from pathlib import Path
from typing import Optional

from .schemas import QuestionOutput
from .utils import format_response_format
from src.dataset_generator import FormattingConfig


def build_prompt_from_question(
    question: QuestionOutput,
    formatting_config: Optional[FormattingConfig] = None,
    formatting_config_path: Optional[Path] = None,
    model_name: Optional[str] = None,
) -> str:
    """
    Build a complete prompt from a question and formatting configuration.

    This is the primary interface for constructing prompts. It combines:
    1. The question text (containing the preference pair options)
    2. The response format (with placeholders replaced)

    Args:
        question: The question to build a prompt for (contains preference pair)
        formatting_config: Pre-loaded formatting configuration (optional)
        formatting_config_path: Path to formatting config JSON (used if formatting_config not provided)
        model_name: Optional model name for model-specific prompt adjustments

    Returns:
        Complete prompt string ready for LLM input

    Example:
        >>> prompt = build_prompt_from_question(question, formatting_config=config)
        >>> # Returns: "You are... [A] $100 in 1 month [B] $500 in 1 year\\nI select: <[A] or [B]>..."

    Note:
        Either formatting_config or formatting_config_path must be provided.
        If neither is provided, the default formatting config is loaded.
    """
    # Load formatting config if not provided
    if formatting_config is None:
        from src.dataset_generator import DatasetGenerator

        if formatting_config_path is None:
            # Use default formatting config
            scripts_dir = Path(__file__).parent.parent
            formatting_config_path = (
                scripts_dir / "configs" / "formatting" / "default_formatting.json"
            )

        formatting_config = DatasetGenerator.load_formatting_config(formatting_config_path)

    # Extract labels from the question
    labels = (
        question.preference_pair.short_term.label,
        question.preference_pair.long_term.label,
    )

    # Use the shared format_response_format function which handles all substitutions
    response_format_str = format_response_format(
        formatting_config=formatting_config,
        labels=labels,
        model_name=model_name,
    )

    # Combine question text with formatted response instructions
    return question.question_text + response_format_str


def build_time_horizon_marker_text(
    time_horizon: Optional[list],
    formatting_config: FormattingConfig,
) -> Optional[str]:
    """
    Build time horizon specification text for marker injection.

    When querying models, we inject a time_horizon_spec marker to indicate
    the decision horizon. This function formats that marker text.

    Args:
        time_horizon: Time horizon as [value, unit] list, or None for no horizon
        formatting_config: Formatting configuration with time_horizon_spec template

    Returns:
        Formatted time horizon text, or None if no horizon specified

    Example:
        >>> text = build_time_horizon_marker_text([5, "years"], config)
        >>> # Returns: "You are primarily concerned about outcome in 5 years."
    """
    if time_horizon is None:
        return None

    if not formatting_config.time_horizon_spec:
        return None

    value, unit = time_horizon
    time_str = f"{value} {unit}"
    return formatting_config.time_horizon_spec.replace("[TIME_HORIZON]", time_str)


__all__ = [
    "build_prompt_from_question",
    "build_time_horizon_marker_text",
]
