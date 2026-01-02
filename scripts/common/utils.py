"""
Shared utility functions for scripts.

Contains helper functions for response parsing and formatting.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.schemas import FormattingConfig


# =============================================================================
# Model Quirks Configuration
# =============================================================================


@dataclass
class ModelQuirks:
    """
    Model-specific configuration for handling quirks in prompting and parsing.

    Attributes:
        prompt_suffix: Additional text to append to prompts for this model
        strip_markdown: Whether to strip markdown formatting from responses
        custom_patterns: Additional regex patterns to try for label extraction
    """

    prompt_suffix: str = ""
    strip_markdown: bool = False
    custom_patterns: list[str] = field(default_factory=list)


# Model quirks registry - maps model name patterns to their quirks
MODEL_QUIRKS: dict[str, ModelQuirks] = {
    "gemma": ModelQuirks(
        prompt_suffix="\nIMPORTANT: Do not use markdown formatting (no ** or __ around text).",
        strip_markdown=True,
    ),
}


def get_model_quirks(model_name: str) -> ModelQuirks:
    """
    Get quirks configuration for a model.

    Args:
        model_name: Full or short model name

    Returns:
        ModelQuirks for the model, or default if no specific quirks
    """
    model_lower = model_name.lower()
    for pattern, quirks in MODEL_QUIRKS.items():
        if pattern in model_lower:
            return quirks
    return ModelQuirks()


# =============================================================================
# Response Format Utilities
# =============================================================================


def format_response_format(
    formatting_config: FormattingConfig,
    labels: tuple[str, str],
    model_name: Optional[str] = None,
) -> str:
    """
    Format the response_format template with labels.

    Args:
        formatting_config: Formatting configuration
        labels: (left_label, right_label) tuple
        model_name: Optional model name for model-specific formatting

    Returns:
        Formatted response format string
    """
    response_fmt = formatting_config.response_format
    response_fmt = response_fmt.replace("[LEFT_TERM_LABEL]", labels[0])
    response_fmt = response_fmt.replace("[RIGHT_TERM_LABEL]", labels[1])
    response_fmt = response_fmt.replace("[CHOICE_PREFIX]", formatting_config.choice_prefix)
    response_fmt = response_fmt.replace("[REASONING_PREFIX]", formatting_config.reasoning_prefix)
    response_fmt = response_fmt.replace(
        "[MAX_REASONING_LENGTH]", formatting_config.max_reasoning_length
    )

    # Apply model-specific prompt additions
    if model_name:
        quirks = get_model_quirks(model_name)
        if quirks.prompt_suffix:
            response_fmt += quirks.prompt_suffix

    # Validate no unreplaced placeholders remain
    validate_no_unreplaced_placeholders(response_fmt, "response_format")

    return response_fmt


def validate_no_unreplaced_placeholders(text: str, context: str = "") -> None:
    """
    Validate that no [PLACEHOLDER] patterns remain in text.

    Args:
        text: Text to check
        context: Description of where this text came from (for error messages)

    Raises:
        ValueError: If unreplaced placeholders are found
    """
    import re
    # Find [WORD] patterns that look like placeholders:
    # - Must contain underscore OR be longer than 2 chars
    # - This excludes labels like [A], [B], [1], [2] which are intentional
    all_brackets = re.findall(r'\[[A-Z][A-Z0-9_]*\]', text)
    placeholders = [p for p in all_brackets if '_' in p or len(p) > 4]  # [XX] = 4 chars
    if placeholders:
        unique = sorted(set(placeholders))
        ctx = f" in {context}" if context else ""
        raise ValueError(
            f"Unreplaced placeholders found{ctx}: {', '.join(unique)}\n"
            f"Text snippet: {text[:200]}..."
        )


# =============================================================================
# Response Parsing Utilities
# =============================================================================


def strip_markdown(text: str) -> str:
    """
    Strip common markdown formatting from text.

    Handles: **bold**, __bold__, *italic*, _italic_

    Args:
        text: Text potentially containing markdown

    Returns:
        Text with markdown formatting removed
    """
    # Strip bold (**text** or __text__)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    # Strip italic (*text* or _text_) - be careful not to strip underscores in words
    text = re.sub(r"(?<!\w)\*([^*]+)\*(?!\w)", r"\1", text)
    text = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"\1", text)
    return text


def _strip_label_punctuation(label: str) -> str:
    """Strip leading and trailing punctuation from label for flexible matching."""
    # Strip leading brackets and trailing punctuation
    # e.g., "[A]" -> "a", "a)" -> "a", "OPTION:" -> "option"
    return label.lstrip("[(").rstrip(":.)],")


def parse_label_from_response(
    text: str,
    labels: list[str],
    choice_prefix: str,
    model_name: Optional[str] = None,
) -> Optional[str]:
    """
    Parse the chosen label from model response.

    Args:
        text: Model response text
        labels: List of valid labels (e.g., ["a)", "b)"])
        choice_prefix: The prefix pattern to look for (e.g., "I choose:")
        model_name: Optional model name for model-specific parsing

    Returns:
        The matched label (lowercased) or None if not found
    """
    # Apply model-specific preprocessing
    if model_name:
        quirks = get_model_quirks(model_name)
        if quirks.strip_markdown:
            text = strip_markdown(text)

    text_lower = text.lower()
    choice_prefix_lower = choice_prefix.lower().rstrip(":")

    # Build regex patterns from labels - try both exact and stripped versions
    # This handles cases like label="SECOND:" but response says "SECOND."
    labels_lower = [label.lower() for label in labels]
    labels_stripped = [_strip_label_punctuation(label.lower()) for label in labels]

    # Build pattern that matches either exact labels OR stripped labels
    # e.g., for labels ["[A]", "[B]"], match "[a]", "[b]", "a", or "b"
    all_variants = set(labels_lower) | set(labels_stripped)
    labels_pattern = "|".join(re.escape(l) for l in sorted(all_variants, key=len, reverse=True))

    # Look for explicit "<choice_prefix> <label>" pattern
    # Match label (with or without brackets) followed by optional punctuation
    match = re.search(
        rf"{re.escape(choice_prefix_lower)}[:\s]+({labels_pattern})[\.:)\],]?",
        text_lower
    )
    if match:
        matched = match.group(1)
        # Return the original label (lowercased) that matches this version
        # First check exact match, then stripped match
        for orig_label in labels_lower:
            if matched == orig_label:
                return orig_label
        for orig_label, stripped in zip(labels_lower, labels_stripped):
            if matched == stripped:
                return orig_label
        return matched

    # Look for "option <label>" or "choice <label>"
    for orig_label, stripped in zip(labels_lower, labels_stripped):
        if f"option {stripped}" in text_lower or f"choice {stripped}" in text_lower:
            return orig_label

    # Look for just "<label>)" or "<label>." at the start
    for orig_label, stripped in zip(labels_lower, labels_stripped):
        text_start = text_lower.strip()
        if text_start.startswith(f"{stripped})") or text_start.startswith(f"{stripped}."):
            return orig_label

    return None


def determine_choice(
    chosen_label: Optional[str],
    short_term_label: str,
    long_term_label: str,
) -> str:
    """
    Determine if chosen label corresponds to short_term or long_term.

    Args:
        chosen_label: The label parsed from response
        short_term_label: Label assigned to short_term option
        long_term_label: Label assigned to long_term option

    Returns:
        "short_term", "long_term", or "unknown"
    """
    if chosen_label is None:
        return "unknown"

    chosen_lower = chosen_label.lower()
    if chosen_lower == short_term_label.lower():
        return "short_term"
    elif chosen_lower == long_term_label.lower():
        return "long_term"
    else:
        return "unknown"


def extract_flip_tokens(labels: tuple[str, str]) -> tuple[str, str]:
    """
    Extract the distinguishing "flip" tokens from two labels.

    Given label pairs like ("a)", "b)") or ("A.", "B.") or ("(1)", "(2)"),
    extract the character(s) that differ between them.

    Args:
        labels: Tuple of two label strings

    Returns:
        Tuple of (flip1, flip2) - the distinguishing tokens for each label
    """
    label1, label2 = labels

    # Find differing character positions
    min_len = min(len(label1), len(label2))
    diff_start = None
    diff_end = None

    # Find first differing position
    for i in range(min_len):
        if label1[i] != label2[i]:
            diff_start = i
            break

    if diff_start is None:
        # Labels are identical up to min_len, difference is in length
        if len(label1) != len(label2):
            # Return the extra part
            if len(label1) > len(label2):
                return label1[min_len:], ""
            else:
                return "", label2[min_len:]
        # Labels are identical
        return label1, label2

    # Find last differing position (from the end)
    for i in range(1, min_len + 1):
        if label1[-i] != label2[-i]:
            diff_end = len(label1) - i + 1
            break

    if diff_end is None:
        diff_end = len(label1)

    # Handle case where labels have different lengths
    if len(label1) != len(label2):
        # Adjust diff_end based on longer label
        len_diff = abs(len(label1) - len(label2))
        if len(label1) > len(label2):
            diff_end1 = diff_end
            diff_end2 = diff_end - len_diff
        else:
            diff_end1 = diff_end - len_diff
            diff_end2 = diff_end
        flip1 = label1[diff_start:diff_end1] if diff_end1 > diff_start else label1[diff_start]
        flip2 = label2[diff_start:diff_end2] if diff_end2 > diff_start else label2[diff_start]
    else:
        flip1 = label1[diff_start:diff_end]
        flip2 = label2[diff_start:diff_end]

    return flip1, flip2


# =============================================================================
# Memory Management
# =============================================================================


def clear_memory() -> None:
    """
    Clear GPU/CPU memory caches to prevent memory leaks.

    This function is useful after running model inference, especially when
    processing many samples. It clears:
    - Python garbage collector
    - CUDA memory cache (if available)
    - MPS (Apple Silicon) memory cache (if available)

    Usage:
        # After processing a batch of samples
        for sample in samples:
            result = model.generate(sample)
            process(result)

        clear_memory()  # Free memory before next batch
    """
    import gc

    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except ImportError:
        pass  # torch not installed
