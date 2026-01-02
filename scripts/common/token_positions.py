"""
Token position resolution for internals capture.

Resolves TokenPosition specs to actual token indices given prompt and continuation tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.schemas import TokenPosition, TokenLocation


@dataclass
class ResolvedPosition:
    """A resolved token position with metadata."""
    index: int  # Absolute index into full sequence
    token: str  # Token string at this position
    source: str  # Description of how it was resolved
    is_prompt: bool  # Whether this is in the prompt


@dataclass
class TokenSequenceInfo:
    """Information about tokenized prompt and continuation."""
    prompt_tokens: list[str]
    continuation_tokens: list[str]
    prompt_len: int  # Number of prompt tokens
    total_len: int  # Total sequence length
    time_horizon_spec_end: int = -1  # Token index where time_horizon_spec ends (-1 if not present)

    @property
    def all_tokens(self) -> list[str]:
        """Get all tokens (prompt + continuation)."""
        return self.prompt_tokens + self.continuation_tokens


def find_text_in_tokens(
    text: str,
    tokens: list[str],
    offset: int = 0,
) -> Optional[int]:
    """
    Find the token position where text ends.

    Args:
        text: Text to search for
        tokens: List of token strings
        offset: Offset to add to returned index

    Returns:
        Token index (with offset) where text ends, or None if not found
    """
    accumulated = ""
    for i, tok in enumerate(tokens):
        accumulated += tok
        # Check if text appears and we're at or past its end
        if text in accumulated:
            # Find where text ends in accumulated
            text_end = accumulated.find(text) + len(text)
            if len(accumulated.rstrip()) >= text_end:
                return i + offset

    return None


def resolve_token_position(
    pos: TokenPosition,
    seq_info: TokenSequenceInfo,
) -> Optional[ResolvedPosition]:
    """
    Resolve a TokenPosition spec to an actual token index.

    Args:
        pos: TokenPosition specification
        seq_info: Token sequence information

    Returns:
        ResolvedPosition with index and metadata, or None if resolution failed
    """
    # Text-based search
    if pos.is_text_search():
        if pos.is_prompt_position():
            # Search in prompt
            search_tokens = seq_info.prompt_tokens
            search_offset = 0

            # If after_time_horizon_spec, start searching after it
            if pos.after_time_horizon_spec and seq_info.time_horizon_spec_end >= 0:
                # Only search tokens after time_horizon_spec_end
                start_idx = seq_info.time_horizon_spec_end + 1
                if start_idx < len(seq_info.prompt_tokens):
                    search_tokens = seq_info.prompt_tokens[start_idx:]
                    search_offset = start_idx
                else:
                    return None  # time_horizon_spec is at end of prompt

            idx = find_text_in_tokens(pos.text, search_tokens, offset=search_offset)
            if idx is not None:
                return ResolvedPosition(
                    index=idx,
                    token=seq_info.prompt_tokens[idx] if idx < len(seq_info.prompt_tokens) else "<OOB>",
                    source=f"text '{pos.text}' in prompt" + (" (after time_horizon_spec)" if pos.after_time_horizon_spec else ""),
                    is_prompt=True,
                )
        else:
            # Search in continuation (default)
            idx = find_text_in_tokens(
                pos.text,
                seq_info.continuation_tokens,
                offset=seq_info.prompt_len,
            )
            if idx is not None:
                cont_idx = idx - seq_info.prompt_len
                return ResolvedPosition(
                    index=idx,
                    token=seq_info.continuation_tokens[cont_idx] if cont_idx < len(seq_info.continuation_tokens) else "<OOB>",
                    source=f"text '{pos.text}' in continuation",
                    is_prompt=False,
                )
        return None  # Text not found

    # Index-based positions
    if pos.prompt_index is not None:
        # Index into prompt
        idx = pos.prompt_index
        if idx < 0:
            idx = seq_info.prompt_len + idx
        if 0 <= idx < seq_info.prompt_len:
            return ResolvedPosition(
                index=idx,
                token=seq_info.prompt_tokens[idx],
                source=f"prompt_index {pos.prompt_index}",
                is_prompt=True,
            )
        return None

    if pos.continuation_index is not None or pos.index is not None:
        # Index into continuation
        idx = pos.continuation_index if pos.continuation_index is not None else pos.index
        cont_len = len(seq_info.continuation_tokens)
        if idx < 0:
            idx = cont_len + idx
        if 0 <= idx < cont_len:
            abs_idx = seq_info.prompt_len + idx
            return ResolvedPosition(
                index=abs_idx,
                token=seq_info.continuation_tokens[idx],
                source=f"continuation_index {pos.continuation_index or pos.index}",
                is_prompt=False,
            )
        return None

    return None


def resolve_all_positions(
    positions: list[TokenPosition],
    seq_info: TokenSequenceInfo,
) -> list[ResolvedPosition]:
    """
    Resolve all token positions, skipping failures with warnings.

    Args:
        positions: List of TokenPosition specs
        seq_info: Token sequence information

    Returns:
        List of successfully resolved positions
    """
    resolved = []
    for pos in positions:
        result = resolve_token_position(pos, seq_info)
        if result is not None:
            resolved.append(result)
        else:
            # Build descriptive warning
            if pos.is_text_search():
                loc = "prompt" if pos.is_prompt_position() else "continuation"
                print(f"Warning: Could not find text '{pos.text}' in {loc}")
            else:
                idx = pos.get_index()
                if pos.prompt_index is not None:
                    print(f"Warning: prompt_index {idx} out of range (prompt has {seq_info.prompt_len} tokens)")
                else:
                    cont_len = len(seq_info.continuation_tokens)
                    print(f"Warning: continuation_index {idx} out of range (continuation has {cont_len} tokens)")

    return resolved


def parse_token_positions(raw_positions: list) -> list[TokenPosition]:
    """
    Parse raw JSON token positions into TokenPosition objects.

    Args:
        raw_positions: List of dicts/ints/strs from JSON

    Returns:
        List of TokenPosition objects
    """
    return [TokenPosition.from_dict(p) for p in raw_positions]
