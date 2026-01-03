"""
Model runner using TransformerLens for inference and internals capture.

Handles model loading, inference, extraction of internal activations,
and activation steering for behavior modification.

Key features:
- Model loading with automatic device/dtype detection
- Text generation with configurable decoding
- Batch processing support for all methods
- Internals capture at specified token positions
- Activation steering during generation

Example with steering:
    from src.steering import SteeringConfig, SteeringOption

    runner = ModelRunner("Qwen/Qwen2.5-7B-Instruct")

    config = SteeringConfig(
        direction=probe.direction,
        layer=26,
        strength=100.0,
        option=SteeringOption.APPLY_TO_ALL,
    )

    response = runner.run(prompt, steering=config)

Example with batching:
    prompts = ["Question 1?", "Question 2?", "Question 3?"]
    responses = runner.run(prompts, batch_size=2)  # Returns list[RunOutput]
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, overload

import numpy as np
import torch
from transformer_lens import HookedTransformer

from .common.schemas import DecodingConfig, InternalsConfig, TokenPosition
from .steering import SteeringConfig, create_steering_hook

# Add scripts to path for common imports
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common.utils import extract_flip_tokens


# =============================================================================
# Output Schemas
# =============================================================================


@dataclass
class CapturedInternals:
    """
    Captured internal activations from a forward pass.

    Attributes:
        activations: Dict mapping "activation_name_layer" to tensor values
                    at specified token positions.
        token_positions: Resolved token position indices used.
        tokens: The token strings at each position.
    """
    activations: dict = field(default_factory=dict)
    token_positions: list = field(default_factory=list)
    tokens: list = field(default_factory=list)


@dataclass
class RunOutput:
    """Output from a single run() call."""
    response: str
    internals: Optional[CapturedInternals] = None


@dataclass
class LabelProbsOutput:
    """Output from get_label_probs()."""
    prob1: float
    prob2: float


@dataclass
class NextTokenProbsOutput:
    """Output from get_next_token_probs()."""
    probs: dict[str, float]


class ModelRunner:
    """
    Model runner using TransformerLens.

    Handles model loading, tokenization, inference, and internals capture.
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize model runner.

        Args:
            model_name: HuggingFace model name or TransformerLens model name
            device: Device to use (auto-detected if None)
            dtype: Data type for model (auto-detected if None)
        """
        self.model_name = model_name

        # Auto-detect device
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device

        # Auto-detect dtype
        if dtype is None:
            if device in ["mps", "cuda"]:
                dtype = torch.float16
            else:
                dtype = torch.float32
        self.dtype = dtype

        # Load model with TransformerLens
        print(f"Loading model {model_name} on {device}...")
        self.model = HookedTransformer.from_pretrained(
            model_name,
            device=device,
            dtype=dtype,
        )
        self.model.eval()

        self._is_chat_model = self._detect_chat_model()
        print(f"Model loaded: {model_name} (chat_model={self._is_chat_model})")
        print(f"  n_layers={self.model.cfg.n_layers}, d_model={self.model.cfg.d_model}\n")

    def _detect_chat_model(self) -> bool:
        """Detect if model is a chat/instruct model based on name."""
        name_lower = self.model_name.lower()
        chat_indicators = ["instruct", "chat", "-it", "rlhf"]
        return any(ind in name_lower for ind in chat_indicators)

    def _apply_chat_template(self, prompt: str) -> str:
        """
        Apply chat template for instruct models.

        Args:
            prompt: Raw prompt text

        Returns:
            Formatted prompt with chat template
        """
        if not self._is_chat_model:
            return prompt

        # Try to use the tokenizer's chat template if available
        tokenizer = self.model.tokenizer
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            try:
                formatted = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return formatted
            except Exception:
                pass  # Fall through to manual template

        # Fallback: simple user message format
        return f"<|user|>\n{prompt}\n<|assistant|>\n"

    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text to token IDs."""
        return self.model.to_tokens(text, prepend_bos=True)

    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs to text."""
        return self.model.to_string(token_ids)

    def get_token_strs(self, token_ids: torch.Tensor) -> list[str]:
        """Get string representation of each token."""
        return self.model.to_str_tokens(token_ids)


    @overload
    def run(
        self,
        prompt: str,
        decoding: Optional[DecodingConfig] = None,
        internals_config: Optional[InternalsConfig] = None,
        marker_text: Optional[str] = None,
        steering: Optional[SteeringConfig] = None,
    ) -> RunOutput: ...

    @overload
    def run(
        self,
        prompt: list[str],
        decoding: Optional[DecodingConfig] = None,
        internals_config: Optional[InternalsConfig] = None,
        marker_text: Optional[str | list[str]] = None,
        steering: Optional[SteeringConfig] = None,
        batch_size: int = 8,
    ) -> list[RunOutput]: ...

    def run(
        self,
        prompt: str | list[str],
        decoding: Optional[DecodingConfig] = None,
        internals_config: Optional[InternalsConfig] = None,
        marker_text: Optional[str | list[str]] = None,
        steering: Optional[SteeringConfig] = None,
        batch_size: int = 8,
    ) -> RunOutput | list[RunOutput]:
        """
        Run inference and optionally capture internals.

        Supports both single prompts and batched processing.

        Args:
            prompt: Text prompt or list of prompts for batch processing
            decoding: Decoding configuration (uses defaults if None)
            internals_config: Configuration for internals capture (None = no capture)
            marker_text: Reference text(s) for token position resolution.
                        For batch, can be single string (used for all) or list matching prompts.
            steering: Optional steering configuration to modify model behavior.
            batch_size: Batch size for processing multiple prompts (default 8)

        Returns:
            RunOutput for single prompt, list[RunOutput] for batch
        """
        # Handle single prompt
        if isinstance(prompt, str):
            return self._run_single(prompt, decoding, internals_config, marker_text, steering)

        # Handle batch
        results = []
        for i in range(0, len(prompt), batch_size):
            batch_prompts = prompt[i:i + batch_size]
            # Handle marker_text for batch
            if marker_text is None:
                batch_markers = [None] * len(batch_prompts)
            elif isinstance(marker_text, str):
                batch_markers = [marker_text] * len(batch_prompts)
            else:
                batch_markers = marker_text[i:i + batch_size]

            # Process batch one at a time (TransformerLens generate doesn't support true batching well)
            for p, m in zip(batch_prompts, batch_markers):
                results.append(self._run_single(p, decoding, internals_config, m, steering))

        return results

    def _run_single(
        self,
        prompt: str,
        decoding: Optional[DecodingConfig] = None,
        internals_config: Optional[InternalsConfig] = None,
        marker_text: Optional[str] = None,
        steering: Optional[SteeringConfig] = None,
    ) -> RunOutput:
        """Run inference for a single prompt."""
        if decoding is None:
            decoding = DecodingConfig()

        # Apply chat template for instruct models
        formatted_prompt = self._apply_chat_template(prompt)

        # Tokenize prompt
        input_ids = self.tokenize(formatted_prompt)
        prompt_len = input_ids.shape[1]

        # Build generation kwargs
        do_sample = decoding.temperature > 0
        gen_kwargs = {
            "max_new_tokens": decoding.max_new_tokens,
            "do_sample": do_sample,
            "stop_at_eos": True,
            "verbose": False,  # Disable tqdm progress bars
        }
        if do_sample:
            gen_kwargs["temperature"] = decoding.temperature
            if decoding.top_k > 0:
                gen_kwargs["top_k"] = decoding.top_k
            if decoding.top_p < 1.0:
                gen_kwargs["top_p"] = decoding.top_p

        # Generate (with optional steering)
        with torch.no_grad():
            if steering is not None:
                # Create steering hook
                hook, pattern_matcher = create_steering_hook(
                    steering,
                    model_dtype=self.dtype,
                    model_device=self.device,
                    tokenizer=self.model.tokenizer,
                )
                # Apply steering during generation
                with self.model.hooks(fwd_hooks=[(steering.hook_name, hook)]):
                    output_ids = self.model.generate(input_ids, **gen_kwargs)
            else:
                output_ids = self.model.generate(input_ids, **gen_kwargs)

        # Decode response (excluding prompt)
        generated_ids = output_ids[0, prompt_len:]
        response_text = self.decode(generated_ids)

        # Capture internals if requested
        captured = None
        if internals_config and (internals_config.activations or internals_config.token_positions):
            captured = self._capture_internals(
                output_ids,
                internals_config,
                prompt_len,
                marker_text,
            )

        return RunOutput(response=response_text, internals=captured)

    def _capture_internals(
        self,
        token_ids: torch.Tensor,
        config: InternalsConfig,
        prompt_len: int,
        marker_text: Optional[str] = None,
    ) -> CapturedInternals:
        """
        Capture internal activations at specified positions.

        Args:
            token_ids: Full sequence token IDs
            config: Internals configuration
            prompt_len: Length of the prompt (for position reference)
            marker_text: Reference text for position resolution. When provided,
                        positions with after_time_horizon_spec=True are resolved
                        relative to where this text ends in the prompt.

        Returns:
            CapturedInternals with requested activations.
            token_positions and tokens arrays always match config length,
            with -1 and "<UNRESOLVED>" as placeholders for unresolved positions.
        """
        from common.token_positions import (
            TokenSequenceInfo,
            parse_token_positions,
            resolve_token_position,
            find_text_in_tokens,
        )

        # Get token strings
        token_strs = self.get_token_strs(token_ids[0])
        prompt_tokens = token_strs[:prompt_len]
        continuation_tokens = token_strs[prompt_len:]

        # Find where marker text ends in prompt (if present)
        # This is used for resolving positions specified as "after_time_horizon_spec"
        time_horizon_spec_end = -1
        if marker_text:
            time_horizon_spec_end = find_text_in_tokens(
                marker_text, prompt_tokens, offset=0
            )
            if time_horizon_spec_end is None:
                time_horizon_spec_end = -1

        # Build sequence info for resolution
        seq_info = TokenSequenceInfo(
            prompt_tokens=prompt_tokens,
            continuation_tokens=continuation_tokens,
            prompt_len=prompt_len,
            total_len=len(token_strs),
            time_horizon_spec_end=time_horizon_spec_end,
        )

        # Parse token positions
        token_positions = parse_token_positions(config.token_positions)
        n_positions = len(token_positions)

        # Initialize with placeholders - arrays always match config length
        resolved_indices = [-1] * n_positions
        position_tokens = ["<UNRESOLVED>"] * n_positions

        # Resolve each position individually, keeping index alignment
        for i, pos in enumerate(token_positions):
            result = resolve_token_position(pos, seq_info)
            if result is not None:
                resolved_indices[i] = result.index
                position_tokens[i] = result.token
            else:
                # Log warning for unresolved positions
                if pos.is_text_search():
                    loc = "prompt" if pos.is_prompt_position() else "continuation"
                    print(f"Warning: Could not find text '{pos.text}' in {loc}")
                    # Print the relevant text for debugging
                    if pos.is_prompt_position():
                        full_text = "".join(prompt_tokens)
                        print(f"  Full prompt: {full_text[:500]}{'...' if len(full_text) > 500 else ''}")
                    else:
                        full_text = "".join(continuation_tokens)
                        print(f"  Full continuation: {full_text}")
                else:
                    idx = pos.get_index()
                    if pos.prompt_index is not None:
                        print(f"Warning: prompt_index {idx} out of range (prompt has {seq_info.prompt_len} tokens)")
                    else:
                        cont_len = len(seq_info.continuation_tokens)
                        print(f"Warning: continuation_index {idx} out of range (continuation has {cont_len} tokens)")

        # Build activation names to capture
        activation_names = []
        for act_type, spec in config.activations.items():
            layers = spec.get("layers", [])
            for layer in layers:
                # Handle negative layer indices
                if layer < 0:
                    layer = self.model.cfg.n_layers + layer
                activation_names.append(f"blocks.{layer}.hook_{act_type}")

        if not activation_names:
            return CapturedInternals(
                activations={},
                token_positions=resolved_indices,
                tokens=position_tokens,
            )

        # Run forward pass with hooks to capture activations
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                token_ids,
                names_filter=lambda name: any(act in name for act in activation_names),
            )

        # Extract activations only at successfully resolved positions
        activations = {}
        for name in activation_names:
            if name in cache:
                act = cache[name]  # Shape: [batch, seq, d_model]
                # Extract at each resolved position (skip -1 placeholders)
                for pos in resolved_indices:
                    if pos >= 0 and pos < act.shape[1]:
                        key = f"{name}_pos{pos}"
                        activations[key] = act[0, pos, :].cpu().numpy().tolist()

        return CapturedInternals(
            activations=activations,
            token_positions=resolved_indices,
            tokens=position_tokens,
        )

    def run_with_cache(
        self,
        prompt: str,
        names_filter: Optional[callable] = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Run forward pass and return full activation cache.

        Args:
            prompt: Text prompt
            names_filter: Optional filter for which activations to cache

        Returns:
            Tuple of (logits, activation_cache)
        """
        input_ids = self.tokenize(prompt)
        with torch.no_grad():
            logits, cache = self.model.run_with_cache(
                input_ids,
                names_filter=names_filter,
            )
        return logits, cache

    @property
    def n_layers(self) -> int:
        """Number of transformer layers."""
        return self.model.cfg.n_layers

    @property
    def d_model(self) -> int:
        """Model hidden dimension."""
        return self.model.cfg.d_model

    @property
    def vocab_size(self) -> int:
        """Vocabulary size."""
        return self.model.cfg.d_vocab

    @overload
    def get_next_token_probs(
        self,
        prompt: str,
        target_tokens: list[str],
    ) -> NextTokenProbsOutput: ...

    @overload
    def get_next_token_probs(
        self,
        prompt: list[str],
        target_tokens: list[str],
        batch_size: int = 8,
    ) -> list[NextTokenProbsOutput]: ...

    def get_next_token_probs(
        self,
        prompt: str | list[str],
        target_tokens: list[str],
        batch_size: int = 8,
    ) -> NextTokenProbsOutput | list[NextTokenProbsOutput]:
        """
        Get probabilities for specific tokens at the next token position.

        Args:
            prompt: Text prompt or list of prompts
            target_tokens: List of token strings to get probabilities for
            batch_size: Batch size for processing multiple prompts

        Returns:
            NextTokenProbsOutput for single, list for batch
        """
        if isinstance(prompt, str):
            return self._get_next_token_probs_single(prompt, target_tokens)

        results = []
        for i in range(0, len(prompt), batch_size):
            batch = prompt[i:i + batch_size]
            for p in batch:
                results.append(self._get_next_token_probs_single(p, target_tokens))
        return results

    def _get_next_token_probs_single(
        self,
        prompt: str,
        target_tokens: list[str],
    ) -> NextTokenProbsOutput:
        """Get next token probabilities for a single prompt."""
        # Apply chat template for instruct models
        formatted_prompt = self._apply_chat_template(prompt)

        # Tokenize prompt
        input_ids = self.tokenize(formatted_prompt)

        # Get logits
        with torch.no_grad():
            logits = self.model(input_ids)  # [batch, seq, vocab]

        # Get logits at last position (next token prediction)
        next_logits = logits[0, -1, :]  # [vocab]

        # Convert to probabilities
        probs = torch.softmax(next_logits, dim=-1)

        # Get probabilities for target tokens
        result = {}
        for token_str in target_tokens:
            # Try to find token ID - handle variations
            token_ids = self.model.tokenizer.encode(token_str, add_special_tokens=False)
            if token_ids:
                # Use first token if multi-token
                token_id = token_ids[0]
                result[token_str] = probs[token_id].item()
            else:
                result[token_str] = 0.0

        return NextTokenProbsOutput(probs=result)

    @overload
    def get_label_probs(
        self,
        prompt: str,
        labels: tuple[str, str],
    ) -> LabelProbsOutput: ...

    @overload
    def get_label_probs(
        self,
        prompt: list[str],
        labels: tuple[str, str] | list[tuple[str, str]],
        batch_size: int = 8,
    ) -> list[LabelProbsOutput]: ...

    def get_label_probs(
        self,
        prompt: str | list[str],
        labels: tuple[str, str] | list[tuple[str, str]],
        batch_size: int = 8,
    ) -> LabelProbsOutput | list[LabelProbsOutput]:
        """
        Get probabilities for two label options.

        Extracts the distinguishing "flip" token from each label and
        gets probabilities for those tokens with case variations.

        Args:
            prompt: Text prompt or list of prompts
            labels: Tuple of (label1, label2) or list of tuples for batch
            batch_size: Batch size for processing multiple prompts

        Returns:
            LabelProbsOutput for single, list for batch
        """
        if isinstance(prompt, str):
            return self._get_label_probs_single(prompt, labels)

        # Handle batch
        results = []
        for i in range(0, len(prompt), batch_size):
            batch_prompts = prompt[i:i + batch_size]
            if isinstance(labels, list):
                batch_labels = labels[i:i + batch_size]
            else:
                batch_labels = [labels] * len(batch_prompts)

            for p, lbl in zip(batch_prompts, batch_labels):
                results.append(self._get_label_probs_single(p, lbl))
        return results

    def _get_label_probs_single(
        self,
        prompt: str,
        labels: tuple[str, str],
    ) -> LabelProbsOutput:
        """Get label probabilities for a single prompt."""
        # Extract the distinguishing flip tokens
        flip1, flip2 = extract_flip_tokens(labels)

        # Generate case and space variations for each flip token
        def get_variations(token: str) -> list[str]:
            """Get case and leading space variations for a token."""
            base_vars = [token, token.upper(), token.lower()]
            # Also add first-char variations for multi-char tokens
            # (tokenizers often encode " F" differently than first token of " FIRST")
            if len(token) > 1:
                first_char = token[0]
                base_vars.extend([first_char, first_char.upper(), first_char.lower()])
            space_vars = [" " + v for v in base_vars]
            variations = base_vars + space_vars
            seen = set()
            unique = []
            for v in variations:
                if v not in seen:
                    seen.add(v)
                    unique.append(v)
            return unique

        flip1_vars = get_variations(flip1)
        flip2_vars = get_variations(flip2)

        # Get all probabilities
        all_tokens = flip1_vars + flip2_vars
        probs_output = self._get_next_token_probs_single(prompt, all_tokens)
        probs = probs_output.probs

        # Take max probability across variations
        prob1 = max(probs.get(v, 0.0) for v in flip1_vars)
        prob2 = max(probs.get(v, 0.0) for v in flip2_vars)

        return LabelProbsOutput(prob1=prob1, prob2=prob2)

    def get_first_continuation_position(self, prompt: str) -> int:
        """
        Get the token position where model continuation begins.

        This is the index of the first token the model would generate,
        which equals the prompt length in tokens.

        Args:
            prompt: Text prompt (chat template will be applied if instruct model)

        Returns:
            Token index of first continuation position
        """
        formatted_prompt = self._apply_chat_template(prompt)
        tokens = self.tokenize(formatted_prompt)
        return tokens.shape[1]

    @overload
    def generate_with_steering(
        self,
        prompt: str,
        steering: SteeringConfig,
        max_new_tokens: int = 64,
    ) -> str: ...

    @overload
    def generate_with_steering(
        self,
        prompt: list[str],
        steering: SteeringConfig,
        max_new_tokens: int = 64,
        batch_size: int = 8,
    ) -> list[str]: ...

    def generate_with_steering(
        self,
        prompt: str | list[str],
        steering: SteeringConfig,
        max_new_tokens: int = 64,
        batch_size: int = 8,
    ) -> str | list[str]:
        """
        Generate text with activation steering applied.

        Convenience method for steering without internals capture.
        Uses greedy decoding (temperature=0).

        Args:
            prompt: Text prompt or list of prompts
            steering: Steering configuration (direction, layer, strength, option)
            max_new_tokens: Maximum tokens to generate
            batch_size: Batch size for processing multiple prompts

        Returns:
            Generated text for single, list of texts for batch
        """
        decoding = DecodingConfig(max_new_tokens=max_new_tokens, temperature=0.0)
        result = self.run(prompt, decoding=decoding, steering=steering, batch_size=batch_size)

        if isinstance(result, list):
            return [r.response for r in result]
        return result.response

    @overload
    def generate_baseline(
        self,
        prompt: str,
        max_new_tokens: int = 64,
    ) -> str: ...

    @overload
    def generate_baseline(
        self,
        prompt: list[str],
        max_new_tokens: int = 64,
        batch_size: int = 8,
    ) -> list[str]: ...

    def generate_baseline(
        self,
        prompt: str | list[str],
        max_new_tokens: int = 64,
        batch_size: int = 8,
    ) -> str | list[str]:
        """
        Generate text without steering (baseline).

        Convenience method for baseline generation.
        Uses greedy decoding (temperature=0).

        Args:
            prompt: Text prompt or list of prompts
            max_new_tokens: Maximum tokens to generate
            batch_size: Batch size for processing multiple prompts

        Returns:
            Generated text for single, list of texts for batch
        """
        decoding = DecodingConfig(max_new_tokens=max_new_tokens, temperature=0.0)
        result = self.run(prompt, decoding=decoding, batch_size=batch_size)

        if isinstance(result, list):
            return [r.response for r in result]
        return result.response

    def get_label_probs_with_steering(
        self,
        prompt: str,
        labels: tuple[str, str],
        steering: SteeringConfig,
        choice_prefix: str = "I select:",
    ) -> LabelProbsOutput:
        """
        Get label probabilities with steering applied.

        Runs a forward pass with the steering hook active to get probabilities
        that reflect what the steered model would predict.

        Args:
            prompt: Text prompt (user message content)
            labels: Tuple of (label1, label2) e.g. ("a)", "b)")
            steering: Steering configuration
            choice_prefix: Prefix for the choice (added to assistant turn)

        Returns:
            LabelProbsOutput with prob1 and prob2
        """
        from src.steering import create_steering_hook

        # Extract the distinguishing flip tokens (same as get_label_probs)
        flip1, flip2 = extract_flip_tokens(labels)

        # Generate case and space variations for each flip token
        def get_variations(token: str) -> list[str]:
            """Get case and leading space variations for a token."""
            base_vars = [token, token.upper(), token.lower()]
            # Also add first-char variations for multi-char tokens
            # (tokenizers often encode " F" differently than first token of " FIRST")
            if len(token) > 1:
                first_char = token[0]
                base_vars.extend([first_char, first_char.upper(), first_char.lower()])
            space_vars = [" " + v for v in base_vars]
            variations = base_vars + space_vars
            seen = set()
            unique = []
            for v in variations:
                if v not in seen:
                    seen.add(v)
                    unique.append(v)
            return unique

        flip1_vars = get_variations(flip1)
        flip2_vars = get_variations(flip2)
        all_tokens = flip1_vars + flip2_vars

        # Find common prefix of labels (e.g., "(" for "(A)" and "(B)")
        label1, label2 = labels
        common_prefix = ""
        for c1, c2 in zip(label1, label2):
            if c1 == c2:
                common_prefix += c1
            else:
                break

        # Format prompt with choice_prefix as partial assistant response
        if self._is_chat_model and hasattr(self.model.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.model.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # Add choice_prefix, space, and common prefix so next token is the distinguishing char
            formatted_prompt = formatted_prompt + choice_prefix + " " + common_prefix
        else:
            formatted_prompt = prompt + "\n" + choice_prefix + " " + common_prefix

        # Tokenize prompt
        input_ids = self.tokenize(formatted_prompt)

        # Create steering hook
        hook_fn, _ = create_steering_hook(
            steering,
            model_dtype=self.model.cfg.dtype,
            model_device=str(self.model.cfg.device),
            tokenizer=self.model.tokenizer,
        )

        # Hook name for the target layer
        hook_name = f"blocks.{steering.layer}.hook_resid_post"

        # Run forward pass with hook
        with torch.no_grad():
            logits = self.model.run_with_hooks(
                input_ids,
                fwd_hooks=[(hook_name, hook_fn)],
            )

        # Get logits at last position (next token prediction)
        next_logits = logits[0, -1, :]  # [vocab]

        # Convert to probabilities
        probs_tensor = torch.softmax(next_logits, dim=-1)

        # Get probabilities for all token variations
        probs = {}
        for token_str in all_tokens:
            token_ids = self.model.tokenizer.encode(token_str, add_special_tokens=False)
            if token_ids:
                token_id = token_ids[0]
                probs[token_str] = probs_tensor[token_id].item()
            else:
                probs[token_str] = 0.0

        # Take max probability across variations
        prob1 = max(probs.get(v, 0.0) for v in flip1_vars)
        prob2 = max(probs.get(v, 0.0) for v in flip2_vars)

        return LabelProbsOutput(prob1=prob1, prob2=prob2)

    def get_activation_with_steering(
        self,
        prompt: str,
        layer: int,
        position: int,
        steering: SteeringConfig,
    ) -> Optional[np.ndarray]:
        """
        Get activation at a specific layer/position after applying steering.

        Runs forward pass with steering hook and captures activation.
        This is for single-prompt use (not batched).

        Args:
            prompt: Single text prompt (not a list)
            layer: Layer to extract activation from
            position: Token position to extract activation from
            steering: Steering configuration

        Returns:
            Activation array of shape (d_model,), or None if position out of bounds
        """
        from src.steering import create_steering_hook

        # Format prompt
        formatted_prompt = self._apply_chat_template(prompt)
        input_ids = self.tokenize(formatted_prompt)
        seq_len = input_ids.shape[1]

        # Validate position
        if position >= seq_len:
            return None

        # Create steering hook
        hook_fn, _ = create_steering_hook(
            steering,
            model_dtype=self.model.cfg.dtype,
            model_device=str(self.model.cfg.device),
            tokenizer=self.model.tokenizer,
        )

        # Hook to capture activation at specified position
        captured_activation = [None]
        capture_hook_name = f"blocks.{layer}.hook_resid_post"

        def capture_hook(activation: torch.Tensor, hook=None) -> torch.Tensor:
            # activation shape: [batch, seq_len, d_model]
            captured_activation[0] = activation[0, position, :].detach().cpu().numpy()
            return activation

        # Steering hook name
        steering_hook_name = f"blocks.{steering.layer}.hook_resid_post"

        # Build hook list
        hooks = []
        if steering.layer == layer:
            # Same layer: combine hooks (steering first, then capture)
            def combined_hook(activation: torch.Tensor, hook=None) -> torch.Tensor:
                activation = hook_fn(activation, hook)
                captured_activation[0] = activation[0, position, :].detach().cpu().numpy()
                return activation
            hooks.append((steering_hook_name, combined_hook))
        else:
            # Different layers: separate hooks
            hooks.append((steering_hook_name, hook_fn))
            hooks.append((capture_hook_name, capture_hook))

        # Run forward pass with hooks
        with torch.no_grad():
            self.model.run_with_hooks(input_ids, fwd_hooks=hooks)

        if captured_activation[0] is None:
            return None

        return np.asarray(captured_activation[0])
