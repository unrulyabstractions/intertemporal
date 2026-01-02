"""
Model runner using TransformerLens for inference and internals capture.

Handles model loading, inference, extraction of internal activations,
and activation steering for behavior modification.

Key features:
- Model loading with automatic device/dtype detection
- Text generation with configurable decoding
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

    response, _ = runner.run(prompt, steering=config)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import torch
from transformer_lens import HookedTransformer

from .common.schemas import DecodingConfig, InternalsConfig, TokenPosition
from .steering import SteeringConfig, create_steering_hook

# Add scripts to path for common imports
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common.utils import extract_flip_tokens


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


    def run(
        self,
        prompt: str,
        decoding: Optional[DecodingConfig] = None,
        internals_config: Optional[InternalsConfig] = None,
        marker_text: Optional[str] = None,
        steering: Optional[SteeringConfig] = None,
    ) -> tuple[str, Optional[CapturedInternals]]:
        """
        Run inference and optionally capture internals.

        Args:
            prompt: Text prompt
            decoding: Decoding configuration (uses defaults if None)
            internals_config: Configuration for internals capture (None = no capture)
            marker_text: Reference text in the prompt for token position resolution.
                        When capturing internals, some token positions may be specified
                        relative to this marker (e.g., "after the time horizon specification").
                        Pass the actual text that appears in the prompt so positions can
                        be resolved correctly. Example: "within the next 2 years"
            steering: Optional steering configuration to modify model behavior.
                     When provided, applies the steering direction vector to activations
                     at the specified layer during generation.

        Returns:
            Tuple of (generated_text, captured_internals or None)
        """
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

        return response_text, captured

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

    def get_next_token_probs(
        self,
        prompt: str,
        target_tokens: list[str],
    ) -> dict[str, float]:
        """
        Get probabilities for specific tokens at the next token position.

        Args:
            prompt: Text prompt (will have chat template applied if instruct model)
            target_tokens: List of token strings to get probabilities for

        Returns:
            Dict mapping token string to probability
        """
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

        return result

    def get_label_probs(
        self,
        prompt: str,
        labels: tuple[str, str],
    ) -> tuple[float, float]:
        """
        Get probabilities for two label options.

        Extracts the distinguishing "flip" token from each label and
        gets probabilities for those tokens with case variations.

        Args:
            prompt: Text prompt
            labels: Tuple of (label1, label2) strings

        Returns:
            Tuple of (prob1, prob2) for the two labels
        """
        # Extract the distinguishing flip tokens
        flip1, flip2 = extract_flip_tokens(labels)

        # Generate case and space variations for each flip token
        def get_variations(token: str) -> list[str]:
            """Get case and leading space variations for a token."""
            # Base case variations
            base_vars = [token, token.upper(), token.lower()]
            # Add leading space variations (models often predict " A" not "A")
            space_vars = [" " + v for v in base_vars]
            variations = base_vars + space_vars
            # Remove duplicates while preserving order
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
        probs = self.get_next_token_probs(prompt, all_tokens)

        # Take max probability across variations
        prob1 = max(probs.get(v, 0.0) for v in flip1_vars)
        prob2 = max(probs.get(v, 0.0) for v in flip2_vars)

        return prob1, prob2

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

    def generate_with_steering(
        self,
        prompt: str,
        steering: SteeringConfig,
        max_new_tokens: int = 64,
    ) -> str:
        """
        Generate text with activation steering applied.

        Convenience method for steering without internals capture.
        Uses greedy decoding (temperature=0).

        Args:
            prompt: Text prompt
            steering: Steering configuration (direction, layer, strength, option)
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated text (continuation only, excluding prompt)
        """
        decoding = DecodingConfig(max_new_tokens=max_new_tokens, temperature=0.0)
        response, _ = self.run(prompt, decoding=decoding, steering=steering)
        return response

    def generate_baseline(
        self,
        prompt: str,
        max_new_tokens: int = 64,
    ) -> str:
        """
        Generate text without steering (baseline).

        Convenience method for baseline generation.
        Uses greedy decoding (temperature=0).

        Args:
            prompt: Text prompt
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated text (continuation only, excluding prompt)
        """
        decoding = DecodingConfig(max_new_tokens=max_new_tokens, temperature=0.0)
        response, _ = self.run(prompt, decoding=decoding)
        return response
