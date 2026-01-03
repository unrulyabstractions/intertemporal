#!/usr/bin/env python
"""
End-to-end pipeline test for the intertemporal preference research framework.

Tests the complete workflow using modular schema-based approach:
1. Generate a minimal dataset using DatasetGenerator
2. Query model using ModelRunner
3. Train probes on activations
4. Run steering with trained probe directions

Uses schemas directly - no filesystem operations or subprocess calls.
Should complete in ~1-2 minutes.

Usage:
    # Run as script
    python tests/test_pipeline.py

    # Run via pytest
    pytest tests/test_pipeline.py -v -s
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from src.dataset_generator import DatasetGenerator
from src.common.schemas import (
    ContextConfig,
    DatasetConfig,
    FormattingConfig,
    OptionRangeConfig,
    StepType,
    DecodingConfig,
)
from src.types import TimeValue
from src.model_runner import ModelRunner
from sklearn.linear_model import LogisticRegression
from src.steering import SteeringConfig, SteeringOption

from common import (
    build_prompt_from_question,
    clear_memory,
    determine_choice,
    parse_label_from_response,
    QuestionOutput,
    OptionOutput,
    PreferencePairOutput,
)


def create_minimal_dataset_config() -> DatasetConfig:
    """Create a minimal dataset config for testing.

    Uses extreme value differences to ensure some choices go each way:
    - Some scenarios: short_term=$900 in 1 month vs long_term=$100 in 2 years (choose short)
    - Some scenarios: short_term=$100 in 1 month vs long_term=$5000 in 1 year (choose long)
    """
    return DatasetConfig(
        name="pipeline_test",
        context=ContextConfig(
            reward_unit="dollars",
            role="a decision maker",
            situation="Choose between options.",
            action_in_question="select",
            reasoning_ask="why you chose this",
            domain="test",
            labels=["a)", "b)"],
            method="grid",
            seed=42,
        ),
        options={
            "short_term": OptionRangeConfig(
                reward_range=(100, 900),
                time_range=(TimeValue(1, "months"), TimeValue(1, "months")),
                reward_steps=(3, StepType.LINEAR),
                time_steps=(0, StepType.LINEAR),
            ),
            "long_term": OptionRangeConfig(
                reward_range=(100, 5000),
                time_range=(TimeValue(1, "years"), TimeValue(2, "years")),
                reward_steps=(3, StepType.LINEAR),
                time_steps=(1, StepType.LINEAR),
            ),
        },
        time_horizons=[None],
        add_formatting_variations=False,
    )


def create_minimal_formatting_config() -> FormattingConfig:
    """Create a minimal formatting config for testing."""
    return FormattingConfig(
        question_template=(
            "SITUATION: [SITUATION]\n"
            "TASK: You, [ROLE], are tasked to decide to either [ACTION_IN_QUESTION]:\n"
            "[LEFT_TERM_LABEL] [LEFT_TERM_REWARD] [REWARD_UNITS] in [LEFT_TERM_TIME]\n"
            "[RIGHT_TERM_LABEL] [RIGHT_TERM_REWARD] [REWARD_UNITS] in [RIGHT_TERM_TIME]\n"
            "CONSIDER:Think deeply about WHICH option is preferable.\n\n"
            " ACTION:Select one of the two options, and provide reasoning on [REASONING_ASK]."
        ),
        response_format=(
            "\n\nRespond in this format:\n"
            "[CHOICE_PREFIX] <[LEFT_TERM_LABEL] or [RIGHT_TERM_LABEL]>.\n"
            " [REASONING_PREFIX] <reasoning in [MAX_REASONING_LENGTH]>"
        ),
        choice_prefix="I select:",
        reasoning_prefix="Reasoning:",
        time_horizon_spec="You are primarily concerned about outcome in [TIME_HORIZON].",
        max_reasoning_length="1-3 sentences",
    )


def test_full_pipeline():
    """Test the complete pipeline end-to-end using schemas."""
    print("\n" + "=" * 60)
    print("INTERTEMPORAL PREFERENCE PIPELINE TEST (Schema-based)")
    print("=" * 60)

    # Step 1: Generate dataset
    print("\nStep 1: Generate dataset...")
    dataset_config = create_minimal_dataset_config()
    formatting_config = create_minimal_formatting_config()

    generator = DatasetGenerator(dataset_config, formatting_config, seed=42)
    samples, metadata = generator.generate()

    print(f"  Generated {len(samples)} samples")
    assert len(samples) >= 10, f"Expected at least 10 samples, got {len(samples)}"

    # Convert samples to question format for querying
    questions = []
    for i, sample in enumerate(samples):
        st = sample.prompt.question.pair.short_term
        lt = sample.prompt.question.pair.long_term
        # Format time horizon as list if present
        th = sample.prompt.question.time_horizon
        time_horizon = [th.value, th.unit] if th else None
        question = QuestionOutput(
            sample_id=i,
            question_text=sample.prompt.text,
            preference_pair=PreferencePairOutput(
                short_term=OptionOutput(
                    label=st.label,
                    time=[st.time.value, st.time.unit],
                    reward=st.reward.value,
                ),
                long_term=OptionOutput(
                    label=lt.label,
                    time=[lt.time.value, lt.time.unit],
                    reward=lt.reward.value,
                ),
            ),
            time_horizon=time_horizon,
        )
        questions.append(question)

    print(f"  Converted to {len(questions)} questions")

    # Step 2: Query model
    print("\nStep 2: Query model...")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    decoding = DecodingConfig(
        max_new_tokens=50,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
    )

    # Use MPS if available, otherwise CPU
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  Loading model on {device}...")

    runner = ModelRunner(model_name, device=device)
    print(f"  Model loaded: {model_name}")
    print(f"  Layers: {runner.model.cfg.n_layers}, d_model: {runner.model.cfg.d_model}")

    # Query each sample and collect activations
    responses = []
    activations = []
    choices = []

    # Capture last layer residual stream at last token
    capture_layer = runner.model.cfg.n_layers - 1

    for i, question in enumerate(questions[:16]):  # Limit to 16 for speed
        prompt = build_prompt_from_question(question, formatting_config, model_name=model_name)

        # Run generation without internals first
        run_output = runner.run(prompt, decoding=decoding)
        response = run_output.response

        # Now get activations via run_with_cache on the prompt
        formatted = runner._apply_chat_template(prompt)
        logits, cache = runner.run_with_cache(formatted)

        # Extract last token activation from last layer
        hook_name = f"blocks.{capture_layer}.hook_resid_post"
        if hook_name in cache:
            act = cache[hook_name][0, -1, :].detach().cpu()  # Last token
            activations.append(act)

        # Parse response
        labels = [question.preference_pair.short_term.label, question.preference_pair.long_term.label]
        chosen_label = parse_label_from_response(
            response, labels, formatting_config.choice_prefix, model_name
        )
        choice = determine_choice(
            chosen_label,
            question.preference_pair.short_term.label,
            question.preference_pair.long_term.label,
        )

        responses.append(response)
        choices.append(choice)

        if i == 0:
            print(f"  Sample response: {response[:100]}...")

    print(f"  Queried {len(responses)} samples")
    print(f"  Choices: {choices.count('short_term')} short, {choices.count('long_term')} long, {choices.count('unknown')} unknown")

    # Clean up memory
    clear_memory()

    # Step 3: Train probe
    print("\nStep 3: Train probe...")

    # Create training data (X: activations, y: binary labels)
    valid_indices = [i for i, c in enumerate(choices) if c in ("short_term", "long_term")]
    if len(valid_indices) < 4:
        print(f"  WARNING: Only {len(valid_indices)} valid samples, skipping probe training")
        return True

    X = torch.stack([activations[i] for i in valid_indices])
    y = torch.tensor([1 if choices[i] == "long_term" else 0 for i in valid_indices])

    print(f"  Training data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Labels: {(y == 0).sum().item()} short_term, {(y == 1).sum().item()} long_term")

    # Simple train/test split
    n_train = int(0.7 * len(valid_indices))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # Check we have both classes in training set
    if len(y_train.unique()) < 2:
        print("  WARNING: Only one class in training set - skipping probe training/steering")
        print("  (Model consistently chose one option - this can happen with small models)")

        # Clean up
        del runner
        clear_memory()

        print("\n" + "=" * 60)
        print("PIPELINE TEST COMPLETED (partial - single class)")
        print("=" * 60)
        print(f"Dataset samples: {len(samples)}")
        print(f"Samples queried: {len(responses)}")
        print("Note: Probe training skipped due to single-class data")
        return True

    # Train a simple linear probe using sklearn
    probe = LogisticRegression(max_iter=1000, C=1.0)
    probe.fit(X_train.numpy(), y_train.numpy())
    print(f"  Probe trained (LogisticRegression)")

    # Test probe
    preds = probe.predict(X_test.numpy())
    accuracy = (preds == y_test.numpy()).mean()
    print(f"  Test accuracy: {accuracy:.2%}")

    # Step 4: Steering test
    print("\nStep 4: Steering test...")

    # Get probe direction (for logistic regression, it's the coefficients)
    probe_direction = probe.coef_[0]  # numpy array
    print(f"  Probe direction shape: {probe_direction.shape}")

    # Test steering on a single sample
    test_question = questions[0]
    test_prompt = build_prompt_from_question(test_question, formatting_config, model_name=model_name)

    # Baseline (no steering)
    baseline_output = runner.run(
        test_prompt,
        decoding=decoding,
    )
    baseline_response = baseline_output.response
    baseline_labels = [test_question.preference_pair.short_term.label, test_question.preference_pair.long_term.label]
    baseline_label = parse_label_from_response(baseline_response, baseline_labels, formatting_config.choice_prefix, model_name)
    baseline_choice = determine_choice(
        baseline_label,
        test_question.preference_pair.short_term.label,
        test_question.preference_pair.long_term.label,
    )
    print(f"  Baseline choice: {baseline_choice}")

    # Steered response (apply steering vector)
    steering_strength = 50.0
    steering_config = SteeringConfig(
        layer=capture_layer,
        direction=probe_direction,
        strength=steering_strength,
        option=SteeringOption.APPLY_TO_ALL,
    )

    steered_response = runner.generate_with_steering(
        test_prompt,
        steering_config,
        max_new_tokens=decoding.max_new_tokens,
    )
    steered_label = parse_label_from_response(steered_response, baseline_labels, formatting_config.choice_prefix, model_name)
    steered_choice = determine_choice(
        steered_label,
        test_question.preference_pair.short_term.label,
        test_question.preference_pair.long_term.label,
    )
    print(f"  Steered choice (strength={steering_strength}): {steered_choice}")

    flipped = baseline_choice != steered_choice and baseline_choice != "unknown" and steered_choice != "unknown"
    print(f"  Choice flipped: {flipped}")

    # Clean up
    del runner
    clear_memory()

    print("\n" + "=" * 60)
    print("PIPELINE TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Dataset samples: {len(samples)}")
    print(f"Samples queried: {len(responses)}")
    print(f"Probe accuracy: {accuracy:.2%}")
    print(f"Steering tested: {'yes' if steered_choice != 'unknown' else 'no'}")

    return True


if __name__ == "__main__":
    try:
        success = test_full_pipeline()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nPIPELINE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
